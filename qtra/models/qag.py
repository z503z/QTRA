"""QTRA paper models. Extracted from the development implementation.

Legacy class/parameter names are retained for checkpoint compatibility.
Unused experimental architectures are intentionally omitted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .x3d_change3d import create_x3d
from .clip import build_model
from .common import SimpleArgs

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def create_qag_x3d(args):
    return create_x3d(
        input_clip_length=3,
        depth_factor=getattr(args, "x3d_depth_factor", 5.0),
    )


def load_qag_x3d_pretrained(x3d, args):
    if args.pretrained is None:
        return

    try:
        state = torch.load(args.pretrained, map_location="cpu")
        state_dict = state["model_state"] if "model_state" in state else state
        depth_factor = getattr(args, "x3d_depth_factor", 5.0)
        if abs(depth_factor - 5.0) < 1e-6:
            msg = x3d.load_state_dict(state_dict, strict=True)
            print(f"[Change3D] loaded pretrained: {args.pretrained}, {msg}")
            return

        model_state = x3d.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        msg = x3d.load_state_dict(compatible, strict=False)
        skipped = len(state_dict) - len(compatible)
        print(
            f"[Change3D] partially loaded pretrained: {args.pretrained}, "
            f"matched={len(compatible)}, skipped={skipped}, {msg}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load X3D initialization: {args.pretrained}") from e


class ProgressiveMaskDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=(128, 96, 64, 32, 16)):
        super().__init__()
        channels = [in_channels] + list(hidden_channels)
        self.stages = nn.ModuleList(
            ConvBNReLU(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        )
        self.pred = nn.Conv2d(channels[-1], 1, kernel_size=1)

    def forward(self, x, output_size):
        for stage in self.stages:
            if x.shape[-2:] != output_size:
                next_h = min(x.shape[-2] * 2, output_size[0])
                next_w = min(x.shape[-1] * 2, output_size[1])
                x = F.interpolate(
                    x,
                    size=(next_h, next_w),
                    mode="bilinear",
                    align_corners=False,
                )
            x = stage(x)

        if x.shape[-2:] != output_size:
            x = F.interpolate(
                x,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return self.pred(x)


class DynamicKernelMaskDecoder(nn.Module):
    def __init__(self, in_channels, text_dim=512, hidden_channels=(128, 96, 64), kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        channels = [in_channels] + list(hidden_channels)
        self.stages = nn.ModuleList(
            ConvBNReLU(channels[i], channels[i + 1], kernel_size=3)
            for i in range(len(channels) - 1)
        )
        final_channels = channels[-1]
        self.kernel_mlp = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(inplace=True),
            nn.Linear(text_dim, final_channels * kernel_size * kernel_size + 1),
        )

    def forward(self, visual_feat, text_vec, output_size):
        x = visual_feat
        for stage in self.stages:
            if x.shape[-2:] != output_size:
                next_h = min(x.shape[-2] * 2, output_size[0])
                next_w = min(x.shape[-1] * 2, output_size[1])
                x = F.interpolate(x, size=(next_h, next_w), mode="bilinear", align_corners=False)
            x = stage(x)
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        b, c, h, w = x.shape
        params = self.kernel_mlp(text_vec.float())
        weight = params[:, :-1].view(b, c, self.kernel_size, self.kernel_size)
        bias = params[:, -1]
        x = x.reshape(1, b * c, h, w)
        logits = F.conv2d(x, weight, bias=bias, padding=self.kernel_size // 2, groups=b)
        return logits.view(b, 1, h, w)


class GatedResidualAnswerCalibrationHead(nn.Module):
    def __init__(self, in_dim, num_answers, hidden_dim=512, bottleneck_dim=256, alpha_init=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, num_answers),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_answers),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, fused):
        fused = self.norm(fused)
        return self.alpha * self.gate_head(fused) * self.delta_head(fused)


class MaskGuidedResidualAnswerHead(nn.Module):
    def __init__(self, visual_dim=192, text_dim=512, num_answers=23,
                 hidden_dim=512, dropout=0.1, alpha_init=0.1):
        super().__init__()
        evidence_dim = visual_dim * 3 + text_dim + 4
        self.norm = nn.LayerNorm(evidence_dim)
        self.fuse = nn.Sequential(
            nn.Linear(evidence_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(hidden_dim // 2, num_answers)
        self.gate_head = nn.Sequential(nn.Linear(hidden_dim // 2, num_answers), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, visual_feat, text_vec, mask_logits):
        mask_prob = torch.sigmoid(mask_logits.float())
        mask_prob = F.interpolate(
            mask_prob, size=visual_feat.shape[-2:], mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        visual_feat = visual_feat.float()
        inside_denom = mask_prob.sum(dim=(2, 3)).clamp_min(1e-4)
        outside_prob = 1.0 - mask_prob
        outside_denom = outside_prob.sum(dim=(2, 3)).clamp_min(1e-4)
        inside_vec = (visual_feat * mask_prob).sum(dim=(2, 3)) / inside_denom
        outside_vec = (visual_feat * outside_prob).sum(dim=(2, 3)) / outside_denom
        global_vec = F.adaptive_avg_pool2d(visual_feat, 1).flatten(1)
        area_mean = mask_prob.mean(dim=(2, 3))
        area_std = mask_prob.flatten(2).std(dim=2, unbiased=False)
        area_max = mask_prob.amax(dim=(2, 3))
        area_entropy = -(
            mask_prob * torch.log(mask_prob.clamp_min(1e-6))
            + (1.0 - mask_prob) * torch.log((1.0 - mask_prob).clamp_min(1e-6))
        ).mean(dim=(2, 3))
        area_stats = torch.cat([area_mean, area_std, area_max, area_entropy], dim=1)
        evidence = torch.cat(
            [global_vec, inside_vec, outside_vec, text_vec.float(), area_stats], dim=1
        )
        hidden = self.fuse(self.norm(evidence))
        return self.alpha * self.gate_head(hidden) * self.delta_head(hidden)


class _QRAStream2D(nn.Module):
    """One task branch of a question-conditioned residual adapter.

    The output projection is zero-initialized, so the branch is an identity
    residual at initialization regardless of the text gates. This keeps a loaded
    Change3D-style checkpoint numerically aligned before finetuning.
    """

    def __init__(
        self,
        in_channels=96,
        text_dim=512,
        token_dim=512,
        adapter_dim=192,
        attn_dim=96,
        alpha_init=0.05,
        alpha_max=0.2,
        use_token_attention=True,
        num_groups=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.adapter_dim = adapter_dim
        self.attn_dim = attn_dim
        self.alpha_max = float(alpha_max)
        self.use_token_attention = use_token_attention

        def group_count(channels):
            for groups in (num_groups, 16, 8, 4, 2, 1):
                if channels % groups == 0:
                    return groups
            return 1

        self.input_norm = nn.GroupNorm(group_count(in_channels), in_channels)
        self.visual_proj = nn.Sequential(
            nn.Conv2d(in_channels, adapter_dim, kernel_size=1, bias=False),
            nn.GroupNorm(group_count(adapter_dim), adapter_dim),
            nn.GELU(),
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim),
        )

        if use_token_attention:
            self.q_proj = nn.Conv2d(adapter_dim, attn_dim, kernel_size=1, bias=False)
            self.k_proj = nn.Linear(token_dim, attn_dim, bias=False)
            self.v_proj = nn.Linear(token_dim, attn_dim, bias=False)
            self.ctx_proj = nn.Conv2d(attn_dim, adapter_dim, kernel_size=1, bias=False)
        else:
            self.ctx_from_text = nn.Sequential(
                nn.LayerNorm(text_dim),
                nn.Linear(text_dim, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, adapter_dim),
            )

        self.spatial_gate = nn.Conv2d(adapter_dim * 2, 1, kernel_size=1)
        self.dwconv = nn.Conv2d(
            adapter_dim,
            adapter_dim,
            kernel_size=3,
            padding=1,
            groups=adapter_dim,
            bias=False,
        )
        self.out_act = nn.GELU()
        self.out_proj = nn.Conv2d(adapter_dim, in_channels, kernel_size=1)

        ratio = float(alpha_init) / max(float(alpha_max), 1e-6)
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
        self.alpha_raw = nn.Parameter(torch.logit(torch.tensor(ratio)))

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    def _token_context(self, visual_feat, text_tokens, text_mask=None):
        if text_tokens is None:
            raise ValueError("Token-attention QRA branch requires token-level text features.")
        b, _, h, w = visual_feat.shape
        q = self.q_proj(visual_feat).flatten(2).transpose(1, 2)
        k = self.k_proj(text_tokens.float())
        v = self.v_proj(text_tokens.float())
        attn = torch.matmul(q, k.transpose(1, 2)) * (self.attn_dim ** -0.5)
        if text_mask is not None:
            attn = attn.masked_fill(~text_mask.unsqueeze(1), torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, self.attn_dim, h, w)
        return self.ctx_proj(ctx)

    def forward(self, h_mid, text_vec, text_tokens=None, text_mask=None):
        dtype = h_mid.dtype
        text_vec = text_vec.float()
        visual = self.visual_proj(self.input_norm(h_mid.float())).to(dtype=dtype)
        channel_gate = 2.0 * torch.sigmoid(self.channel_mlp(text_vec))
        channel_gate = channel_gate.view(text_vec.shape[0], self.adapter_dim, 1, 1).to(dtype=dtype)

        if self.use_token_attention:
            ctx = self._token_context(visual.float(), text_tokens, text_mask).to(dtype=dtype)
        else:
            ctx = self.ctx_from_text(text_vec).view(text_vec.shape[0], self.adapter_dim, 1, 1)
            ctx = ctx.expand(-1, -1, visual.shape[-2], visual.shape[-1]).to(dtype=dtype)

        spatial_gate = 2.0 * torch.sigmoid(
            self.spatial_gate(torch.cat([visual.float(), ctx.float()], dim=1))
        ).to(dtype=dtype)
        residual_input = visual * channel_gate * spatial_gate
        delta = self.out_proj(self.out_act(self.dwconv(residual_input.float()))).to(dtype=dtype)
        alpha = self.alpha().to(dtype=dtype)
        out = h_mid + alpha * delta

        with torch.no_grad():
            stats = {
                "alpha": alpha.detach().float().mean(),
                "channel_gate_mean": channel_gate.detach().float().mean(),
                "spatial_gate_mean": spatial_gate.detach().float().mean(),
                "delta_abs_mean": delta.detach().float().abs().mean(),
                "delta_to_mid": delta.detach().float().abs().mean()
                / h_mid.detach().float().abs().mean().clamp_min(1e-6),
            }
        return out, stats


class SharedQRAAdapter3D(nn.Module):
    """One question-conditioned residual stream shared by answer and mask."""

    def __init__(
        self,
        in_channels=96,
        text_dim=512,
        token_dim=512,
        adapter_dim=192,
        attn_dim=96,
        alpha_init=0.05,
        alpha_max=0.2,
    ):
        super().__init__()
        self.shared_adapter = _QRAStream2D(
            in_channels=in_channels,
            text_dim=text_dim,
            token_dim=token_dim,
            adapter_dim=adapter_dim,
            attn_dim=attn_dim,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            use_token_attention=True,
        )
        self.reset_stats()

    def reset_stats(self):
        self._stats_sum = {}
        self._stats_count = 0

    def pop_stats(self):
        if self._stats_count == 0:
            return {}
        stats = {
            f"shared_{key}": value / self._stats_count
            for key, value in self._stats_sum.items()
        }
        self.reset_stats()
        return stats

    def forward(self, x, text_vec, text_tokens=None, text_mask=None):
        if x.shape[2] < 3:
            raise ValueError("SharedQRAAdapter3D expects a 3-frame temporal tensor.")
        middle_idx = x.shape[2] // 2
        h_shared, stats = self.shared_adapter(
            x[:, :, middle_idx], text_vec, text_tokens, text_mask
        )
        x_shared = x.clone()
        x_shared[:, :, middle_idx] = h_shared
        with torch.no_grad():
            for key, value in stats.items():
                value = value.detach().float().mean().item()
                self._stats_sum[key] = self._stats_sum.get(key, 0.0) + value
            self._stats_count += 1
        return x_shared


class QChange3DSharedQRAEncoderForQAG(nn.Module):
    BLOCK_OUT_CHANNELS = {
        1: 24,
        2: 48,
        3: 96,
    }

    def __init__(
        self,
        args,
        text_dim=512,
        token_dim=512,
        adapter_dim=192,
        attn_dim=96,
        alpha_init=0.05,
        alpha_max=0.2,
        insert_after_block=3,
    ):
        super().__init__()
        if insert_after_block not in self.BLOCK_OUT_CHANNELS:
            supported = ", ".join(str(idx) for idx in self.BLOCK_OUT_CHANNELS)
            raise ValueError(
                "Shared-QRA supports insertion after X3D residual blocks "
                f"{supported}; got {insert_after_block}."
            )
        self.args = args
        self.insert_after_block = insert_after_block
        self.x3d = create_qag_x3d(args)
        load_qag_x3d_pretrained(self.x3d, args)
        self.perception_frames = nn.Parameter(
            torch.randn(
                1, 3, args.num_perception_frame,
                args.in_height, args.in_width,
            ),
            requires_grad=True,
        )
        self.shared_qra = SharedQRAAdapter3D(
            in_channels=self.BLOCK_OUT_CHANNELS[insert_after_block],
            text_dim=text_dim,
            token_dim=token_dim,
            adapter_dim=adapter_dim,
            attn_dim=attn_dim,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
        )

    def forward_static(self, img1, img2):
        b = img1.shape[0]
        p0 = self.perception_frames.expand(b, -1, -1, -1, -1)
        x = torch.cat([img1.unsqueeze(2), p0, img2.unsqueeze(2)], dim=2)
        for i in range(5):
            x = self.x3d.blocks[i](x)
        return x[:, :, self.args.num_perception_frame]

    def forward(self, img1, img2, text_vec, text_tokens=None, text_mask=None):
        b = img1.shape[0]
        p0 = self.perception_frames.expand(b, -1, -1, -1, -1)
        x = torch.cat([img1.unsqueeze(2), p0, img2.unsqueeze(2)], dim=2)
        for i in range(self.insert_after_block + 1):
            x = self.x3d.blocks[i](x)
        x = self.shared_qra(x, text_vec, text_tokens, text_mask)
        for i in range(self.insert_after_block + 1, 5):
            x = self.x3d.blocks[i](x)
        return x[:, :, self.args.num_perception_frame]

    def reset_split_qra_stats(self):
        self.shared_qra.reset_stats()

    def pop_split_qra_stats(self):
        return self.shared_qra.pop_stats()


class QChange3D_QAG_Simple(nn.Module):
    def __init__(
        self,
        change3d_pretrained=None,
        clip_pretrained="pretrain/RN101.pt",
        num_answers=23,
        image_size=512,
        encoder_type="shared_qra",
        visual_dim=192,
        text_dim=512,
        x3d_depth_factor=5.0,
        split_qra_adapter_dim=192,
        split_qra_attn_dim=96,
        split_qra_ans_alpha_init=0.05,
        split_qra_ans_alpha_max=0.2,
        split_qra_insert_after_block=3,
    ):
        super().__init__()

        args = SimpleArgs(
            pretrained=change3d_pretrained,
            num_perception_frame=1,
            in_height=image_size,
            in_width=image_size,
        )
        args.x3d_depth_factor = x3d_depth_factor
        if encoder_type != "shared_qra":
            raise ValueError("This release supports the shared QTRA encoder only.")

        self.change_encoder = QChange3DSharedQRAEncoderForQAG(
            args,
            text_dim=text_dim,
            token_dim=text_dim,
            adapter_dim=split_qra_adapter_dim,
            attn_dim=split_qra_attn_dim,
            alpha_init=split_qra_ans_alpha_init,
            alpha_max=split_qra_ans_alpha_max,
            insert_after_block=split_qra_insert_after_block,
        )

        clip_model = torch.jit.load(clip_pretrained, map_location="cpu").eval()
        self.text_encoder = build_model(clip_model.state_dict(), 40).float()

        self.answer_head = nn.Sequential(
            nn.Linear(visual_dim + text_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_answers),
        )

        self.channel_gate = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.ReLU(inplace=True),
            nn.Linear(visual_dim, visual_dim),
            nn.Sigmoid(),
        )

        self.mask_decoder = ProgressiveMaskDecoder(visual_dim)
        self.skip_mask_forward = False

    def reset_qtem_stats(self):
        if hasattr(self.change_encoder, "reset_qtem_stats"):
            self.change_encoder.reset_qtem_stats()

    def pop_qtem_stats(self):
        if hasattr(self.change_encoder, "pop_qtem_stats"):
            return self.change_encoder.pop_qtem_stats()
        return {}

    def reset_split_qra_stats(self):
        if hasattr(self.change_encoder, "reset_split_qra_stats"):
            self.change_encoder.reset_split_qra_stats()

    def pop_split_qra_stats(self):
        if hasattr(self.change_encoder, "pop_split_qra_stats"):
            return self.change_encoder.pop_split_qra_stats()
        return {}


    def encode_question(self, question):
        text_frozen = not any(param.requires_grad for param in self.text_encoder.parameters())
        if text_frozen:
            with torch.no_grad():
                return self.text_encoder.encode_text(question)
        return self.text_encoder.encode_text(question)

    def decode_default_mask(self, visual_feat, text_vec, output_size):
        if self.skip_mask_forward:
            return None
        gate = self.channel_gate(text_vec.float()).unsqueeze(-1).unsqueeze(-1)
        mask_feat = visual_feat * gate
        return self.mask_decoder(mask_feat, output_size=output_size)

    def forward(self, img1, img2, question):
        """
        img1/img2: [B, 3, H, W]
        question: [B, 40]
        returns:
            {
                "answer_logits": [B, num_answers],
                "mask_logits": [B, 1, H, W],
            }
        """
        text_tokens, text_vec = self.encode_question(question)
        text_mask = question != 0

        visual_feat = self.change_encoder(img1, img2, text_vec, text_tokens, text_mask)
        visual_vec = F.adaptive_avg_pool2d(visual_feat, 1).flatten(1)

        fused = torch.cat([visual_vec, text_vec.float()], dim=1)
        answer_logits = self.answer_head(fused)

        mask_logits = self.decode_default_mask(visual_feat, text_vec, img1.shape[-2:])

        return {
            "answer_logits": answer_logits,
            "mask_logits": mask_logits,
        }


class QChange3D_QAG_SharedQRA(QChange3D_QAG_Simple):
    def __init__(self, *args, **kwargs):
        kwargs["encoder_type"] = "shared_qra"
        super().__init__(*args, **kwargs)


def _zero_init_answer_calibration(calibration):
    """Make a newly attached calibration branch an exact identity residual."""
    final_layer = calibration.delta_head[-1]
    nn.init.zeros_(final_layer.weight)
    nn.init.zeros_(final_layer.bias)


class QChange3D_QAG_SharedQRAAnswerCalibration(QChange3D_QAG_SharedQRA):
    """Frozen Shared-QRA encoder plus a trainable residual answer calibrator."""

    def __init__(self, *args, **kwargs):
        visual_dim = kwargs.get("visual_dim", 192)
        text_dim = kwargs.get("text_dim", 512)
        num_answers = kwargs.get("num_answers", 23)
        super().__init__(*args, **kwargs)
        self.answer_calibration = GatedResidualAnswerCalibrationHead(
            visual_dim + text_dim, num_answers, alpha_init=0.2
        )
        _zero_init_answer_calibration(self.answer_calibration)
        self.answer_only_trainable_modules = ("answer_calibration",)

    def forward(self, img1, img2, question):
        text_tokens, text_vec = self.encode_question(question)
        text_mask = question != 0
        visual_feat = self.change_encoder(
            img1, img2, text_vec, text_tokens, text_mask
        )
        visual_vec = F.adaptive_avg_pool2d(visual_feat, 1).flatten(1)
        fused = torch.cat([visual_vec, text_vec.float()], dim=1)
        answer_logits = self.answer_head(fused) + self.answer_calibration(fused)
        mask_logits = self.decode_default_mask(
            visual_feat, text_vec, img1.shape[-2:]
        )
        return {"answer_logits": answer_logits, "mask_logits": mask_logits}


class QChange3D_QAG_SharedQRAAnswerCalibrationDualGate(
    QChange3D_QAG_SharedQRAAnswerCalibration
):
    """Frozen Shared-QRA answer path plus a trainable DualGate mask decoder."""

    def __init__(self, *args, **kwargs):
        visual_dim = kwargs.get("visual_dim", 192)
        text_dim = kwargs.get("text_dim", 512)
        super().__init__(*args, **kwargs)
        self.dynamic_mask_decoder = DynamicKernelMaskDecoder(
            visual_dim, text_dim=text_dim
        )
        self.dual_gate = nn.Sequential(
            nn.Linear(text_dim + visual_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        nn.init.normal_(self.dual_gate[-1].weight, std=1e-3)
        with torch.no_grad():
            self.dual_gate[-1].bias[0].fill_(4.5951199)
            self.dual_gate[-1].bias[1].fill_(-2.1972246)
        self.mask_only_trainable_modules = (
            "mask_decoder",
            "dynamic_mask_decoder",
            "dual_gate",
        )

    def forward(self, img1, img2, question):
        text_tokens, text_vec = self.encode_question(question)
        text_mask = question != 0
        visual_feat = self.change_encoder(
            img1, img2, text_vec, text_tokens, text_mask
        )

        visual_vec = F.adaptive_avg_pool2d(visual_feat, 1).flatten(1)
        fused = torch.cat([visual_vec, text_vec.float()], dim=1)
        answer_logits = self.answer_head(fused) + self.answer_calibration(fused)

        output_size = img1.shape[-2:]
        dynamic_logits = self.dynamic_mask_decoder(
            visual_feat, text_vec, output_size=output_size
        )
        static_logits = self.mask_decoder(visual_feat, output_size=output_size)
        gates = torch.sigmoid(
            self.dual_gate(
                torch.cat([text_vec.float(), visual_vec.float()], dim=1)
            )
        )
        dynamic_gate = gates[:, 0].view(-1, 1, 1, 1)
        static_gate = gates[:, 1].view(-1, 1, 1, 1)
        mask_logits = dynamic_gate * dynamic_logits + static_gate * static_logits
        return {"answer_logits": answer_logits, "mask_logits": mask_logits}


class QChange3D_QAG_SharedQRAAnswerCalibrationDualGateMaskGuided(
    QChange3D_QAG_SharedQRAAnswerCalibrationDualGate
):
    """Shared-QRA DualGate model plus a mask-guided answer residual."""

    def __init__(self, *args, **kwargs):
        visual_dim = kwargs.get("visual_dim", 192)
        text_dim = kwargs.get("text_dim", 512)
        num_answers = kwargs.get("num_answers", 23)
        super().__init__(*args, **kwargs)
        self.mask_guided_answer = MaskGuidedResidualAnswerHead(
            visual_dim=visual_dim,
            text_dim=text_dim,
            num_answers=num_answers,
        )
        nn.init.zeros_(self.mask_guided_answer.delta_head.weight)
        nn.init.zeros_(self.mask_guided_answer.delta_head.bias)
        self.answer_only_trainable_modules = ("mask_guided_answer",)
        self.answer_requires_mask = True

    def forward(self, img1, img2, question):
        text_tokens, text_vec = self.encode_question(question)
        text_mask = question != 0
        visual_feat = self.change_encoder(
            img1, img2, text_vec, text_tokens, text_mask
        )
        visual_vec = F.adaptive_avg_pool2d(visual_feat, 1).flatten(1)
        fused = torch.cat([visual_vec, text_vec.float()], dim=1)

        output_size = img1.shape[-2:]
        dynamic_logits = self.dynamic_mask_decoder(
            visual_feat, text_vec, output_size=output_size
        )
        static_logits = self.mask_decoder(visual_feat, output_size=output_size)
        gates = torch.sigmoid(
            self.dual_gate(
                torch.cat([text_vec.float(), visual_vec.float()], dim=1)
            )
        )
        dynamic_gate = gates[:, 0].view(-1, 1, 1, 1)
        static_gate = gates[:, 1].view(-1, 1, 1, 1)
        mask_logits = dynamic_gate * dynamic_logits + static_gate * static_logits

        answer_logits = self.answer_head(fused) + self.answer_calibration(fused)
        answer_logits = answer_logits + self.mask_guided_answer(
            visual_feat.detach(), text_vec.detach(), mask_logits.detach()
        )
        return {"answer_logits": answer_logits, "mask_logits": mask_logits}
