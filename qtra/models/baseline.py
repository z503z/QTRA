import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from .common import SimpleArgs
from .qag import ProgressiveMaskDecoder
from .x3d_change3d import create_x3d
from .clip import build_model


class Change3DEncoderForQAG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.x3d = create_x3d(input_clip_length=3, depth_factor=5.0)

        if args.pretrained is not None:
            try:
                state = torch.load(args.pretrained, map_location="cpu")
                if "model_state" in state:
                    state_dict = state["model_state"]
                else:
                    state_dict = state
                msg = self.x3d.load_state_dict(state_dict, strict=True)
                print(f"[Change3D] loaded pretrained: {args.pretrained}, {msg}")
            except Exception as e:
                raise RuntimeError(f"Failed to load X3D initialization: {args.pretrained}") from e

        self.perception_frames = nn.Parameter(
            torch.randn(
                1,
                3,
                args.num_perception_frame,
                args.in_height,
                args.in_width,
            ),
            requires_grad=True,
        )

    def forward(self, img1, img2):
        b = img1.shape[0]

        p0 = repeat(
            self.perception_frames,
            "1 c t h w -> b c t h w",
            b=b,
        )

        frames = torch.cat(
            [
                img1.unsqueeze(2),
                p0,
                img2.unsqueeze(2),
            ],
            dim=2,
        )

        x = frames
        for i in range(5):
            x = self.x3d.blocks[i](x)

        return x[:, :, self.args.num_perception_frame]


class Change3D_QAG_Simple(nn.Module):
    def __init__(
        self,
        change3d_pretrained=None,
        clip_pretrained="pretrain/RN101.pt",
        num_answers=23,
        image_size=512,
        visual_dim=192,
        text_dim=512,
    ):
        super().__init__()

        args = SimpleArgs(
            pretrained=change3d_pretrained,
            num_perception_frame=1,
            in_height=image_size,
            in_width=image_size,
        )

        self.change_encoder = Change3DEncoderForQAG(args)

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
        _, text_vec = self.text_encoder.encode_text(question)

        visual_feat = self.change_encoder(img1, img2)
        visual_vec = F.adaptive_avg_pool2d(visual_feat, 1).flatten(1)

        fused = torch.cat([visual_vec, text_vec.float()], dim=1)
        answer_logits = self.answer_head(fused)

        gate = self.channel_gate(text_vec.float()).unsqueeze(-1).unsqueeze(-1)
        mask_feat = visual_feat * gate
        mask_logits = self.mask_decoder(mask_feat, output_size=img1.shape[-2:])

        return {
            "answer_logits": answer_logits,
            "mask_logits": mask_logits,
        }
