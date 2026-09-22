import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SimpleArgs
from .qag import (
    GatedResidualAnswerCalibrationHead,
    QChange3DSharedQRAEncoderForQAG,
)
from .clip import build_model


class QChange3D_SharedQRA_CDVQA(nn.Module):
    """Answer-only CDVQA adaptation of the final Shared-QRA architecture.

    CDVQA has no grounding supervision. The model therefore retains the
    Shared-QRA visual-language encoder and the ordinary answer head, while
    omitting the DualGate mask decoder and mask-guided answer residual.
    Answer calibration can be enabled for the full answer-path variant or
    disabled for a matched Shared-QRA ablation.
    """

    def __init__(
        self,
        change3d_pretrained=None,
        clip_pretrained="pretrain/RN101.pt",
        num_answers=23,
        image_size=256,
        visual_dim=192,
        text_dim=512,
        x3d_depth_factor=5.0,
        qra_adapter_dim=192,
        qra_attn_dim=96,
        qra_alpha_init=0.05,
        qra_alpha_max=0.2,
        use_answer_calibration=False,
    ):
        super().__init__()

        args = SimpleArgs(
            pretrained=change3d_pretrained,
            num_perception_frame=1,
            in_height=image_size,
            in_width=image_size,
        )
        args.x3d_depth_factor = x3d_depth_factor

        self.change_encoder = QChange3DSharedQRAEncoderForQAG(
            args,
            text_dim=text_dim,
            token_dim=text_dim,
            adapter_dim=qra_adapter_dim,
            attn_dim=qra_attn_dim,
            alpha_init=qra_alpha_init,
            alpha_max=qra_alpha_max,
            insert_after_block=3,
        )

        clip_model = torch.jit.load(clip_pretrained, map_location="cpu").eval()
        self.text_encoder = build_model(clip_model.state_dict(), 40).float()

        fused_dim = visual_dim + text_dim
        self.answer_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_answers),
        )

        self.use_answer_calibration = bool(use_answer_calibration)
        if self.use_answer_calibration:
            self.answer_calibration = GatedResidualAnswerCalibrationHead(
                fused_dim,
                num_answers,
                alpha_init=0.2,
            )
            # Match the final QAG calibration design: the new residual starts
            # as an exact identity and learns only when it is useful.
            nn.init.zeros_(self.answer_calibration.delta_head[-1].weight)
            nn.init.zeros_(self.answer_calibration.delta_head[-1].bias)
        else:
            self.answer_calibration = None

    def encode_question(self, question):
        text_frozen = not any(
            param.requires_grad for param in self.text_encoder.parameters()
        )
        if text_frozen:
            with torch.no_grad():
                return self.text_encoder.encode_text(question)
        return self.text_encoder.encode_text(question)

    def reset_qra_stats(self):
        self.change_encoder.reset_split_qra_stats()

    def pop_qra_stats(self):
        return self.change_encoder.pop_split_qra_stats()

    def forward(self, img1, img2, question):
        text_tokens, text_vec = self.encode_question(question)
        text_mask = question != 0

        visual_feat = self.change_encoder(
            img1,
            img2,
            text_vec,
            text_tokens,
            text_mask,
        )
        visual_vec = F.adaptive_avg_pool2d(visual_feat, 1).flatten(1)
        fused = torch.cat([visual_vec, text_vec.float()], dim=1)

        logits = self.answer_head(fused)
        if self.answer_calibration is not None:
            logits = logits + self.answer_calibration(fused)
        return logits


class QChange3D_SharedQRACalibration_CDVQA(QChange3D_SharedQRA_CDVQA):
    """Shared-QRA CDVQA model with gated residual answer calibration."""

    def __init__(self, *args, **kwargs):
        kwargs["use_answer_calibration"] = True
        super().__init__(*args, **kwargs)
