"""Fast checks for metric definitions, stage freezing and training separation."""
import ast
import unittest
from pathlib import Path

import torch
from torch import nn

from qtra.metrics import calc_mask_confusion, DiceLoss
from qtra.training import configure_stage, epoch_record, validation_scores


class MockEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_qra = nn.Linear(2, 2)
        self.x3d = nn.Module()
        self.x3d.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(5)])
        self.perception_frames = nn.Parameter(torch.ones(1))


def mock_model():
    model = nn.Module()
    model.change_encoder = MockEncoder()
    for name in ('text_encoder', 'answer_head', 'channel_gate', 'mask_decoder',
                 'answer_calibration', 'dynamic_mask_decoder', 'dual_gate', 'mask_guided_answer'):
        setattr(model, name, nn.Linear(2, 2))
    return model


class TrainingContract(unittest.TestCase):
    def test_training_source_does_not_reference_held_out_split(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'train.py', root / 'qtra/training.py'):
            source = path.read_text()
            self.assertNotIn('test', source.lower())
            ast.parse(source)

    def test_qtra_freeze_then_unfreeze(self):
        model = mock_model()
        cfg = dict(stage='qtra', lr=1e-4, warmup_epochs=3, adapter_warmup_lr=2e-4,
                   adapter_lr=1e-4, head_lr=1e-5, backbone_lr=1e-5)
        configure_stage(model, cfg, 0)
        self.assertTrue(model.change_encoder.shared_qra.weight.requires_grad)
        self.assertFalse(model.change_encoder.x3d.blocks[4].weight.requires_grad)
        configure_stage(model, cfg, 3)
        self.assertTrue(model.change_encoder.x3d.blocks[4].weight.requires_grad)
        self.assertFalse(model.change_encoder.x3d.blocks[3].weight.requires_grad)
        self.assertFalse(model.change_encoder.perception_frames.requires_grad)
        self.assertFalse(model.text_encoder.weight.requires_grad)

    def test_refinement_trains_only_selected_modules(self):
        for stage, expected in [('global', {'answer_calibration'}),
                                ('grounding', {'mask_decoder', 'dynamic_mask_decoder', 'dual_gate'}),
                                ('evidence', {'mask_guided_answer'})]:
            model = mock_model()
            _, groups, _ = configure_stage(model, {'stage': stage, 'lr': 1e-4}, 0)
            self.assertEqual({g['name'] for g in groups}, expected)
            actual = {name.split('.')[0] for name, p in model.named_parameters() if p.requires_grad}
            self.assertEqual(actual, expected)

    def test_record_and_selection_use_validation(self):
        optimizer = torch.optim.Adam([nn.Parameter(torch.zeros(1))])
        optimizer.param_groups[0]['name'] = 'head'
        metrics = {'oa': 0.7, 'aa': 0.6, 'miou': 0.4, 'oiou': 0.5}
        scores = validation_scores(metrics)
        self.assertTrue(all(key.startswith('val_') for key in scores))
        self.assertAlmostEqual(scores['val_grounding'], 0.45)
        record = epoch_record(0, 'global', optimizer, {'loss': 1.2}, metrics)
        self.assertFalse(any('test' in key for key in record))

    def test_mask_metrics(self):
        logits = torch.tensor([[[[20.0, -20.0], [20.0, -20.0]]]])
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        tp, fp, fn = calc_mask_confusion(logits, target, 0.35)
        self.assertEqual((tp.item(), fp.item(), fn.item()), (1, 1, 1))
        self.assertAlmostEqual(DiceLoss()(logits, target).item(), 0.5, places=5)


if __name__ == '__main__':
    unittest.main()
