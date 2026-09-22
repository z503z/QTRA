"""Exercise the real training loop with a tiny synthetic model (not experiment results)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

import train


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.answer_calibration = nn.Linear(3, 2)

    def forward(self, image1, image2, question):
        logits = self.answer_calibration(image1.mean(dim=(2, 3)))
        return {'answer_logits': logits, 'mask_logits': image1[:, :1]}


class EntrypointTest(unittest.TestCase):
    def test_training_resume_and_log_schema(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='qtra-unit-') as directory:
            root = Path(directory)
            config = {
                'task': 'qag', 'model': {'variant': 'full'},
                'training': dict(stage='global', epochs=1, batch_size=2, grad_accum_steps=1,
                                 optimizer='adamw', lr=1e-4, weight_decay=1e-4, precision='fp32',
                                 freeze_text=True, lambda_answer=1, lambda_mask=0),
            }
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            samples = [(torch.ones(3, 2, 2), torch.zeros(3, 2, 2), torch.ones(40).long(),
                        'largest_change', torch.tensor([1.0, 0.0]), torch.ones(1, 2, 2), '0')] * 2
            output = root / 'run'
            command = ['train.py', '--config', str(config_path), '--clip-pretrained', 'unused',
                       '--train-root', 'train-only', '--val-root', 'val-only', '--output', str(output),
                       '--workers', '0', '--device', 'cpu']
            with patch('train.build_model', side_effect=lambda *a: TinyModel()), \
                 patch('train.qag_dataset', return_value=samples) as dataset, \
                 patch('train.load_weights', return_value={}):
                with patch('sys.argv', command + ['--init-checkpoint', 'unused']):
                    train.main()
                self.assertEqual([call.args[1] for call in dataset.call_args_list], ['Train', 'Val'])
            checkpoint = torch.load(output / 'last.pth', weights_only=True)
            self.assertEqual(set(checkpoint['metrics']), {'train', 'val'})
            self.assertTrue(all(key.startswith('val_') for key in checkpoint['best']))
            self.assertTrue((output / 'best_val_answer.pth').exists())
            with patch('train.build_model', side_effect=lambda *a: TinyModel()), \
                 patch('train.qag_dataset', return_value=samples), \
                 patch('sys.argv', command + ['--resume', str(output / 'last.pth'), '--epochs', '2']):
                train.main()
            records = [json.loads(line) for line in (output / 'train_log.jsonl').read_text().splitlines()]
            self.assertEqual([record['epoch'] for record in records], [0, 1])
            for record in records:
                self.assertFalse(any('test' in key for key in record))
            for path in output.glob('*'):
                self.assertNotIn('test', path.name)


if __name__ == '__main__':
    unittest.main()
