import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from evaluation import ensure_disjoint, passes
from features import CausalAuxExtractor
from infer import LUMIInferencer
from train import main as train


class ReleaseTests(unittest.TestCase):
    def test_mixed_fps_matches_individual_clips(self):
        clips = torch.rand(2, 4, 3, 8, 8)
        extractor = CausalAuxExtractor()
        together, _ = extractor(clips, fps=torch.tensor([24., 60.]))
        for i, fps in enumerate([24., 60.]):
            single, _ = extractor(clips[i:i+1], fps=fps)
            torch.testing.assert_close(single, together[i:i+1])

    def test_video_resampling(self):
        engine = LUMIInferencer(weights_path=None, device="cpu", allow_untrained=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.mp4"
            writer = engine._open_mp4_writer(str(src), 30, 32, 32)
            for i in range(6):
                writer.write(np.full((32, 32, 3), 100+i, np.uint8))
            writer.release()
            for fps, expected in [(15, 3), (30, 6), (60, 12)]:
                outputs = [str(root / f"{kind}.mp4") for kind in ("corrected", "mask")]
                predictions = iter([0, 0, 0, 0, 0, 1])
                def predict(rgb, aux, state):
                    return {"corrected": rgb, "state": None,
                            "mask": torch.full_like(rgb[:, :1], next(predictions))}
                with patch.object(engine.model, "step", side_effect=predict), patch.object(
                    cv2, "applyColorMap", wraps=cv2.applyColorMap
                ) as colorize:
                    report = engine.process_video(str(src), *outputs, target_fps=fps, allow_untrained=True)
                self.assertTrue(np.all(colorize.call_args.args[0] == 255))
                self.assertEqual(report["analyzed_frames"], 6)
                for path in outputs:
                    cap = cv2.VideoCapture(path)
                    count = 0
                    while cap.read()[0]:
                        count += 1
                    self.assertEqual(cap.get(cv2.CAP_PROP_FPS), fps)
                    cap.release()
                    self.assertEqual(count, expected)

    def test_training_and_validation_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training, validation = root / "train", root / "val"
            training.mkdir()
            validation.mkdir()
            for i, (folder, fps) in enumerate([(training, 24), (training, 60), (validation, 30)]):
                rgb = np.full((2, 3, 8, 8), 0.25 + i * 0.01, np.float32)
                mask = np.zeros((2, 1, 8, 8), np.float32)
                mask[:, :, :2] = 1
                np.savez(folder / f"{i}.npz", rgb=rgb, clean=rgb, target_mask=mask, fps=fps)
            path = root / "candidate.pth"
            args = ["train.py", "--data-dir", str(training), "--epochs", "1", "--save-path", str(path)]
            with patch.object(sys, "argv", args):
                train()
            with self.assertRaises(RuntimeError):
                LUMIInferencer(str(path))
            # Permissive thresholds exercise the acceptance path in this temporary fixture.
            with patch.object(sys, "argv", args + ["--val-dir", str(validation), "--min-recall", "0",
                                                  "--max-fpr", "1", "--max-hazard-mae", "1"]):
                train()
            self.assertTrue(LUMIInferencer(str(path)).validated)
            ckpt = torch.load(path, weights_only=True)
            self.assertEqual(ckpt["metadata"]["trained_fps"], [24., 60.])
            self.assertFalse(passes(dict(ckpt["evaluation"], recall=0, min_recall=1)))
            with self.assertRaises(ValueError):
                ensure_disjoint(training.glob("*.npz"), training.glob("*.npz"))

            passing = ckpt["evaluation"]
            failing = dict(passing, recall=0, min_recall=1)
            with patch.object(sys, "argv", args + ["--epochs", "2", "--val-dir", str(validation),
                                                   "--min-recall", "0", "--max-fpr", "1",
                                                   "--max-hazard-mae", "1"]), patch(
                "train.evaluate", side_effect=[passing, failing]
            ), patch("train.train_one_video", side_effect=[{"total": 2}, {"total": 1}]):
                train()
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["metadata"]["epochs"], 1)
            self.assertTrue(LUMIInferencer(str(path)).validated)
