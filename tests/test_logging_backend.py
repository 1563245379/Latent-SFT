from pathlib import Path
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "script"
REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


class LoggingBackendTest(unittest.TestCase):
    def test_training_launchers_use_tensorboard_instead_of_wandb(self):
        launchers = sorted(SCRIPT_DIR.glob("run_distill_*.sh"))
        self.assertTrue(launchers, "No training launcher scripts found")

        for launcher in launchers:
            text = launcher.read_text(encoding="utf-8")
            with self.subTest(launcher=launcher.name):
                self.assertNotIn("wandb", text.lower())
                self.assertNotIn("WANDB", text)
                self.assertIn("--report_to tensorboard", text)

    def test_tensorboard_dependency_is_declared(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8")

        self.assertIn("tensorboard", requirements.lower())


if __name__ == "__main__":
    unittest.main()
