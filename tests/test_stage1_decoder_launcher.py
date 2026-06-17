import unittest
from pathlib import Path


class Stage1DecoderLauncherTest(unittest.TestCase):
    def test_debug_and_output_dir_can_be_overridden_for_temporary_runs(self):
        script = Path("script/run_distill_stage1_decoder_gsm8k.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('debug="${DEBUG:-False}"', script)
        self.assertIn('output_dir="${OUTPUT_DIR:-${save_root}/${output_name}}"', script)


if __name__ == "__main__":
    unittest.main()
