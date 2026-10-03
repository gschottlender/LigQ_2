import signal
import unittest
from pathlib import Path
from unittest.mock import patch

from package.common import ROOT
from stop_publication_test import stop


class StopPublicationTests(unittest.TestCase):
    def test_only_verified_isolated_worker_group_is_stopped(self):
        record = {"status":"running", "pid":1234}
        arguments = str(ROOT / "test_publication.py").encode() + b"\0--worker\0"
        with patch("stop_publication_test.read_json", return_value=record), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_bytes", return_value=arguments), \
             patch("stop_publication_test.os.getpgid", return_value=1234), \
             patch("stop_publication_test.os.killpg") as terminate, \
             patch("stop_publication_test.write_json") as save:
            stop(Path("example_job"))
        terminate.assert_called_once_with(1234, signal.SIGTERM)
        self.assertEqual(save.call_args.args[1]["status"], "stopped_by_user")
        self.assertTrue(save.call_args.args[1]["downloaded_files_preserved"])

    def test_reused_pid_or_nonisolated_group_is_never_stopped(self):
        for arguments, group in ((b"unrelated.py\0", 1234),
                                 (str(ROOT / "test_publication.py").encode()+b"\0--worker\0", 99)):
            with patch("stop_publication_test.read_json", return_value={"status":"running", "pid":1234}), \
                 patch.object(Path, "exists", return_value=True), \
                 patch.object(Path, "read_bytes", return_value=arguments), \
                 patch("stop_publication_test.os.getpgid", return_value=group), \
                 patch("stop_publication_test.os.killpg") as terminate:
                with self.assertRaises(ValueError):
                    stop(Path("example_job"))
                terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
