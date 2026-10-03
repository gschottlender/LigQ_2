import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from publish_resources import remote_matches


class PublicationTests(unittest.TestCase):
    def test_small_remote_git_blob_matches_exact_local_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            payload = b"historical metadata"
            path.write_bytes(payload)
            record = {"size_bytes":len(payload), "sha256":hashlib.sha256(payload).hexdigest()}
            sibling = SimpleNamespace(size=len(payload), lfs=None,
                blob_id=hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest())
            self.assertTrue(remote_matches(record, sibling, path))
            sibling.blob_id = "0" * 40
            self.assertFalse(remote_matches(record, sibling, path))

    def test_lfs_uses_digest_and_size_without_reading_large_payload(self):
        record = {"size_bytes":1000, "sha256":"a" * 64}
        sibling = SimpleNamespace(size=1000, lfs=SimpleNamespace(sha256="a" * 64))
        self.assertTrue(remote_matches(record, sibling, Path("not_read")))
        sibling.size = 999
        self.assertFalse(remote_matches(record, sibling, Path("not_read")))
        sibling.size, sibling.lfs.sha256 = 1000, "b" * 64
        self.assertFalse(remote_matches(record, sibling, Path("not_read")))


if __name__ == "__main__":
    unittest.main()
