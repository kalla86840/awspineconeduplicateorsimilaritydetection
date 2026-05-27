import tempfile
import unittest
from pathlib import Path

from scripts.detect_similarity import (
    FileInfo,
    duplicate_groups,
    normalize_tokens,
    shingles,
    similar_pairs,
    text_shingles,
)


class DetectSimilarityTests(unittest.TestCase):
    def test_duplicate_groups_require_same_hash_and_size(self):
        files = [
            FileInfo("a.txt", 4, "abc"),
            FileInfo("b.txt", 4, "abc"),
            FileInfo("c.txt", 5, "abc"),
            FileInfo("d.txt", 4, "def"),
        ]

        groups = duplicate_groups(files)

        self.assertEqual(len(groups), 1)
        self.assertEqual({item.path for item in groups[0]}, {"a.txt", "b.txt"})

    def test_similar_pairs_scores_jaccard_similarity(self):
        left = shingles(normalize_tokens("alpha beta gamma delta epsilon zeta eta theta"), 3)
        right = shingles(normalize_tokens("alpha beta gamma delta epsilon zeta eta changed"), 3)

        pairs = similar_pairs({"left.py": left, "right.py": right}, threshold=0.5, min_shared_shingles=1)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].left, "left.py")
        self.assertEqual(pairs[0].right, "right.py")
        self.assertGreaterEqual(pairs[0].similarity, 0.5)

    def test_text_shingles_skips_non_text_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_file = root / "a.py"
            binary_file = root / "b.bin"
            text_file.write_text("def hello(): return 'world'\n", encoding="utf-8")
            binary_file.write_bytes(b"def hello(): return 'world'\n")
            files = [
                FileInfo("a.py", text_file.stat().st_size, "a"),
                FileInfo("b.bin", binary_file.stat().st_size, "b"),
            ]

            result = text_shingles(root, files, shingle_width=2, max_bytes=1000)

        self.assertIn("a.py", result)
        self.assertNotIn("b.bin", result)


if __name__ == "__main__":
    unittest.main()
