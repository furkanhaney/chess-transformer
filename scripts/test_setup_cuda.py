#!/usr/bin/env python3
"""Focused tests for the local CUDA alias materialization contract."""

import contextlib
import importlib.util
import io
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = pathlib.Path(__file__).with_name("setup_cuda.py")
SPEC = importlib.util.spec_from_file_location("setup_cuda", SCRIPT)
assert SPEC and SPEC.loader
SETUP_CUDA = importlib.util.module_from_spec(SPEC)
sys.dont_write_bytecode = True
SPEC.loader.exec_module(SETUP_CUDA)


class MaterializeCudaAliasesTests(unittest.TestCase):
    def make_toolkit(self, root: pathlib.Path) -> None:
        for index, (alias_name, source_name) in enumerate(SETUP_CUDA.CUDA_ALIASES):
            source = root / source_name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"versioned-{index // 2}\n".encode())
            alias = root / alias_name
            alias.symlink_to(pathlib.Path(source_name).name)

    def test_replaces_all_links_with_independent_regular_copies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_toolkit(root)
            SETUP_CUDA.materialize_cuda_aliases(root)

            for alias_name, source_name in SETUP_CUDA.CUDA_ALIASES:
                alias, source = root / alias_name, root / source_name
                self.assertTrue(alias.is_file())
                self.assertFalse(alias.is_symlink())
                self.assertEqual(alias.read_bytes(), source.read_bytes())
                self.assertNotEqual(alias.stat().st_ino, source.stat().st_ino)

    def test_is_idempotent_and_repairs_corrupt_regular_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_toolkit(root)
            SETUP_CUDA.materialize_cuda_aliases(root)
            first = root / SETUP_CUDA.CUDA_ALIASES[0][0]
            unchanged = root / SETUP_CUDA.CUDA_ALIASES[1][0]
            unchanged_inode = unchanged.stat().st_ino
            first.write_bytes(b"corrupt")

            SETUP_CUDA.materialize_cuda_aliases(root)

            self.assertEqual(first.read_bytes(), (root / SETUP_CUDA.CUDA_ALIASES[0][1]).read_bytes())
            self.assertEqual(unchanged.stat().st_ino, unchanged_inode)

    def test_rejects_unexpected_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_toolkit(root)
            (root / "unexpected").symlink_to("nowhere")
            with self.assertRaisesRegex(RuntimeError, "unexpected CUDA symbolic link: unexpected"):
                SETUP_CUDA.materialize_cuda_aliases(root)

    def test_rejects_linked_versioned_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_toolkit(root)
            source = root / SETUP_CUDA.CUDA_ALIASES[0][1]
            source.unlink()
            source.symlink_to("elsewhere")
            with self.assertRaisesRegex(RuntimeError, "unexpected CUDA symbolic link"):
                SETUP_CUDA.materialize_cuda_aliases(root)

    def test_main_repairs_aliases_after_all_package_markers_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.make_toolkit(root)
            manifest = {}
            for name in SETUP_CUDA.PACKAGES:
                digest = f"digest-{name}"
                manifest[name] = {
                    "linux-x86_64": {
                        "relative_path": f"unused/{name}.tar.xz",
                        "sha256": digest,
                        "size": 0,
                    }
                }
                (root / f".{name}.sha256").write_text(digest + "\n")

            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = __import__("json").dumps(manifest).encode()
            with (
                mock.patch.object(SETUP_CUDA, "DEST", root),
                mock.patch.object(SETUP_CUDA.platform, "system", return_value="Linux"),
                mock.patch.object(SETUP_CUDA.platform, "machine", return_value="x86_64"),
                mock.patch.object(SETUP_CUDA.urllib.request, "urlopen", return_value=response) as urlopen,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                SETUP_CUDA.main()

            self.assertEqual(urlopen.call_count, 1)
            self.assertFalse(any(path.is_symlink() for path in root.rglob("*")))


if __name__ == "__main__":
    unittest.main()
