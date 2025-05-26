import subprocess
from pathlib import Path


def test_cpplint():
    repo_root = Path(__file__).resolve().parents[2]
    cpp_dir = repo_root / "mettagrid"
    patterns = ["*.cpp", "*.h", "*.hpp", "*.cxx"]
    files = [str(p) for pattern in patterns for p in cpp_dir.rglob(pattern) if "third_party" not in str(p)]
    assert files, "no C++ files found"
    result = subprocess.run(
        ["cpplint", "--quiet", *files],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
