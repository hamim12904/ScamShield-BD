import pytest

from src.config import find_nested_data_dir
from tools.build_notebook import validate_modules


def test_validate_modules_rejects_stub(tmp_path):
    module = tmp_path / "src" / "stub.py"
    module.parent.mkdir(parents=True)
    module.write_text("def f():\n    raise NotImplementedError\n", encoding="utf-8")

    with pytest.raises(ValueError, match="NotImplementedError"):
        validate_modules(("src/stub.py",), tmp_path)


def test_validate_modules_requires_imported_module(tmp_path):
    a = tmp_path / "src" / "a.py"
    b = tmp_path / "src" / "b.py"
    a.parent.mkdir(parents=True)
    a.write_text("from src import b\n", encoding="utf-8")
    b.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not in MODULE_PATHS"):
        validate_modules(("src/a.py",), tmp_path)
    validate_modules(("src/a.py", "src/b.py"), tmp_path)


def test_find_nested_data_dir(tmp_path):
    nested = tmp_path / "competition" / "data"
    nested.mkdir(parents=True)
    (nested / "sample.csv").write_text("id,text\n", encoding="utf-8")

    assert find_nested_data_dir(tmp_path) == nested
