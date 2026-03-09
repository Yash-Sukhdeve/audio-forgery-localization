import tempfile
import yaml
import pytest
from core.utils.config import load_config


class TestLoadConfig:
    def test_load_yaml(self):
        cfg = {"batch_size": 8, "lr": 1e-5, "model": {"name": "fara"}}
        path = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
        yaml.dump(cfg, path)
        path.close()
        loaded = load_config(path.name)
        assert loaded.batch_size == 8
        assert loaded.lr == 1e-5
        assert loaded.model.name == "fara"

    def test_nested_access(self):
        cfg = {"a": {"b": {"c": 42}}}
        path = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
        yaml.dump(cfg, path)
        path.close()
        loaded = load_config(path.name)
        assert loaded.a.b.c == 42

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent.yaml")
