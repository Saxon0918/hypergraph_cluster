from datetime import date

from hce.config import config_hash, load_config, project_root, resolve_project_path


def test_config_hash_normalizes_yaml_dates_deterministically():
    as_date = {"source_dates": [date(2009, 10, 1), date(2009, 10, 2)]}
    as_text = {"source_dates": ["2009-10-01", "2009-10-02"]}
    assert config_hash(as_date) == config_hash(as_text)


def test_relative_paths_resolve_from_project_root_not_current_directory(tmp_path):
    root = tmp_path / "project"
    configs = root / "configs"
    configs.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='test'\nversion='0'\n")
    config_path = configs / "portable.yaml"
    config_path.write_text("sources: {}\n")
    config = load_config(config_path)
    assert project_root(config) == root.resolve()
    assert resolve_project_path(config, "data/raw/x.txt") == (
        root / "data/raw/x.txt"
    ).resolve()
