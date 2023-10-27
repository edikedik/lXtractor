from lXtractor.core.config import Config, DefaultConfig


def test_default_config():
    d = DefaultConfig
    group, field, val = "unknowns", "chain_id", "Y"
    assert isinstance(d, Config)
    assert all(isinstance(v, dict) for v in d.values())
    init_value = d[group][field]
    d[group][field] = val
    d.save()
    d.reload()
    assert d[group][field] == val
    d.reset_to_defaults()
    d.clear_user_config()
    d.reload()
    assert d[group][field] == init_value
