from lXtractor.core.config import Config, DefaultConfig


def test_default_config():
    d = DefaultConfig
    print(d.__dict__)
    assert isinstance(d, Config)
    assert all(isinstance(v, dict) for v in d.values())
    init_value = d["bonds"]["non_covalent_upper"]
    d["bonds"]["non_covalent_upper"] = 20
    d.save()
    d.reload()
    assert d["bonds"]["non_covalent_upper"] == 20
    d.reset_to_defaults()
    d.clear_user_config()
    d.reload()
    assert d["bonds"]["non_covalent_upper"] == init_value
