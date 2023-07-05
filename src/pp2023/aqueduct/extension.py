import importlib.resources
import omegaconf as oc


def task_modules():
    return ["pp2023.aqueduct.tasks"]


def aq_config() -> oc.DictConfig:
    cfg_path = importlib.resources.files(__package__) / "config.yaml"
    return oc.OmegaConf.load(cfg_path)
