import omegaconf as oc
import hydra
import logging
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def build_dataloaders_from_config(
    cfg: oc.DictConfig,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    logger.info(f"Using dataset: {cfg.ex.dataset}")
    train_dataset, val_dataset, test_dataset = hydra.utils.instantiate(
        cfg.ex.dataset.maker,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.ex.batch_size,
        num_workers=cfg.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.ex.batch_size,
        num_workers=cfg.num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.ex.batch_size,
        num_workers=cfg.num_workers,
    )

    return train_dataloader, val_dataloader, test_dataloader


def build_model_from_config(cfg: oc.DictConfig) -> torch.nn.Module:
    dataset_cfg = cfg.ex.dataset
    model = hydra.utils.instantiate(
        cfg.ex.model,
        in_features=dataset_cfg.n_features,
        n_variables=2,
        n_parameters=cfg.ex.distribution.n_parameters,
        n_steps=dataset_cfg.n_steps,
        n_stations=dataset_cfg.n_stations,
    )

    return model
