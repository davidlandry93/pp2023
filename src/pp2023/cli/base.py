import omegaconf as oc
import hydra
import logging
import torch
import os
import torch.utils.data

logger = logging.getLogger(__name__)


def build_datasets_from_config(cfg: oc.DictConfig):
    train_dataset, val_dataset, test_dataset = hydra.utils.instantiate(
        cfg.ex.dataset.maker,
    )

    return train_dataset, val_dataset, test_dataset


def build_dataloader_from_dataset(dataset, cfg, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=cfg.ex.get("batch_size", None),
        num_workers=cfg.num_workers,
        # persistent_workers=True,
    )


def build_dataloaders_from_config(
    cfg: oc.DictConfig,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    logger.info(f"Using dataset: {cfg.ex.dataset}")
    train_dataset, val_dataset, test_dataset = build_datasets_from_config(cfg)

    local_rank = os.getenv("LOCAL_RANK", None)

    if local_rank is not None:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset,
            drop_last=False,
        )
        test_sampler = torch.utils.data.DistributedSampler(
            test_dataset,
            drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.ex.get("batch_size", None),
        num_workers=cfg.num_workers,
        sampler=train_sampler,
        persistent_workers=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.ex.get("batch_size", None),
        num_workers=cfg.num_workers,
        sampler=val_sampler,
        persistent_workers=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.ex.get("batch_size", None),
        num_workers=cfg.num_workers,
        sampler=test_sampler,
        persistent_workers=True,
    )

    return train_dataloader, val_dataloader, test_dataloader


def build_model_from_config(cfg: oc.DictConfig) -> torch.nn.Module:
    dataset_cfg = cfg.ex.dataset
    model = hydra.utils.instantiate(
        cfg.ex.model,
        in_features=dataset_cfg.n_features,
        n_parameters=cfg.ex.distribution.n_parameters,
        n_forecasts=dataset_cfg.n_forecasts,
        n_steps=dataset_cfg.n_steps,
        n_stations=dataset_cfg.n_stations,
        n_members=dataset_cfg.n_members,
    )

    return model
