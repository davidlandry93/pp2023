from typing import Any

import logging
import os
import math
import numpy as np
import pathlib
import torch
import torch.utils.data

_logger = logging.getLogger(__name__)


class TorchRecordDataset:
    """Dataset where the examples are already transformed and saved as torch
    dictionaries."""

    def __init__(
        self,
        input_dir,
        limit_features=None,
        to_32bits=False,
        shuffle=False,
        limit=None,
        n_members=None,
    ):
        self.input_dir = pathlib.Path(input_dir)
        self.files = list(self.input_dir.glob("*.pt"))

        if shuffle:
            perm = torch.randperm(len(self.files))
            self.files = [self.files[x] for x in perm]
        else:
            self.files = sorted(self.files)

        self.limit_features = limit_features
        self.to_32bits = to_32bits
        self.limit = limit
        self.n_members = n_members

    def __len__(self):
        if self.limit is not None:
            return min(len(self.files), self.limit)
        else:
            return len(self.files)

    def __getitem__(self, idx):
        example = torch.load(self.files[idx])

        features = example["features"]

        if self.limit_features is not None:
            features = features[..., : self.limit_features]

        example["features"] = features

        if self.to_32bits:
            new_example = {}
            for k in example:
                if example[k].dtype == torch.float64:
                    new_example[k] = example[k].to(dtype=torch.float32)
                else:
                    new_example[k] = example[k]

            example = new_example

        if self.n_members is not None:
            new_example = {}
            for k in example:
                if k in (
                    "forecast",
                    "features",
                    "metadata_features",
                ):
                    new_example[k] = torch.clone(example[k][:, 0 : self.n_members])
                else:
                    new_example[k] = example[k]

            new_example["forecast_sort_idx"] = torch.argsort(
                new_example["forecast"], dim=1
            )

            example = new_example

        return example


class AbstractIterStepDataset(torch.utils.data.IterableDataset):
    """An iterator dataset that returns the examples step by step, instead of the whole
    forecast at once."""

    def __init__(
        self,
        input_dataset: Any,
        transform=None,
        min_rows: int = None,
        shuffle_inner: bool = False,
        shuffle_steps=False,
    ):
        """Args:
        smc_dataset: The source SMCParquetDataset. It will be returned step by step
            by the iterator.
        n_steps: Number of steps for the model. Defaults to 81 which is the number
            of steps in the GDPS model.
        transform: The transform to apply to the example.
        min_rows: If specified, filter out all the examples which have less that the
            minimum amount of rows in them.
        """
        self.input_dataset = input_dataset
        self.transform = transform
        self.shuffle_inner = shuffle_inner
        self.shuffle_steps = shuffle_steps
        self.min_rows = min_rows

    def get_worker_bounds(self):
        local_rank = os.getenv("LOCAL_RANK", None)
        if local_rank is not None:
            return self.get_worker_bounds_ddp()
        else:
            return self.get_worker_bounds_single()

    def get_worker_bounds_single(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            iter_start, iter_end = 0, len(self.input_dataset)
        else:
            # Taken from:
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
            per_worker = int(
                math.ceil(len(self.input_dataset) / float(worker_info.num_workers))
            )
            worker_id = int(worker_info.id)
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.input_dataset))

        return iter_start, iter_end

    def get_worker_bounds_ddp(self):
        local_rank = int(os.getenv("LOCAL_RANK"))
        world_size = int(os.getenv("WORLD_SIZE"))

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            n_workers_per_node = 1
            worker_id = 0
        else:
            n_workers_per_node = int(worker_info.num_workers)
            worker_id = int(worker_info.id)

        n_workers = n_workers_per_node * world_size
        per_worker = int(math.ceil(len(self.input_dataset) / float(n_workers)))

        worker_id = local_rank * n_workers_per_node + worker_id

        iter_start = worker_id * per_worker
        iter_end = min(iter_start + per_worker, len(self.input_dataset))

        return iter_start, iter_end

    def __iter__(self):
        iter_start, iter_end = self.get_worker_bounds()

        inner_idx = torch.tensor(list(range(iter_start, iter_end)))
        if self.shuffle_inner:
            inner_idx = inner_idx[torch.randperm(len(inner_idx))]

        input_dataset_it = (self.input_dataset[i] for i in inner_idx)

        for example in input_dataset_it:
            unique_steps = torch.tensor(self.list_steps(example))

            if self.shuffle_steps:
                unique_steps = unique_steps[torch.randperm(len(unique_steps))]
            else:
                unique_steps = sorted(unique_steps)

            for s in sorted(unique_steps):
                filtered = self.filter_example(example, s)

                # Because there is some missing data, sometimes a timestep is
                # completely missing for a given forecast. In that case, we want
                # to drop the example.
                if self.min_rows is not None and len(filtered.index) < self.min_rows:
                    _logger.warning(
                        f"Dropping time step {s} because not enough rows are present in the example."
                    )
                    continue

                if self.transform:
                    filtered = self.transform(filtered)

                yield filtered

    def list_steps(self, example):
        raise NotImplementedError("IterStepDataset must implement list_steps")

    def filter_example(self, example, step):
        raise NotImplementedError("IterStepDataset must implement filter_example")


class TorchIterStepDataset(AbstractIterStepDataset):
    def __init__(self, inner, shuffle_inner=False, shuffle_steps=False):
        super().__init__(
            inner, shuffle_inner=shuffle_inner, shuffle_steps=shuffle_steps
        )

    def list_steps(self, example):
        unique_steps = list(range(example["target"].shape[0]))
        return unique_steps

    def filter_example(
        self, example: dict[str, torch.Tensor], step: int
    ) -> dict[str, torch.Tensor]:
        to_return = {}
        for k in example:
            if k in [
                "deterministic_forecast",
                "features",
                "forecast_parameters",
                "forecast_sort_idx",
                "forecast",
                "metadata_features",
                "step_idx",
                "step_ns",
                "target",
            ]:
                to_return[k] = example[k][step, ...]
            else:
                to_return[k] = example[k]

        return to_return


class TorchOnlyOneStepDataset(TorchIterStepDataset):
    def __init__(self, inner, step_idx):
        super().__init__(inner)
        self.step_idx = step_idx

    def list_steps(self, example):
        if self.step_idx in example["step_idx"]:
            return [self.step_idx]
        else:
            return []


class CacheDataset:
    def __init__(self, inner_dataset):
        self.inner = inner_dataset
        self.cache = {}

    def __getitem__(self, key):
        if key not in self.cache:
            self.cache[key] = self.inner[key]

        return self.cache[key]

    def __len__(self):
        return len(self.inner)


def make_torch_record_datasets(input_dir, cache=False, **kwargs):
    input_path = pathlib.Path(input_dir)

    datasets = [
        TorchRecordDataset(input_path / "train", shuffle=True, **kwargs),
        TorchRecordDataset(input_path / "val", **kwargs),
        TorchRecordDataset(input_path / "test", **kwargs),
    ]

    if cache:
        datasets = [CacheDataset(d) for d in datasets]

    return datasets


def make_torch_record_step_datasets(input_dir, cache=False, **kwargs):
    train, val, test = make_torch_record_datasets(input_dir, cache=cache, **kwargs)

    return [
        TorchIterStepDataset(train, shuffle_inner=True, shuffle_steps=True),
        TorchIterStepDataset(val),
        TorchIterStepDataset(test),
    ]


def make_one_step_datasets(input_dir, step_idx, **kwargs):
    train, val, test = make_torch_record_datasets(input_dir, **kwargs)

    return [
        TorchOnlyOneStepDataset(train, step_idx),
        TorchOnlyOneStepDataset(val, step_idx),
        TorchOnlyOneStepDataset(test, step_idx),
    ]
