from typing import Any

import logging
import math
import numpy as np
import random
import pathlib
import torch

_logger = logging.getLogger(__name__)


class TorchRecordDataset:
    """Dataset where the examples are already transformed and saved as torch
    dictionaries."""

    def __init__(self, input_dir, limit_features=None):
        self.input_dir = pathlib.Path(input_dir)
        self.files = list(self.input_dir.glob("*.pt"))
        self.limit_features = limit_features

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        example = torch.load(self.files[idx])

        if self.limit_features is not None:
            example["features"] = example["features"][..., : self.limit_features]
        return example


class AbstractIterStepDataset(torch.utils.data.IterableDataset):
    """An iterator dataset that returns the examples step by step, instead of the whole
    forecast at once."""

    def __init__(
        self,
        input_dataset: Any,
        transform=None,
        min_rows: int = None,
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
        self.shuffle_steps = shuffle_steps
        self.min_rows = min_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            input_dataset_it = iter(self.input_dataset)
        else:
            # Taken from:
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
            per_worker = int(
                math.ceil(len(self.input_dataset) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.input_dataset))

            _logger.debug(f"Worker bounds: {iter_start} to {iter_end}")

            input_dataset_it = (
                self.input_dataset[i] for i in range(iter_start, iter_end)
            )

        for example in input_dataset_it:
            unique_steps = self.list_steps(example)

            if self.shuffle_steps:
                random.shuffle(unique_steps)
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
    def __init__(self, path, **kwargs):
        super().__init__(TorchRecordDataset(path, **kwargs))

    def list_steps(self, example):
        unique_steps = list(range(example["target"].shape[0]))
        return unique_steps

    def filter_example(
        self, example: dict[str, torch.Tensor], step: int
    ) -> dict[str, torch.Tensor]:
        to_return = {"step_idx": torch.from_numpy(np.array(step))}
        for k in [
            "features",
            "target",
            "forecast_parameters",
            "deterministic_forecast",
        ]:
            to_return[k] = example[k][step, ...]

        for k in ["month", "day_of_year"]:
            to_return[k] = example[k]

        return to_return


def make_torch_record_datasets(input_dir):
    input_path = pathlib.Path(input_dir)
    return [
        TorchRecordDataset(input_path / "train"),
        TorchRecordDataset(input_path / "val"),
        TorchRecordDataset(input_path / "test"),
    ]


def make_torch_record_step_datasets(input_dir, **kwargs):
    input_path = pathlib.Path(input_dir)
    return [
        TorchIterStepDataset(input_path / "train", **kwargs),
        TorchIterStepDataset(input_path / "val", **kwargs),
        TorchIterStepDataset(input_path / "test", **kwargs),
    ]
