import enum
from collections.abc import Callable

import torch
from torch.nn import functional as F


class Task(object):
    def __init__(self,
                 batch_size: int,
                 dataset_size: int,
                 dim_data: int,
                 device: torch.device | str,
                 *,
                 data_source: Callable = None):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.dim_data = dim_data
        self.device = device
        self.data_source = data_source or torch.randn

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def loss_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearRegression(Task):
    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data_source(self.batch_size, self.dataset_size, self.dim_data, device=self.device)
        w = F.normalize(x.new_empty(self.batch_size, self.dim_data).normal_())
        return x, torch.einsum("bsd,bd->bs", x, w)

    def loss_f(self, x: torch.Tensor, y: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
        return F.mse_loss(x, y, reduction=reduction)


class Dataset(enum.StrEnum):
    linear_regression = enum.auto()

    def build(self,
              batch_size: int,
              dataset_size: int,
              dim_data: int,
              device: torch.device | str
              ) -> Task:
        match self:
            case Dataset.linear_regression:
                task = LinearRegression(batch_size, dataset_size, dim_data, device=device)
            case _:
                raise ValueError()

        return task
