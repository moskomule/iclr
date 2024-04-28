import dataclasses
import enum

import torch
from torch.nn import functional as F


@dataclasses.dataclass
class Task:
    batch_size: int
    dataset_size: int
    dim_data: int
    device: str | torch.device
    seed: int = 0

    def __post_init__(self):
        self.generator = torch.Generator(self.device).manual_seed(self.seed)

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def loss_f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearRegression(Task):
    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.batch_size, self.dataset_size, self.dim_data, device=self.device,
                        generator=self.generator)
        w = F.normalize(x.new_empty(self.batch_size, self.dim_data).normal_())
        return x, torch.einsum("bsd,bd->bs", x, w)

    def loss_f(self, x: torch.Tensor, y: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
        return F.mse_loss(x, y, reduction=reduction)


class LinearClassification(Task):
    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.batch_size, self.dataset_size, self.dim_data, device=self.device,
                        generator=self.generator)

        w = F.normalize(x.new_empty(self.batch_size, self.dim_data).normal_())
        y = torch.einsum("bsd,bd->bs", x, w).sign().clamp_(0, 1)  # 0 or 1
        return x, y

    def loss_f(self, x: torch.Tensor, y: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x, y, reduction=reduction)


class Dataset(enum.StrEnum):
    linear_regression = enum.auto()
    linear_classification = enum.auto()

    def build(self,
              batch_size: int,
              dataset_size: int,
              dim_data: int,
              device: torch.device | str
              ) -> Task:
        match self:
            case Dataset.linear_regression:
                task = LinearRegression(batch_size, dataset_size, dim_data, device=device)
            case Dataset.linear_classification:
                task = LinearClassification(batch_size, dataset_size, dim_data, device=device)
            case _:
                raise ValueError()

        return task
