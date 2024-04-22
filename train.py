import pathlib

import chika
import torch
import torch.distributed
from rich import print
from tqdm.rich import trange

from iclr.data import Dataset
from iclr.transformers import STTransformer


@chika.config
class Config:
    dataset: Dataset = Dataset.linear_regression
    batch_size: int = 64
    dataset_size: int = 100
    dim_data: int = 20
    num_iters: int = 500_000

    num_layers: int = 12
    dim_embed: int = 21
    num_heads: int = 8

    lr: float = 1e-4
    weight_decay: float = None
    grad_clip: float = 0

    seed: int = 0
    gpu: int = 0
    compile: bool = False

    log_freq: int = 1_000
    save_freq: int = 10_000
    save_dir: str = "outputs"
    data_cache: str = "~/.torch/data"


@chika.main(Config, strict=True)
def main(cfg: Config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu)

    print(cfg)
    path = pathlib.Path(cfg.save_dir).resolve()

    task = cfg.dataset.build(cfg.batch_size, cfg.dataset_size, cfg.dim_data, device)

    model = STTransformer(cfg.num_layers, cfg.dim_embed, cfg.num_heads, enable_flash_attention=True)
    model.to(device)
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr)

    if cfg.compile:
        model = torch.compile(model)

    loss_history = []
    for i in trange(cfg.num_iters):
        xs, ys = task()
        output = model(xs, ys)
        loss = task.loss_f(output[:, -1, 1], ys[:, -1])
        loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(True)

        if i % cfg.log_freq == 0:
            print(f"loss={loss.item():.4f}")
            loss_history.append((i, loss.item()))

        if i % cfg.save_freq == 0:
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter": i,
                        "cfg": cfg.to_dict(),
                        "loss_history": loss_history},
                       path / f"checkpoint{i:08}.pt")


if __name__ == '__main__':
    main()
