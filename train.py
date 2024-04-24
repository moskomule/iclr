import pathlib
import statistics

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
    dim_embed: int = 128
    num_heads: int = 8

    lr: float = 1e-4
    weight_decay: float = 1e-2
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
    torch.manual_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu)

    print(cfg)
    path = pathlib.Path(cfg.save_dir).resolve()
    path.mkdir(exist_ok=True, parents=True)

    task = cfg.dataset.build(cfg.batch_size, cfg.dataset_size, cfg.dim_data, device)

    model = STTransformer(cfg.num_layers, cfg.dim_embed, cfg.num_heads, cfg.dim_data + 1, enable_flash_attention=True)
    model.to(device)
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr)

    if cfg.compile:
        model = torch.compile(model)

    loss_history = []
    for i in trange(cfg.num_iters):
        xs, ys = task()
        output = model(xs, ys)
        loss = task.loss_f(output[:, -1, 0], ys[:, -1])
        loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(True)

        loss_history.append(loss.item())
        if i > 0 and i % cfg.log_freq == 0:
            print(f"[{i:>8}] loss={statistics.mean(loss_history[-cfg.log_freq:]):.4f}")

        if i % cfg.save_freq == 0:
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter": i,
                        "cfg": cfg.to_dict(),
                        "loss_history": loss_history},
                       path / f"checkpoint{i:08}.pt")


if __name__ == '__main__':
    main()
