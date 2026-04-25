import os

import sys

import torch.multiprocessing as mp
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from src.EditSR.architectures.model import Model
from src.EditSR.architectures.data import DataModule
import wandb
import gc, torch
from sympy.core.cache import clear_cache
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
from pathlib import Path
import torch
from src.EditSR.project_paths import resolve_path, scripts_path
torch.set_float32_matmul_precision('medium')
class MemoryClearCallback(pl.Callback):
    """
    定期手动释放 Python / CUDA / SymPy 的缓存。
    clear_every_n_batches: 多少个 batch 清一次；为 0 则只在 epoch 结束时清。
    """
    def __init__(self, clear_every_n_batches: int = 0):
        super().__init__()
        self.clear_every_n_batches = clear_every_n_batches

    # 每 N 个 batch 结束触发
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.clear_every_n_batches and (batch_idx + 1) % self.clear_every_n_batches == 0:
            self._clear()

    # 每个 epoch 结束触发
    def on_train_epoch_end(self, trainer, pl_module):
        if not self.clear_every_n_batches:
            self._clear()

    # 同理可加到 validation/test 的回调里
    def _clear(self):
        # 1) Python 垃圾回收
        gc.collect()

        # 2) SymPy 缓存
        clear_cache()

        # 3) CUDA 显存（若使用 GPU）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@hydra.main(config_name="config", version_base = '1.1')
def main(cfg):
    seed_everything(0)
    train_path = resolve_path(cfg.train_path, base="scripts")
    val_path = resolve_path(cfg.val_path, base="scripts")
    test_path = resolve_path(cfg.test_path, base="scripts")
    data = DataModule(
        train_path,
        val_path,
        test_path,
        cfg
    )

    model = Model(cfg=cfg.architecture)

    model_path = resolve_path(cfg.model_path, base="scripts")
    # model = Model.load_from_checkpoint(model_path, cfg=cfg.architecture)
    model = Model(cfg=cfg.architecture)
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    # # 当前模型的参数字典
    model_dict = model.state_dict()
    # # 定义只加载的子模块
    # # submodules_to_freeze = ["visual_encoder", "visual_decoder", "vq"]
    # # 只保留名称、shape匹配且属于指定子模块的参数项
    filtered_dict = {
         k: v for k, v in state_dict.items()
         if (k in model_dict and model_dict[k].shape == v.shape )
     }
    # # 显示加载了哪些参数（可选）
    # print(f"加载了 {len(filtered_dict)} 个匹配的参数项，共有 {len(model_dict)} 项")
    skipped = [k for k in state_dict.keys() if k not in filtered_dict]
    # print(f"跳过了 {len(skipped)} 个 shape 不匹配、参数名不匹配或不存在的参数项：")
    for k in skipped:
         print(f"  {k} -> shape: {state_dict[k].shape}")
    #
    # # 更新模型参数
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)


    if cfg.wandb:
        wandb.login(key='cbd7304a9dcb55202c48245bfb925a6e9e96b7d7')
        wandb.finish()
        wandb.init(project="NIPS")
        config = wandb.config
        wandb_logger = WandbLogger()
    else:
        wandb_logger = None

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss", #/dataloader_idx_0",
        dirpath=str(scripts_path("Exp_weights")),
        filename=train_path.stem+"train_loss_"+"-{epoch:02d}-{train_loss:.2f}",
        mode="min",
    )
    trainer = pl.Trainer(
        accelerator=cfg.gpu,
        max_epochs=cfg.architecture.epochs,
        val_check_interval=cfg.val_check_interval,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback,MemoryClearCallback(clear_every_n_batches=200)],
    )
    trainer.fit(model, data)
    # trainer.test(model, data)

if __name__ == "__main__":
    mp.freeze_support()
    start_method = 'spawn' if os.name == 'nt' else 'forkserver'
    mp.set_start_method(start_method, force=True)
    main()
