# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import logging
import yaml
from pathlib import Path

import torch, platform
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  # WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config

logging.getLogger("numba").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.INFO)
torch.set_float32_matmul_precision("high")
from AR.utils import get_newest_ckpt
from collections import OrderedDict


class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        # if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if (
                self._every_n_epochs >= 1
                and (trainer.current_epoch + 1) % self._every_n_epochs == 0
            ):
                if (
                    self.if_save_latest == True
                ):  ####如果设置只保存最后一个ckpt，在保存下一个ckpt后要清理掉之前的所有ckpt
                    to_clean = list([i for i in os.listdir(self.dirpath) if i.endswith('.ckpt')])
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest == True:
                    for name in to_clean:
                        try:
                            os.remove("%s/%s" % (self.dirpath, name))
                        except:
                            pass
                if self.if_save_every_weights == True:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt:
                        to_save_od["weight"][key] = dictt[key].half()
                    to_save_od["config"] = self.config
                    to_save_od["info"] = "GPT-e%s" % (trainer.current_epoch + 1)
                    torch.save(
                        to_save_od,
                        "%s/%s-e%s.ckpt"
                        % (
                            self.half_weights_save_dir,
                            self.exp_name,
                            trainer.current_epoch + 1,
                        ),
                    )
            self._save_last_checkpoint(trainer, monitor_candidates)


def modify_config(config, args):
    config["train"]["epochs"] = args.epochs
    config["train"]["batch_size"] = args.batch_size
    config["train"]["precision"] = '16-mixed' if args.is_half else '32'
    config["train"]["save_every_n_epoch"] = args.save_every_epoch
    config["train"]["if_dpo"] = args.if_dpo
    config["data"]["num_workers"] = args.num_workers
    config["pretrained_gpt"] = args.pretrained_gpt


def main(args):
    config = load_yaml_config(args.config)
    seed_everything(config["train"]["seed"], workers=True)
    output_dir = os.path.join(args.log_dir , args.name, 'gpt')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modify_config(config, args)
    
    half_weights_save_dir=os.path.join(args.model_dir, 'gpt_weights', args.name)
    os.makedirs(half_weights_save_dir, exist_ok=True)
    
    with open(f"{output_dir}/config.yaml", "w") as f:
        f.write(yaml.dump(config, default_flow_style=False))
    print("Finish writing config!")

    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=args.if_save_latest,
        if_save_every_weights=args.if_save_every_weights,
        half_weights_save_dir=half_weights_save_dir,
        exp_name=args.name,
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=args.save_every_epoch,
        dirpath=output_dir,
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir.parent)

    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="gpu",
        # val_check_interval=9999999999999999999999,###不要验证
        # check_val_every_n_epoch=None,
        limit_val_batches=0,
        devices=-1,
        benchmark=False,
        fast_dev_run=False,
        strategy = "auto" if torch.backends.mps.is_available() else DDPStrategy(
            process_group_backend="nccl" if platform.system() != "Windows" else "gloo"
        ),  # mps 不支持多节点训练
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(
        config, output_dir
    )

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=os.path.join(args.log_dir, args.name, args.train_semantic_path),
        train_phoneme_path=os.path.join(args.log_dir, args.name, args.train_phoneme_path),
        # dev_semantic_path=args.dev_semantic_path,
        # dev_phoneme_path=args.dev_phoneme_path
    )

    try:
        # 使用正则表达式匹配文件名中的数字部分，并按数字大小进行排序
        newest_ckpt_name = get_newest_ckpt(os.listdir(output_dir))
        ckpt_path = output_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example with modularized command line arguments.')

    # General arguments
    general = parser.add_argument_group('General')
    general.add_argument('-c', '--config', type=str, default='src/configs/s1longer.yaml', help='YAML file for configuration')
    general.add_argument('-n', '--name', type=str, required=True, help='speaker name')
    general.add_argument('--train_semantic_path', type=str, default="name2semantic.tsv")
    general.add_argument('--train_phoneme_path', type=str, default="text2phonemes.txt")

    # Training arguments
    training_args = parser.add_argument_group('Training')
    training_args.add_argument('-e', '--epochs', type=int, default=20, help='finetune epochs')
    training_args.add_argument('-bs', '--batch_size', type=int, default=16, help='batch size')
    training_args.add_argument('--pretrained_gpt', type=str, default='pretrained_models/gpt_weights/pretrained/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt', help='pretrained model')
    training_args.add_argument('--is_half', action='store_true', help='use half precision')
    training_args.add_argument('--if_dpo', action='store_false', help='use dpo')
    training_args.add_argument('-nw', '--num_workers', type=int, default=0, help='num workers for dataloader')
    
    # Saving/Logging arguments
    save_log_args = parser.add_argument_group('Saving/Logging')
    save_log_args.add_argument('-d', '--model_dir', type=str, default="pretrained_models", help='Model directory')
    save_log_args.add_argument('-l', '--log_dir', type=str, default="logs/", help='Log directory')
    save_log_args.add_argument('--if_save_latest', action='store_false', help='Save the latest weights')
    save_log_args.add_argument('--if_save_every_weights', action='store_false', help='Save every weights')
    save_log_args.add_argument('--save_every_epoch', type=int, default=5, help='Save every n epoch')
    
    args = parser.parse_args()
    logging.info(str(args))
    
    main(args)
