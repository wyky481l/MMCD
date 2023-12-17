#!/usr/bin/env python
# _*_coding:utf-8_*_
import argparse
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from condfig import setting_init
from models.mmcd import MMCD
from modules.data_module.dataModule import MMCDDataModule, set_random_seed

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    set_random_seed(2023)
    args = setting_init()

    tb_logger = pl.loggers.TensorBoardLogger(
        args.out_log_dir,
        name="mmcd_train",
        version=None,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_checkpoint_dir,
        monitor='total_loss',
        filename='diffusion-{epoch:02d}-{structDiff_loss:.4f}-{seq_score:.4f}-{total_loss:.4f}',
        save_top_k=args.save_top_k,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )

    model = MMCD(
        n_timestep=args.n_timestep,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        temperature=args.temperature,
        learning_rate_struct=args.learning_rate_struct,
        learning_rate_seq=args.learning_rate_seq,
        learning_rate_cont=args.learning_rate_cont
    )
    dataModule = MMCDDataModule(batch_size=args.batch_size)
    trainer.fit(model, dataModule)
