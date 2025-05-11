import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from txtox.data.datamodules import PyGAnnDataGraphDataModule
from txtox.models.gnn_het_reg_2d_msl_mnist import LitGNNHetReg2dMSL
from txtox.utils import get_datetime, get_paths

parser = argparse.ArgumentParser(description="Training config")
parser.add_argument("--expname", type=str, default="debug_msl")
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--skew", type=int, default=0)
parser.add_argument("--load_ckpt_path", type=str, default=None)
args = parser.parse_args()


def main(expname: str, max_epochs: int, skew: int, load_ckpt_path: str):
    # data parameters, we'll eventually obtain this from the data.

    skew = skew > 0  # turn into boolean
    n_genes = 784
    n_labels = 10

    # paths
    paths = get_paths()
    expname = get_datetime(expname=expname)
    log_path = paths["data_root"] + f"logs/{expname}"
    checkpoint_path = paths["data_root"] + f"checkpoints/{expname}"

    # helpers
    tb_logger = TensorBoardLogger(save_dir=log_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, monitor="val_rmse_overall", filename="{epoch}-{val_rmse_overall:.2f}"
    )

    # data
    datamodule = PyGAnnDataGraphDataModule(
        data_dir=paths["data_root"],
        file_names=["mnist.h5ad"],
        cell_type="subclass",
        spatial_coords=["x_section", "y_section", "z_section"],
        batch_size=5,
        val_batch_size=50,
        n_hops=2,
        num_workers=8,
    )

    # model
    if load_ckpt_path is not None:
        assert Path(load_ckpt_path).exists(), f"Checkpoint: {load_ckpt_path} does not exist"
        model = LitGNNHetReg2dMSL.load_from_checkpoint(
            load_ckpt_path,
            input_dim=n_genes,
            n_labels=n_labels,
            weight_msl_nll=1.0,
            weight_ce=0.0,
            skew=skew,
        )
        print(f"Loaded checkpoint: {load_ckpt_path}")
    else:
        model = LitGNNHetReg2dMSL(
            input_dim=n_genes,
            n_labels=n_labels,
            weight_msl_nll=1.0,
            weight_ce=0.0,
            skew=skew,
        )

    # fit wrapper
    trainer = L.Trainer(
        limit_train_batches=1000,
        max_epochs=max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )
    trainer.fit(model=model, datamodule=datamodule)

    # save model at the end of training
    trainer.save_checkpoint(checkpoint_path + f"/end-epoch-{max_epochs}.ckpt")


if __name__ == "__main__":
    main(expname=args.expname, max_epochs=args.max_epochs, load_ckpt_path=args.load_ckpt_path, skew=args.skew)
