import argparse
import pickle
from pathlib import Path

import lightning as L
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from txtox.data.datamodules import PyGAnnDataGraphDataModule
from txtox.models.gnn_tx_msl import LitGNNHetReg2dMSL
from txtox.utils import get_datetime, get_paths

parser = argparse.ArgumentParser(description="Training config")
parser.add_argument("--expname", type=str, default="debug_msl")
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--skew", type=int, default=0)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--load_ckpt_path", type=str, default=None)
args = parser.parse_args()


def main(expname: str, max_epochs: int, skew: int, load_ckpt_path: str, k: int):
    # data parameters, we'll eventually obtain this from the data.

    skew = skew > 0  # turn into boolean
    n_genes = 500
    n_labels = 188

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
        file_names=[f"3section_k{k}_{i}.h5ad" for i in range(3)],
        cell_type="subclass",
        spatial_coords=["x_section", "y_section", "z_section"],
        batch_size=50,
        val_batch_size=100,
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
            weight_ce=0.1,
            skew=skew,
        )
        print(f"Loaded checkpoint: {load_ckpt_path}")
    else:
        model = LitGNNHetReg2dMSL(
            input_dim=n_genes,
            n_labels=n_labels,
            weight_msl_nll=1.0,
            weight_ce=0.1,
            skew=skew,
        )

    # fit wrapper
    trainer = L.Trainer(
        limit_train_batches=1000,
        max_epochs=max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        accelerator="cpu",
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.save_checkpoint(checkpoint_path + f"/end-epoch-{max_epochs}.ckpt")

    result = {}
    when = {}

    # get predictions from last saved model
    result["last"] = trainer.predict(model, datamodule=datamodule)
    when["last"] = max_epochs

    # get results from best checkpoint
    ckpt = next(iter(checkpoint_callback.best_k_models.keys()))
    model = LitGNNHetReg2dMSL.load_from_checkpoint(ckpt, input_dim=n_genes, n_labels=n_labels)
    result["best"] = trainer.predict(model, datamodule=datamodule)
    when["best"] = int(ckpt.split("epoch=")[1].split("-")[0])

    for key, predictions in result.items():
        xy_mu_pred = np.concatenate([predictions[batch][0] for batch in range(len(predictions))], axis=0)
        xy_L_pred = np.concatenate([predictions[batch][1] for batch in range(len(predictions))], axis=0)
        xy_gamma_pred = np.concatenate([predictions[batch][2] for batch in range(len(predictions))], axis=0)
        celltype_pred = np.concatenate([predictions[batch][3] for batch in range(len(predictions))], axis=0)

        # convert L to covariance matrices using L @ L.T
        xy_cov_pred = xy_L_pred @ xy_L_pred.transpose(0, 2, 1)
        print(xy_cov_pred.shape)

        # calculates the eigenvalues and eigenvectors for all covariance matrices
        eigvals, eigvecs = np.linalg.eig(xy_cov_pred)

        # order them in descending order of eigenvalues
        order = np.argsort(-eigvals, axis=1)
        eigvals_ord = np.take_along_axis(eigvals, order, axis=1)
        eigvecs_ord = np.take_along_axis(eigvecs, order[:, np.newaxis, :], axis=2)

        eigvals = eigvals_ord
        eigvecs = eigvecs_ord
        del eigvals_ord, eigvecs_ord

        xy = datamodule.dataset.adata.obs[["x_section", "y_section"]].values

        # data for clustering
        data = {
            "eigvals": eigvals,
            "eigvecs": eigvecs,
            "xy_cov_pred": xy_cov_pred,
            "xy_mu_pred": xy_mu_pred,
            "xy_gamma_pred": xy_gamma_pred,
            "xy": xy,
            "section_idx": datamodule.dataset.adata.obs["z_section"].values,
            "subclass": datamodule.dataset.adata.obs["subclass"].values,
            "subclass_color": datamodule.dataset.adata.obs["subclass_color"].values,
        }

        pickle.dump(data, open(paths["data_root"] + f"/results/knn-part2/{key}_{expname}.pkl", "wb"))


if __name__ == "__main__":
    main(expname=args.expname, max_epochs=args.max_epochs, load_ckpt_path=args.load_ckpt_path, skew=args.skew, k=args.k)
