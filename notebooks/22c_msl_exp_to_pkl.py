import os
import pickle

import anndata as ad
import lightning as L
import numpy as np
import seaborn as sns

from txtox.data.datamodules import PyGAnnDataGraphDataModule
from txtox.models.gnn_het_reg_2d_msl_mnist import LitGNNHetReg2dMSL
from txtox.utils import get_paths

n_genes = 784
n_labels = 10
paths = get_paths()

for term in ["val_rmse", "end-epoch"]:
    # get experiment name and checkpoint file name
    fnames = [f for f in os.listdir(paths["data_root"] + "checkpoints/") if "mnist_msl" in f]
    for expname in fnames:
        if "skew" in expname:
            skew = 1
        elif "sym" in expname:
            skew = 0

        # get name of .ckpt file from this folder
        exp_path = paths["data_root"] + f"checkpoints/{expname}/"
        ckpt_files = [f for f in os.listdir(exp_path) if f.endswith(".ckpt") and term in f]
        assert len(ckpt_files) == 1, "Expecting a single checkpoint file."
        ckpt_file = exp_path + ckpt_files[0]

        print(f"====={expname}---{ckpt_file}")

        # data
        datamodule = PyGAnnDataGraphDataModule(
            data_dir=paths["data_root"],
            file_names=["mnist.h5ad"],
            cell_type="subclass",
            spatial_coords=["x_section", "y_section", "z_section"],
            batch_size=200,
            n_hops=2,
        )

        model = LitGNNHetReg2dMSL.load_from_checkpoint(
            ckpt_file,
            input_size=n_genes,
            n_labels=n_labels,
            skew=skew,
        )

        # setup for predictions
        trainer = L.Trainer()  # use limit_predict_batches for test runs
        predictions = trainer.predict(model, datamodule=datamodule)
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

        # Get adata fields
        path = get_paths()["data_root"] + "mnist.h5ad"
        adata = ad.read_h5ad(path)
        xy = adata.obs[["x_section", "y_section"]].values
        # derive colors from matplotlib pastel palette using the categories in subclass
        unique_subclasses = adata.obs["subclass"].unique()
        pastel_palette = sns.color_palette("pastel", len(unique_subclasses))
        subclass_color_map = dict(zip(unique_subclasses, pastel_palette))
        adata.obs["subclass_color"] = adata.obs["subclass"].astype(int).map(subclass_color_map)

        # data for clustering
        data = {
            "eigvals": eigvals,
            "eigvecs": eigvecs,
            "xy_cov_pred": xy_cov_pred,
            "xy_mu_pred": xy_mu_pred,
            "xy_gamma_pred": xy_gamma_pred,
            "xy": xy,
            "section_idx": adata.obs["z_section"].values,
            "subclass": adata.obs["subclass"].values,
            "subclass_color": adata.obs["subclass_color"].values,
        }

        result_name = "_".join(expname.split("_")[2:]) + "_" + term
        print(f"Saving {result_name}")
        pickle.dump(data, open(paths["data_root"] + f"/results/{result_name}.pkl", "wb"))
