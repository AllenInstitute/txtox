import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from txtox.data.datamodules import PyGAnnDataGraphDataModule
from txtox.models.gnn_het_reg_2d_fix_pyg import LitGNNHetRegGauss2d
from txtox.utils import get_datetime, get_paths


def main():
    # data parameters, we'll eventually obtain this from the data.
    n_genes = 500
    n_labels = 158  # changed to 158 for test_one_section_hemi.h5ad

    # paths
    paths = get_paths()
    expname = get_datetime(expname="one_sec_hemi_nhood_GNNhetreg_GNLL2d_pyg")
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
        file_names=["test_one_section_hemi.h5ad"],
        cell_type="subclass",
        spatial_coords=["x_section", "y_section", "z_section"],
        batch_size=5,
        n_hops=2,
    )

    # model
    model = LitGNNHetRegGauss2d(input_dim=n_genes, n_labels=n_labels, weight_gnll=1.0, weight_ce=0.1)

    # fit wrapper
    trainer = L.Trainer(
        limit_train_batches=1000,
        limit_val_batches=100,
        max_epochs=200,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
