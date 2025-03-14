import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from txtox.data.datamodules import AnnDataGraphDataModule
from txtox.models.gnn import LitGNNv0
from txtox.utils import get_datetime, get_paths


def main():
    # data parameters, we'll eventually obtain this from the data.
    n_genes = 500
    n_labels = 126

    # paths
    paths = get_paths()
    expname = get_datetime(expname="VISp_nhood_GNN")
    log_path = paths["data_root"] + f"logs/{expname}"
    checkpoint_path = paths["data_root"] + f"checkpoints/{expname}"

    # helpers
    tb_logger = TensorBoardLogger(save_dir=log_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, monitor="val_rmse_overall", filename="{epoch}-{val_rmse_overall:.2f}"
    )

    # data, model and fitting
    datamodule = AnnDataGraphDataModule(data_dir=paths["data_root"], file_names=["VISp_nhood.h5ad"], batch_size=1)
    model = LitGNNv0(input_dim=n_genes, n_labels=n_labels, weight_mse=1.0, weight_ce=0.1)
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
