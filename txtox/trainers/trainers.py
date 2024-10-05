import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from txtox.data.datamodules import AnnDataDataModule
from txtox.models.simple import LitMLPv0
from txtox.utils import get_datetime, get_paths


def main():
    # paths
    paths = get_paths()
    expname = get_datetime(expname="test")
    log_path = paths["data_root"] + f"logs/{expname}"
    checkpoint_path = paths["data_root"] + f"checkpoints/{expname}"

    # helpers
    tb_logger = TensorBoardLogger(save_dir=log_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, monitor="val_rmse_overall", filename="{epoch}-{val_rmse_overall:.2f}"
    )

    # data, model and fitting
    datamodule = AnnDataDataModule(data_dir=paths["data_root"], file_names=["VISp.h5ad"], batch_size=100)
    model = LitMLPv0(input_size=500, n_labels=126, weight_mse=1.0, weight_ce=1.0)
    trainer = L.Trainer(limit_train_batches=10, max_epochs=10, logger=tb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
