
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from txtox.data.datamodules import AnnDataDataModule
from txtox.models.simple import LitMLPv0
from txtox.utils import get_paths


def main():
    paths = get_paths()
    log_path = paths["data_root"] + "logs/tests"
    
    
    tb_logger = TensorBoardLogger(save_dir=log_path)
    datamodule = AnnDataDataModule(data_dir=paths["data_root"], file_names=["VISp.h5ad"], batch_size=100)
    model = LitMLPv0(input_size=500, n_labels=126, weight_mse=1.0, weight_ce=1.0)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=100, logger=tb_logger)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
