from pathlib import Path

import anndata as ad
import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from txtox.data.datasets import AnnDataDataset
from txtox.utils import get_paths


class AnnDataDataModule(L.LightningDataModule):
    def __init__(self, data_dir: None, file_names: list[str] = ["VISp.h5ad"], batch_size: int = 100):
        super().__init__()
        if data_dir is None:
            data_dir = get_paths()["data_root"]
        self.adata_paths = [str(data_dir) + file_name for file_name in file_names]
        for adata_path in self.adata_paths:
            if not Path(adata_path).exists():
                raise FileNotFoundError(f"File not found: {adata_path}")

        self.batch_size = batch_size


    def setup(self, stage: str):
        if stage == "fit":
            self.adatas = []
            for adata_path in self.adata_paths:
                self.adatas.append(AnnDataDataset(adata_path))
            self.data_full = ConcatDataset(self.adatas)
            self.data_train, self.data_test = random_split(
                self.data_full, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
            )
            self.data_train, self.data_val = random_split(
                self.data_train, [0.8, 0.2], generator=torch.Generator().manual_seed(1)
            )

        if stage == "test":
            self.data_test = self.data_test

        if stage == "predict":
            self.data_predict = self.data_full

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size, shuffle=False)


def test_anndatadatamodule():
    from txtox.data.datamodules import AnnDataDataModule
    from txtox.utils import get_paths

    path = get_paths()["data_root"]
    datamodule = AnnDataDataModule(data_dir=path, file_names=["VISp.h5ad", "VISp_2.h5ad"])
    datamodule.setup(stage="fit")
    print(f"full: {len(datamodule.data_full)}")
    print(f"train: {len(datamodule.data_train)}")
    print(f"val: {len(datamodule.data_val)}")
    print(f"test: {len(datamodule.data_test)}")
    return datamodule