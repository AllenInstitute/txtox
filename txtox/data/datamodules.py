from pathlib import Path

import anndata as ad
import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch_geometric.data import Data as PyGData
from torch_geometric.loader.neighbor_loader import NeighborLoader
from torch_geometric.transforms import NodePropertySplit, RandomNodeSplit

from txtox.data.datasets import AnnDataDataset, AnnDataGraphDataset, PyGAnnData
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
        self.adatas = []
        for adata_path in self.adata_paths:
            self.adatas.append(AnnDataDataset(adata_path))
        self.data_full = ConcatDataset(self.adatas)
        self.data_train, self.data_test = random_split(
            self.data_full, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
        )

        if stage == "fit":
            self.data_train, self.data_val = random_split(
                self.data_train, [0.8, 0.2], generator=torch.Generator().manual_seed(1)
            )

        if stage == "test":  # Note: this is not the test set. Just a quick way to check the model through lightining.
            _, self.data_test = random_split(self.data_full, [0.9, 0.1], generator=torch.Generator().manual_seed(0))

        if stage == "predict":
            self.data_predict = self.data_full

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)


class AnnDataGraphDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: None,
        file_names: list[str] = ["VISp_nhood.h5ad"],
        batch_size: int = 1,
        celltype: str = "subclass",
        spatial_coords: list[str] = ["x_ccf", "y_ccf", "z_ccf"],
    ):
        super().__init__()
        if data_dir is None:
            data_dir = get_paths()["data_root"]
        self.adata_paths = [str(data_dir) + file_name for file_name in file_names]
        for adata_path in self.adata_paths:
            if not Path(adata_path).exists():
                raise FileNotFoundError(f"File not found: {adata_path}")

        self.celltype = celltype
        self.spatial_coords = spatial_coords
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.adatas = []
        for adata_path in self.adata_paths:
            self.adatas.append(
                AnnDataGraphDataset(adata_path, cell_type=self.celltype, spatial_coords=self.spatial_coords)
            )
        self.data_full = ConcatDataset(self.adatas)
        self.data_train, self.data_test = random_split(
            self.data_full, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
        )

        if stage == "fit":
            self.data_train, self.data_val = random_split(
                self.data_train, [0.8, 0.2], generator=torch.Generator().manual_seed(1)
            )

        if stage == "test":  # Note: this is not the test set. Just a quick way to check the model through lightining.
            _, self.data_test = random_split(self.data_full, [0.9, 0.1], generator=torch.Generator().manual_seed(0))

        if stage == "predict":
            self.data_predict = self.data_full

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)


class PyGAnnDataGraphDataModule(L.LightningDataModule):
    """
    Data module using PyG functions to return graph patches.
    """

    def __init__(
        self,
        data_dir: None,
        file_names: list[str] = ["VISp_nhood.h5ad"],
        batch_size: int = 1,
        n_hops: int = 2,
        split_method: str = "pass",
        cell_type: str = "subclass",
        spatial_coords: list[str] = ["x_section", "y_section", "z_section"],
        d_threshold: float = 1e7,
    ):
        super().__init__()
        if data_dir is None:
            data_dir = get_paths()["data_root"]
        self.adata_paths = [str(data_dir) + file_name for file_name in file_names]
        self.batch_size = batch_size
        self.n_hops = n_hops
        self.split_method = split_method
        self.cell_type = cell_type
        self.spatial_coords = spatial_coords
        self.d_threshold = d_threshold

    def setup(self, stage: str):
        # including self.dataset for debugging.
        # consider removing this if we run into cpu memory limits.
        self.dataset = PyGAnnData(
            self.adata_paths, cell_type=self.cell_type, spatial_coords=self.spatial_coords, d_threshold=self.d_threshold
        )
        self.data = self.dataset.get_pygdata_obj()

    def train_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=32,
            input_nodes=self.data.train_mask,
        )

    def val_dataloader(self):
        return NeighborLoader(
            self.data,
            input_nodes=self.data.val_mask,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
        )

    def predict_dataloader(self):
        return NeighborLoader(
            self.data,
            input_nodes=None,
            num_neighbors=[-1] * self.n_hops,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
        )


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

    print("anndatadatamodule checks passed")
    return datamodule


def test_pyg_anndatagraphdatamodule():
    import numpy as np

    from txtox.data.datamodules import PyGAnnDataGraphDataModule
    from txtox.utils import get_paths

    path = get_paths()["data_root"]
    datamodule = PyGAnnDataGraphDataModule(data_dir=path, file_names=["VISp_nhood.h5ad"], batch_size=1, n_hops=2)
    datamodule.setup(stage="fit")
    dataloader = iter(datamodule.train_dataloader())

    for i in range(3):
        batch = next(dataloader)

        # 2-hop neighborhood from NeighborLoader:
        nhood = batch.n_id.numpy()
        nhood = list(set(nhood))
        nhood.sort()

        # calculate 2-hop neighborhood directly:
        ref_cell = batch.input_id.numpy()  # cell index from which 2-hop neighborhood is calculated.
        nhood_ = np.where(datamodule.dataset.adata.obsp["spatial_connectivities"][ref_cell, :].toarray())[1]
        nhood_ = set(nhood_)
        for i in nhood_:
            ref_cell_nhood_2 = np.where(datamodule.dataset.adata.obsp["spatial_connectivities"][i, :].toarray())[1]
            ref_cell_nhood_2 = set(ref_cell_nhood_2)
            nhood_ = nhood_.union(ref_cell_nhood_2)
        nhood_ = list(nhood_)
        nhood_.sort()

        assert len(set(nhood_) - set(nhood)) == 0, "Difference between ref_cell_nhood and nhood is not empty"

    print("anndatagraphdatamodule checks passed")

    return


if __name__ == "__main__":
    test_anndatadatamodule()
    test_pyg_anndatagraphdatamodule()
