import warnings

import anndata as ad
from anndata._core.aligned_df import ImplicitModificationWarning
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp

from txtox.utils import get_paths

warnings.filterwarnings("ignore", category=ImplicitModificationWarning, message="Transforming to str index.")


class AnnDataDataset(Dataset):
    """
    Tabular AnnData dataset.

    Args:
        path (str): Path to the AnnData file.
        keep_genes (list): List of genes to keep.
        keep_cells (list): List of cells to keep.
        spatial_coords (list): Anndata.obs columns for spatial coordinates
        cell_type (str): Anndata.obs column for cell types.
    """

    def __init__(
        self, path, keep_genes=None, keep_cells=None, spatial_coords=["x_ccf", "y_ccf", "z_ccf"], cell_type="supertype"
    ):
        super().__init__()
        self.path = path

        adata = ad.read_h5ad(self.path)

        # filter genes
        if keep_genes is not None:
            adata = adata[:, keep_genes].copy()
        else:
            keep_genes = get_non_blank_genes(adata)
            adata = adata[:, keep_genes].copy()

        # filter cells
        if keep_cells is not None:
            adata = adata[keep_cells, :].copy()

        self.adata = adata
        self.spatial_coords = spatial_coords
        self.cell_type = cell_type
        self.cell_type_list = adata.obs[cell_type].cat.categories.tolist()
        self.cell_type_labelencoder = LabelEncoder()
        self.cell_type_labelencoder.fit(self.cell_type_list)
        self.data_issparse = issparse(adata.X)

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        gene_exp = self.adata.X[idx, :]
        if self.data_issparse:
            gene_exp = (
                gene_exp.toarray()
                .astype(np.float32)
                .reshape(
                    -1,
                )
            )
        xyz = self.adata.obs.iloc[idx][self.spatial_coords].values.astype(np.float32)
        celltype = self.cell_type_labelencoder.transform([self.adata.obs.iloc[idx][self.cell_type]])

        return gene_exp, xyz, celltype


class AnnDataGraphDataset(Dataset):
    """
    Tabular AnnData dataset.

    Args:
        path (str): Path to the AnnData file.
        keep_genes (list): List of genes to keep.
        keep_cells (list): List of cells to keep.
        spatial_coords (list): Anndata.obs columns for spatial coordinates (for regression)
        cell_type (str): Anndata.obs column for cell types.
        max_order (int): Maximum order of neighbors to consider.
        d_threshold (float): Distance threshold (in mm) for considering neighbors.
    """

    def __init__(
        self,
        path,
        keep_genes=None,
        keep_cells=None,
        spatial_coords=["x_ccf", "y_ccf", "z_ccf"],
        cell_type="supertype",
        max_order=2,
        d_threshold=1000,
    ):
        super().__init__()
        self.path = path

        adata = ad.read_h5ad(self.path)
        assert (
            "spatial_connectivities" in adata.obsp.keys()
        ), "Spatial connectivities not found. Run `sc.pp.neighbors` first."
        assert "spatial_distances" in adata.obsp.keys(), "Spatial distances not found. Run `sc.pp.neighbors` first."

        # filter genes
        if keep_genes is not None:
            adata = adata[:, keep_genes].copy()
        else:
            keep_genes = get_non_blank_genes(adata)
            adata = adata[:, keep_genes].copy()

        # filter cells
        if keep_cells is not None:
            adata = adata[keep_cells, :].copy()

        self.adata = adata
        self.max_order = max_order
        self.d_threshold = d_threshold

        # create binary adjacency matrix without self-loops
        adj = self.adata.obsp["spatial_connectivities"].copy()
        adj = adj.astype(bool).astype(int)
        adj[self.adata.obsp["spatial_distances"] > self.d_threshold] = 0
        adj.setdiag(0)

        # create adjacency matrices up to max_order
        self.adj_matrices = {}
        self.adj_matrices[1] = adj.copy()
        if self.max_order > 1:
            for i in range(2, self.max_order + 1):
                self.adj_matrices[i] = adj.dot(self.adj_matrices[i - 1])

        self.spatial_coords = spatial_coords
        self.cell_type = cell_type
        self.cell_type_list = adata.obs[cell_type].cat.categories.tolist()
        self.cell_type_labelencoder = LabelEncoder()
        self.cell_type_labelencoder.fit(self.cell_type_list)
        self.data_issparse = issparse(adata.X)

    def get_neighbors(self, idx):
        """
        Get all neighbors up to max_order. Self is included as the last entry
        """
        nhood_idx = []
        for i in range(1, self.max_order + 1):
            nhood_idx.append(np.where(self.adj_matrices[i][idx, :].toarray().flatten())[0])
        nhood_idx = np.unique(np.concatenate(nhood_idx, axis=0))
        # remove idx from nhood_idx if it exists
        nhood_idx = np.setdiff1d(nhood_idx, [idx])
        # place it at the end
        nhood_idx = np.concatenate([nhood_idx, [idx]])
        return nhood_idx

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        # get all neighbors
        nhood_idx = self.get_neighbors(idx)
        local_adj = self.adj_matrices[1][np.ix_(nhood_idx, nhood_idx)]
        edgelist = np.array(local_adj.nonzero()).T

        gene_exp = self.adata.X[nhood_idx, :]
        if self.data_issparse:
            gene_exp = (gene_exp.toarray().astype(np.float32))
        xyz = self.adata.obs.iloc[nhood_idx][self.spatial_coords].values.astype(np.float32)
        celltype = self.cell_type_labelencoder.transform(self.adata.obs.iloc[nhood_idx][self.cell_type])
        return gene_exp, edgelist, xyz, celltype


def get_non_blank_genes(adata):
    keep_genes = adata.var[~adata.var.index.str.startswith("Blank")].index
    return keep_genes


def test_anndatadataset():
    from txtox.data.datasets import AnnDataDataset
    from txtox.utils import get_paths

    paths = get_paths()
    dataset = AnnDataDataset(
        path=paths["data_root"] + "VISp.h5ad",
        keep_genes=None,
        keep_cells=None,
        spatial_coords=["x_ccf", "y_ccf", "z_ccf"],
        cell_type="supertype",
    )
    gene_exp, xyz, celltype = dataset.__getitem__([1])
    return dataset


def test_anndatagraphdataset():
    from txtox.data.datasets import AnnDataGraphDataset
    from txtox.utils import get_paths

    paths = get_paths()
    dataset = AnnDataGraphDataset(
        path=paths["data_root"] + "VISp_nhood.h5ad",
        keep_genes=None,
        keep_cells=None,
        spatial_coords=["x_ccf", "y_ccf", "z_ccf"],
        cell_type="supertype",
        max_order=2,
        d_threshold=1000,
    )
    gene_exp, local_adj, xyz, celltype = dataset.__getitem__(0)
    return dataset


if __name__ == "__main__":
    test_anndatagraphdataset()
