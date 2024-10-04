import warnings

import anndata as ad
from anndata._core.aligned_df import ImplicitModificationWarning
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import numpy as np

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
            gene_exp = gene_exp.toarray().astype(np.float32).reshape(-1,)
        xyz = self.adata.obs.iloc[idx][self.spatial_coords].values.astype(np.float32)
        celltype = self.cell_type_labelencoder.transform([self.adata.obs.iloc[idx][self.cell_type]])
    
        return gene_exp, xyz, celltype


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
    gene_exp, xyz, celltype = dataset.__getitem__([1, 200])
    return dataset
