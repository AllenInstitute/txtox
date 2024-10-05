from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import toml


def get_paths(verbose: bool = False) -> dict:
    """
    Get custom paths from config.toml that is in the root directory.
    """

    # get path of this file
    root_path = Path(__file__).parent.parent
    config_path = root_path / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    config = toml.load(config_path)
    config["package_root"] = root_path
    if verbose:
        print(config)
    return config


def get_datetime(expname: str = ""):
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if expname is None:
        expname = datetime_str
    else:
        expname = f"{datetime_str}_{expname}"
    return expname


def get_adata(path: str):
    adata = ad.read_h5ad(path)
    adata.obsm["ccf"] = np.concatenate(
        (
            np.expand_dims(np.array(adata.obs["x_ccf"]), axis=1),
            np.expand_dims(np.array(adata.obs["y_ccf"]), axis=1),
            np.expand_dims(np.array(adata.obs["z_ccf"]), axis=1),
        ),
        axis=1,
    )
    adata.var.set_index("gene_symbol", inplace=True, drop=False)

    return adata


if __name__ == "__main__":
    pass
