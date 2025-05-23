# heteroscedastic regression (variable variance per-sample) with GNNs with normal distribution.

# In this version:
# section coordinates are used.
# z-section is provided as input.
# Only x and y section coordinates are learned.
# PyG dataloaders: "minibatching" involves multiple input_nodes and their neighbors.
# loss + forward calculation is modified based on recommendation in Stirn et al. 2023 (equation 5)

import lightning as L
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATv2Conv
from torchmetrics import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy


class LitGNNHetRegGauss2d(L.LightningModule):
    def __init__(self, input_dim=500, hidden_dim=10, n_labels=126, weight_gnll=1.0, weight_ce=1.0):
        super(LitGNNHetRegGauss2d, self).__init__()

        self.weight_gnll = weight_gnll
        self.weight_ce = weight_ce

        # fmt:off
        n_heads = 5
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=n_heads, concat=True)
        self.conv2 = GATv2Conv(hidden_dim*n_heads, hidden_dim, heads=n_heads, concat=True)
        self.skip = torch.nn.Linear(input_dim, hidden_dim*n_heads)
        self.encoder = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(hidden_dim*n_heads, 20), torch.nn.LayerNorm(20))

        self.spatial_mu_out = torch.nn.Sequential(torch.nn.Linear(21, 20), torch.nn.GELU(), torch.nn.Linear(20, 2))
        self.spatial_l_out = torch.nn.Sequential(torch.nn.Linear(21, 20), torch.nn.GELU(), torch.nn.Linear(20, 3))
        self.label_out = torch.nn.Sequential(torch.nn.Linear(21, 20), torch.nn.GELU(), torch.nn.Linear(20, n_labels))
        self.gelu = torch.nn.GELU()

        # losses
        self.loss_l2norm = L2NormLoss()
        self.loss_gnll2d = GaussianNLLLoss2d()
        self.loss_ce = nn.CrossEntropyLoss()

        # metrics
        self.metric_rmse = MeanSquaredError(squared=False, num_outputs=2)
        self.metric_rmse_overall = MeanSquaredError(squared=False, num_outputs=1)
        self.metric_overall_acc = MulticlassAccuracy(num_classes=n_labels, top_k=1, average="weighted", multidim_average="global")
        self.metric_macro_acc = MulticlassAccuracy(num_classes=n_labels, top_k=1, average="macro", multidim_average="global")
        self.metric_multiclass_acc = MulticlassAccuracy(num_classes=n_labels, top_k=1, average=None, multidim_average="global")

    def forward(self, x, section_idx, edge_index):
        # x is gene expression.
        # section_idx is the section co-ordinate.

        y = self.conv1(x, edge_index)
        y = self.gelu(y)
        y = self.conv2(y, edge_index)
        y = self.gelu(y)
        x = self.skip(x) + y
        x = self.encoder(x)

        x = torch.cat([x, section_idx], dim=1)
        xy_mu = self.spatial_mu_out(x)
        xy_L = vec2mat_cholesky2d(self.spatial_l_out(x.detach()))
        celltype = self.label_out(x)
        return xy_mu, xy_L, celltype

    def proc_batch(self, batch):
        # model specific processing of the batch.

        gene_exp, edgelist, xyz, celltype = batch.gene_exp, batch.edge_index.T, batch.xyz, batch.celltype
        gene_exp = gene_exp.squeeze(dim=0)
        edgelist = edgelist.squeeze(dim=0).T
        xyz = xyz.squeeze(dim=0)

        # hardcoded: third dimension is section co-ordinate.
        xy = xyz[:, :2].reshape(-1, 2).contiguous()
        section_idx = xyz[:, 2].reshape(-1, 1)

        celltype = celltype.squeeze(dim=0)
        return gene_exp, edgelist, xy, section_idx, celltype

    def training_step(self, batch, batch_idx):
        # for GNN there isn't a batch dimension.
        gene_exp, edgelist, xy, section_idx, celltype = self.proc_batch(batch)
        m_pred, L_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)

        # Training loss should be calculated for all nodes (including input_nodes and neighbors).
        # batch_size passed to NeighborLoader refers to the number of input_nodes only.
        batch_size = batch.gene_exp.size(0)

        # Calculate losses
        l2norm_loss = 0.5 * self.loss_l2norm(m_pred[batch["train_mask"]], xy[batch["train_mask"]])
        gnll_loss = self.loss_gnll2d(m_pred[batch["train_mask"]], L_pred[batch["train_mask"]], xy[batch["train_mask"]])
        ce_loss = self.loss_ce(celltype_pred[batch["train_mask"]], celltype[batch["train_mask"]].squeeze())
        total_loss = self.weight_gnll * l2norm_loss + self.weight_gnll * gnll_loss + self.weight_ce * ce_loss

        # Log losses
        self.log(
            "train_gnll_loss", gnll_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size
        )
        self.log(
            "train_ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size
        )
        self.log(
            "train_total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        # Calculate metrics
        train_metric_rmse_overall = self.metric_rmse_overall(m_pred[batch["train_mask"]], xy[batch["train_mask"]])
        train_overall_acc = self.metric_overall_acc(
            preds=celltype_pred[batch["train_mask"]], target=celltype[batch["train_mask"]].reshape(-1)
        )
        train_macro_acc = self.metric_macro_acc(
            preds=celltype_pred[batch["train_mask"]], target=celltype[batch["train_mask"]].reshape(-1)
        )
        train_metric_rmse = self.metric_rmse(m_pred[batch["train_mask"]], xy[batch["train_mask"]])

        # Log metrics
        # fmt:off
        self.log("train_rmse_overall", train_metric_rmse_overall, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_overall_acc", train_overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_macro_acc", train_macro_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        log_dict = {
            "train_x_rmse": train_metric_rmse[0],
            "train_y_rmse": train_metric_rmse[1]
        }
        # fmt:on
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)

        return total_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        gene_exp, edgelist, xy, section_idx, celltype = self.proc_batch(batch)
        m_pred, L_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)

        # Validation metrics should only be calculated for the input_nodes,
        # and not their neighbors (which are allowed to be part of the training set)
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]
        batch_size = idx.shape[0]

        # slice the data for metrics not calculated for neighbors.
        xy_ = xy[idx]
        celltype_ = celltype[idx]
        m_pred_ = m_pred[idx]
        L_pred_ = L_pred[idx]
        celltype_pred_ = celltype_pred[idx]

        # Calculate losses
        gnll_loss = self.loss_gnll2d(m_pred_, L_pred_, xy_)
        ce_loss = self.loss_ce(celltype_pred_, celltype_.squeeze())

        # Log losses
        self.log(
            "val_gnll_loss", gnll_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size
        )
        self.log("val_ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # Calculate metrics
        val_metric_rmse = self.metric_rmse(m_pred_, xy_)
        val_metric_rmse_overall = self.metric_rmse_overall(m_pred_, xy_)
        val_overall_acc = self.metric_overall_acc(preds=celltype_pred_, target=celltype_.reshape(-1))
        val_macro_acc = self.metric_macro_acc(preds=celltype_pred_, target=celltype_.reshape(-1))
        val_metric_multiclass_acc = self.metric_multiclass_acc(preds=celltype_pred_, target=celltype_.reshape(-1))

        log_dict = {"val_x_rmse": val_metric_rmse[0], "val_y_rmse": val_metric_rmse[1]}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        self.log(
            "val_rmse_overall",
            val_metric_rmse_overall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "val_overall_acc",
            val_overall_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "val_macro_acc",
            val_macro_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        gene_exp, edgelist, xy, section_idx, celltype = self.proc_batch(batch)
        m_pred, L_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)

        # slice the data for metrics not calculated for neighbors.
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]

        return (
            m_pred[idx].to("cpu").numpy(),
            L_pred[idx].to("cpu").numpy(),
            celltype_pred[idx].to("cpu").numpy(),
        )

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx):
        gene_exp, edgelist, xy, section_idx, celltype = self.proc_batch(batch)
        m_pred, L_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)

        # Validation metrics should only be calculated for the input_nodes,
        # and not their neighbors (which are allowed to be part of the training set)
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]
        batch_size = idx.shape[0]

        m_pred = m_pred[idx].to("cpu").numpy()
        L_pred = L_pred[idx].to("cpu").numpy()
        celltype_pred = celltype_pred[idx].to("cpu").numpy()
        return m_pred, L_pred, celltype_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def vec2mat_cholesky2d(l_vec):
    """
    Use `l_vec` entries to populate the lower triangular matrix L with positive diagonal elements.
    ```
    L = | L00  0  |
        | L10 L11 |
    ```
    Args:
        l_vec (torch.Tensor): (batch_size, 3)

    Returns:
        torch.Tensor: (batch_size, 2, 2)
    """
    L = torch.zeros((l_vec.size(0), 2, 2), dtype=l_vec.dtype, device=l_vec.device)

    eps = 1e-5  # add to diagonal for numerical stability
    L[:, 0, 0] = nn.functional.softplus(l_vec[:, 0]) + eps
    L[:, 1, 1] = nn.functional.softplus(l_vec[:, 1]) + eps
    L[:, 1, 0] = l_vec[:, 2]  # off-diagonal element
    return L


class GaussianNLLLoss2d(nn.Module):
    def __init__(self, reduce=True):
        super(GaussianNLLLoss2d, self).__init__()
        self.reduce = reduce

    def forward(self, mu, L, x):
        diff = (x - mu).unsqueeze(-1)  # (batch_size, 2, 1)
        z = torch.cholesky_solve(diff, L, upper=False)  # (batch_size, 2, 1)
        quadratic = torch.bmm(diff.transpose(1, 2), z).squeeze()  # (batch_size, )
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2)), dim=1)  # (batch_size, )
        nll = 0.5 * (quadratic + log_det + 2 * torch.log(torch.tensor(2 * torch.pi)))  # (batch_size, )
        if self.reduce:
            return torch.mean(nll)
        else:
            return nll


class L2NormLoss(nn.Module):
    def __init__(self, reduce=True):
        super(L2NormLoss, self).__init__()
        self.reduce = reduce

    def forward(self, mu, x):
        diff = x - mu  # (batch_size, 2)
        l2_norm = torch.sum(diff * diff, dim=1)  # (batch_size, x_dim)
        if self.reduce:
            return torch.mean(l2_norm)
        else:
            return l2_norm
