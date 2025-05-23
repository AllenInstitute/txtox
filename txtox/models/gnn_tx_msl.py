# heteroscedastic regression (predicting variance per-sample) with GNNs.

# In this version:
# section coordinates are used.
# z-section is provided as input.
# mean and covariance is only learnt for x and y dimensions.
# loss + forward calculation is modified based on recommendation in Stirn et al. 2023 (equation 5)

import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATv2Conv
from torchmetrics import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy

EULER_GAMMA = np.euler_gamma
PI = np.pi


class LitGNNHetReg2dMSL(L.LightningModule):
    def __init__(self, input_dim=500, hidden_dim=10, n_labels=126, weight_msl_nll=1.0, weight_ce=1.0, skew=False):
        super(LitGNNHetReg2dMSL, self).__init__()

        self.skew = skew
        self.weight_msl_nll = weight_msl_nll
        self.weight_ce = weight_ce

        # fmt:off
        n_heads = 5
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=n_heads, concat=True)
        self.conv2 = GATv2Conv(hidden_dim*n_heads, hidden_dim, heads=n_heads, concat=True)
        self.skip = torch.nn.Linear(input_dim, hidden_dim*n_heads)
        self.encoder = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(hidden_dim*n_heads, 20), torch.nn.LayerNorm(20))

        self.spatial_mu_out = torch.nn.Sequential(torch.nn.Linear(21, 20), torch.nn.GELU(), torch.nn.Linear(20, 2))
        self.spatial_l_out = torch.nn.Sequential(torch.nn.Linear(21, 20), torch.nn.GELU(), torch.nn.Linear(20, 3))
        self.spatial_gamma_out = torch.nn.Sequential(torch.nn.Linear(21, 20), torch.nn.GELU(), torch.nn.Linear(20, 2))
        self.label_out = torch.nn.Sequential(torch.nn.Linear(21, 20), torch.nn.GELU(), torch.nn.Linear(20, n_labels))
        self.gelu = torch.nn.GELU()

        # losses
        self.loss_msl_nll2d = MultivariateSkewLaplaceNLLLoss2d()
        self.loss_ce = nn.CrossEntropyLoss()

        # training metrics
        self.train_metric_rmse = MeanSquaredError(squared=False, num_outputs=2)
        self.train_metric_rmse_overall = MeanSquaredError(squared=False, num_outputs=1)
        self.train_metric_overall_acc = MulticlassAccuracy(num_classes=n_labels, top_k=1, average="weighted", multidim_average="global")
        self.train_metric_macro_acc = MulticlassAccuracy(num_classes=n_labels, top_k=1, average="macro", multidim_average="global")
        self.train_metric_multiclass_acc = MulticlassAccuracy(num_classes=n_labels, top_k=1, average=None, multidim_average="global")

        # validation metrics
        self.val_metric_rmse = MeanSquaredError(squared=False, num_outputs=2)
        self.val_metric_rmse_overall = MeanSquaredError(squared=False, num_outputs=1)
        self.val_metric_overall_acc = MulticlassAccuracy(num_classes=n_labels, top_k=1, average="weighted", multidim_average="global")
        self.val_metric_macro_acc = MulticlassAccuracy(num_classes=n_labels, top_k=1, average="macro", multidim_average="global")
        self.val_metric_multiclass_acc = MulticlassAccuracy(num_classes=n_labels, top_k=1, average=None, multidim_average="global")

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
        xy_gamma = self.spatial_gamma_out(x)
        celltype = self.label_out(x)
        if not self.skew:
            xy_gamma = xy_gamma.detach()
            xy_gamma = torch.zeros_like(xy_gamma)
        return xy_mu, xy_L, xy_gamma, celltype

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
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        gene_exp, edgelist, xy, section_idx, celltype = self.proc_batch(batch)
        xy_mu_pred, xy_L_pred, xy_gamma_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)

        # Training loss should be calculated for all nodes (including input_nodes and neighbors).
        # Validation nodes can be part of the neighborhood; loss calculation should be done over training nodes.
        # batch_size passed to NeighborLoader refers to the number of input_nodes only.
        batch_size = batch["train_mask"].sum()

        # Calculate losses
        msl_nll_loss = self.loss_msl_nll2d(
            xy_mu_pred[batch["train_mask"]],
            xy_L_pred[batch["train_mask"]],
            xy_gamma_pred[batch["train_mask"]],
            xy[batch["train_mask"]],
        )
        ce_loss = self.loss_ce(celltype_pred[batch["train_mask"]], celltype[batch["train_mask"]].squeeze())
        total_loss = self.weight_msl_nll * msl_nll_loss + self.weight_ce * ce_loss

        # Log losses - provide batch_size to self.log.
        log_config = dict(on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_msl_nll_loss", msl_nll_loss, batch_size=batch_size, **log_config)
        self.log("train_ce_loss", ce_loss, batch_size=batch_size, **log_config)
        self.log("train_total_loss", total_loss, batch_size=batch_size, **log_config)

        # Calculate metrics (these are torchmetrics objects)
        _pred = celltype_pred[batch["train_mask"]]
        _target = celltype[batch["train_mask"]].reshape(-1)
        self.train_metric_overall_acc(preds=_pred, target=_target)
        self.train_metric_macro_acc(preds=_pred, target=_target)
        self.train_metric_rmse_overall(xy_mu_pred[batch["train_mask"]], xy[batch["train_mask"]])
        # Non-scalar values, computed+reset at epoch end manually.
        # See: https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html
        self.train_metric_rmse.update(xy_mu_pred[batch["train_mask"]], xy[batch["train_mask"]])

        # Log metrics
        # fmt:off
        self.log("train_rmse_overall", self.train_metric_rmse_overall, **log_config)
        self.log("train_overall_acc", self.train_metric_overall_acc, **log_config)
        self.log("train_macro_acc", self.train_metric_macro_acc, **log_config)
        return total_loss

    def on_train_epoch_end(self):
        train_metric_rmse = self.train_metric_rmse.compute()
        log_dict = {"train_x_rmse": train_metric_rmse[0], "train_y_rmse": train_metric_rmse[1]}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.train_metric_rmse.reset()

    def validation_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        gene_exp, edgelist, xy, section_idx, celltype = self.proc_batch(batch)
        xy_mu_pred, xy_L_pred, xy_gamma_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)

        # Validation metrics should only be calculated for the input_nodes,
        # and not their neighbors (which are allowed to be part of the training set)
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]
        batch_size = idx.shape[0]

        # slice the data for metrics not calculated for neighbors.
        xy_ = xy[idx]
        celltype_ = celltype[idx]
        xy_mu_pred_ = xy_mu_pred[idx]
        xy_L_pred_ = xy_L_pred[idx]
        xy_gamma_pred_ = xy_gamma_pred[idx]
        celltype_pred_ = celltype_pred[idx]

        # Calculate losses
        msl_nll_loss = self.loss_msl_nll2d(
            xy_mu_pred_,
            xy_L_pred_,
            xy_gamma_pred_,
            xy_,
        )
        ce_loss = self.loss_ce(celltype_pred_, celltype_.squeeze())
        total_loss = self.weight_msl_nll * msl_nll_loss + self.weight_ce * ce_loss

        # Log losses - provide batch_size to self.log.
        log_config = dict(on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_msl_nll_loss", msl_nll_loss, batch_size=batch_size, **log_config)
        self.log("val_ce_loss", ce_loss, batch_size=batch_size, **log_config)
        self.log("val_total_loss", total_loss, batch_size=batch_size, **log_config)

        # Calculate metrics
        self.val_metric_rmse.update(xy_mu_pred_, xy_)
        self.val_metric_rmse_overall.update(xy_mu_pred_, xy_)
        self.val_metric_overall_acc.update(preds=celltype_pred_, target=celltype_.reshape(-1))
        self.val_metric_macro_acc.update(preds=celltype_pred_, target=celltype_.reshape(-1))
        # self.val_metric_multiclass_acc.update(preds=celltype_pred_, target=celltype_.reshape(-1))

        log_config = dict(on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_rmse_overall", self.val_metric_rmse_overall, **log_config)
        self.log("val_overall_acc", self.val_metric_overall_acc, **log_config)
        self.log("val_macro_acc", self.val_metric_macro_acc, **log_config)

    def on_validation_epoch_end(self):
        val_metric_rmse = self.val_metric_rmse.compute()
        log_dict = {"val_x_rmse": val_metric_rmse[0], "val_y_rmse": val_metric_rmse[1]}
        self.log_dict(log_dict, on_epoch=True, prog_bar=False, logger=True)
        self.val_metric_rmse.reset()

    def test_step(self, batch, batch_idx):
        gene_exp, edgelist, xy, section_idx, celltype = self.proc_batch(batch)
        xy_mu_pred, xy_L_pred, xy_gamma_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)
        return (
            xy_mu_pred.to("cpu").numpy(),
            xy_L_pred.to("cpu").numpy(),
            xy_gamma_pred.to("cpu").numpy(),
            celltype_pred.to("cpu").numpy(),
        )

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx):
        gene_exp, edgelist, xy, section_idx, celltype = self.proc_batch(batch)
        xy_mu_pred, xy_L_pred, xy_gamma_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)

        # Validation metrics should only be calculated for the input_nodes,
        # and not their neighbors (which are allowed to be part of the training set)
        idx = torch.where(batch.n_id == batch.input_id.unsqueeze(-1))[0]
        batch_size = idx.shape[0]

        xy_mu_pred = xy_mu_pred[idx].to("cpu").numpy()
        xy_L_pred = xy_L_pred[idx].to("cpu").numpy()
        xy_gamma_pred = xy_gamma_pred[idx].to("cpu").numpy()
        celltype_pred = celltype_pred[idx].to("cpu").numpy()
        return xy_mu_pred, xy_L_pred, xy_gamma_pred, celltype_pred

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


class MultivariateSkewLaplaceNLLLoss2d(nn.Module):
    def __init__(self, reduce=True, normalize=True):
        super(MultivariateSkewLaplaceNLLLoss2d, self).__init__()
        self.reduce = reduce
        self.normalize = normalize
        p = torch.as_tensor(2)  # dimensionality
        self.register_buffer(
            "norm",
            p * math.log(2) + ((p - 1) / 2) * math.log(math.pi) + torch.lgamma((p + 1) / 2),
        )

    def forward(self, mu, L, gamma, x):
        sigma_inv = torch.cholesky_inverse(L)  # batch_size, 2, 2
        alpha = torch.sqrt(1 + torch.einsum("bi,bij,bj -> b", gamma, sigma_inv, gamma))  # batch_size,
        diff = x - mu  # batch_size, 2
        root_exp = (torch.einsum("bi,bij,bj->b", diff, sigma_inv, diff) + 1e-8).sqrt()
        add_exp = torch.einsum("bi,bij,bj-> b", diff, sigma_inv, gamma)
        constants = 0.5 * torch.logdet(sigma_inv) - torch.log(alpha)
        if self.normalize:
            constants = constants - self.norm
        log_prob = constants - alpha * root_exp + add_exp

        if self.reduce:
            return -torch.mean(log_prob)
        else:
            return -log_prob
