# heteroscedastic regression (predicting variance per-sample) with GNNs.

# In this version:
# section coordinates are used.
# z-section is provided as input.
# mean and covariance is only learnt for x and y dimensions.
# NLL loss is updated accordingly.

import lightning as L
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATv2Conv
from torchmetrics import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy


class LitGNNHetReg2d(L.LightningModule):
    def __init__(self, input_dim=500, hidden_dim=10, n_labels=126, weight_gnll=1.0, weight_ce=1.0):
        super(LitGNNHetReg2d, self).__init__()

        self.weight_gnll = weight_gnll
        self.weight_ce = weight_ce

        # fmt:off
        n_heads = 5
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=n_heads, concat=True)
        self.conv2 = GATv2Conv(hidden_dim*n_heads, hidden_dim, heads=n_heads, concat=True)
        self.skip = torch.nn.Linear(input_dim, hidden_dim*n_heads)
        self.encoder = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(hidden_dim*n_heads, 20), torch.nn.GELU())

        self.spatial_mu_out = torch.nn.Sequential(torch.nn.Linear(21, 20), torch.nn.GELU(), torch.nn.Linear(20, 2))
        self.spatial_l_out = torch.nn.Sequential(torch.nn.Linear(21, 20), torch.nn.GELU(), torch.nn.Linear(20, 3))
        self.label_out = torch.nn.Sequential(torch.nn.Linear(21, 20), torch.nn.GELU(), torch.nn.Linear(20, n_labels))
        self.gelu = torch.nn.GELU()

        # losses
        # self.loss_mse = nn.MSELoss()
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
        xy_L = vec2mat_cholesky2d(self.spatial_l_out(x))
        celltype = self.label_out(x)
        return xy_mu, xy_L, celltype

    def proc_batch(self, batch):
        # model specific processing of the batch.

        gene_exp, edgelist, xyz, celltype = batch
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
        xy_mu_pred, xy_L_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)

        # Calculate losses
        gnll_loss = self.loss_gnll2d(xy_mu_pred, xy_L_pred, xy)
        ce_loss = self.loss_ce(celltype_pred, celltype.squeeze())
        total_loss = self.weight_gnll * gnll_loss + self.weight_ce * ce_loss

        # Log losses
        self.log("train_gnll_loss", gnll_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Calculate metrics
        train_metric_rmse_overall = self.metric_rmse_overall(xy_mu_pred, xy)
        train_overall_acc = self.metric_overall_acc(preds=celltype_pred, target=celltype.reshape(-1))
        train_macro_acc = self.metric_macro_acc(preds=celltype_pred, target=celltype.reshape(-1))
        train_metric_rmse = self.metric_rmse(xy_mu_pred, xy)

        # Log metrics
        # fmt:off
        self.log("train_rmse_overall", train_metric_rmse_overall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_overall_acc", train_overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_macro_acc", train_macro_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        log_dict = {
            "train_x_rmse": train_metric_rmse[0],
            "train_y_rmse": train_metric_rmse[1]
        }
        # fmt:on
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return total_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        gene_exp, edgelist, xy, section_idx, celltype = self.proc_batch(batch)
        xy_mu_pred, xy_L_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)

        # Calculate metrics
        val_metric_rmse = self.metric_rmse(xy_mu_pred, xy)
        val_metric_rmse_overall = self.metric_rmse_overall(xy_mu_pred, xy)
        val_overall_acc = self.metric_overall_acc(preds=celltype_pred, target=celltype.reshape(-1))
        val_macro_acc = self.metric_macro_acc(preds=celltype_pred, target=celltype.reshape(-1))
        val_metric_multiclass_acc = self.metric_multiclass_acc(preds=celltype_pred, target=celltype.reshape(-1))

        log_dict = {"val_x_rmse": val_metric_rmse[0], "val_y_rmse": val_metric_rmse[1]}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_rmse_overall", val_metric_rmse_overall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_overall_acc", val_overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_macro_acc", val_macro_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        gene_exp, edgelist, xy, section_idx, celltype = self.proc_batch(batch)
        xy_mu_pred, xy_L_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)
        return xy_mu_pred.to("cpu").numpy(), xy_L_pred.to("cpu").numpy(), celltype_pred.to("cpu").numpy()

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx):
        gene_exp, edgelist, xy, section_idx, celltype = self.proc_batch(batch)
        xy_mu_pred, xy_L_pred, celltype_pred = self.forward(gene_exp, section_idx, edgelist)

        # return only the last entry. This corresponds to idx passed to __getitem__.
        xy_mu_pred = xy_mu_pred.to("cpu").numpy()[-1, :]
        xy_L_pred = xy_L_pred.to("cpu").numpy()[-1, :]
        celltype_pred = celltype_pred.to("cpu").numpy()[-1, :]
        return xy_mu_pred, xy_L_pred, celltype_pred

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
