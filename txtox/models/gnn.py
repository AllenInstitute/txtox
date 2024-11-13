import lightning as L
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torchmetrics import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy


class LitGNNv0(L.LightningModule):
    def __init__(self, input_dim=500, hidden_dim=100, n_labels=126, weight_mse=1.0, weight_ce=1.0):
        super(LitGNNv0, self).__init__()

        self.weight_mse = weight_mse
        self.weight_ce = weight_ce

        # fmt:off
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.encoder = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(hidden_dim, 20), torch.nn.GELU())
        self.spatial_out = torch.nn.Sequential(torch.nn.Linear(20, 20), torch.nn.GELU(), torch.nn.Linear(20, 3))
        self.label_out = torch.nn.Sequential(torch.nn.Linear(20, 20), torch.nn.GELU(), torch.nn.Linear(20, n_labels))
        self.gelu = torch.nn.GELU()

        # losses
        self.loss_mse = nn.MSELoss()
        self.loss_ce = nn.CrossEntropyLoss()

        # metrics
        self.metric_rmse = MeanSquaredError(squared=False, num_outputs=3)
        self.metric_rmse_overall = MeanSquaredError(squared=False, num_outputs=1)
        self.metric_overall_acc = MulticlassAccuracy(
            num_classes=n_labels, top_k=1, average="weighted", multidim_average="global"
        )
        self.metric_macro_acc = MulticlassAccuracy(
            num_classes=n_labels, top_k=1, average="macro", multidim_average="global"
        )
        self.metric_multiclass_acc = MulticlassAccuracy(
            num_classes=n_labels, top_k=1, average=None, multidim_average="global"
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.gelu(x)
        x = self.conv2(x, edge_index)
        x = self.gelu(x)
        x = self.encoder(x)
        xyz = self.spatial_out(x)
        celltype = self.label_out(x)
        return xyz, celltype

    def training_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        gene_exp, edgelist, xyz, celltype = batch
        gene_exp = gene_exp.squeeze(dim=0)
        edgelist = edgelist.squeeze(dim=0).T
        xyz = xyz.squeeze(dim=0)
        celltype = celltype.squeeze(dim=0)

        xyz_pred, celltype_pred = self.forward(gene_exp, edgelist)

        # Calculate losses
        mse_loss = self.loss_mse(xyz_pred, xyz)
        ce_loss = self.loss_ce(celltype_pred, celltype.squeeze())
        total_loss = self.weight_mse * mse_loss + self.weight_ce * ce_loss

        # Log losses
        self.log("train_mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Calculate metrics
        train_metric_rmse_overall = self.metric_rmse_overall(xyz_pred, xyz)
        train_overall_acc = self.metric_overall_acc(preds=celltype_pred, target=celltype.reshape(-1))
        train_macro_acc = self.metric_macro_acc(preds=celltype_pred, target=celltype.reshape(-1))
        train_metric_rmse = self.metric_rmse(xyz_pred, xyz)

        # Log metrics
        self.log(
            "train_rmse_overall", train_metric_rmse_overall, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train_overall_acc", train_overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_macro_acc", train_macro_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        log_dict = {
            "train_x_rmse": train_metric_rmse[0],
            "train_y_rmse": train_metric_rmse[1],
            "train_z_rmse": train_metric_rmse[2],
        }
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return total_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        gene_exp, edgelist, xyz, celltype = batch
        gene_exp = gene_exp.squeeze(dim=0)
        edgelist = edgelist.squeeze(dim=0).T
        xyz = xyz.squeeze(dim=0)
        celltype = celltype.squeeze(dim=0)

        xyz_pred, celltype_pred = self.forward(gene_exp, edgelist)

        # Calculate metrics
        val_metric_rmse = self.metric_rmse(xyz_pred, xyz)
        val_metric_rmse_overall = self.metric_rmse_overall(xyz_pred, xyz)
        val_overall_acc = self.metric_overall_acc(preds=celltype_pred, target=celltype.reshape(-1))
        val_macro_acc = self.metric_macro_acc(preds=celltype_pred, target=celltype.reshape(-1))
        val_metric_multiclass_acc = self.metric_multiclass_acc(preds=celltype_pred, target=celltype.reshape(-1))

        log_dict = {
            "val_x_rmse": val_metric_rmse[0],
            "val_y_rmse": val_metric_rmse[1],
            "val_z_rmse": val_metric_rmse[2],
        }
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_rmse_overall", val_metric_rmse_overall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_overall_acc", val_overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_macro_acc", val_macro_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        gene_exp, edgelist, xyz, celltype = batch
        gene_exp = gene_exp.squeeze(dim=0)
        edgelist = edgelist.squeeze(dim=0).T
        xyz = xyz.squeeze(dim=0)
        celltype = celltype.squeeze(dim=0)

        xyz_pred, celltype_pred = self.forward(gene_exp)
        return xyz_pred.to("cpu").numpy(), celltype_pred.to("cpu").numpy()

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        gene_exp, edgelist, xyz, celltype = batch
        gene_exp = gene_exp.squeeze(dim=0)
        edgelist = edgelist.squeeze(dim=0).T
        xyz = xyz.squeeze(dim=0)
        celltype = celltype.squeeze(dim=0)

        xyz_pred, celltype_pred = self.forward(gene_exp, edgelist)
        # return only the last entry. This corresponds to idx passed to __getitem__.
        xyz_pred = xyz_pred.to("cpu").numpy()[-1, :]
        celltype_pred = celltype_pred.to("cpu").numpy()[-1, :]
        return xyz_pred, celltype_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
