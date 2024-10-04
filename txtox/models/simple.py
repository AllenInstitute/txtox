import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy


class LitMLPv0(L.LightningModule):
    def __init__(self, input_size=500, n_labels=10, weight_mse=1.0, weight_ce=1.0):
        super(LitMLPv0, self).__init__()

        self.weight_mse = weight_mse
        self.weight_ce = weight_ce

        # fmt:off
        self.encoder = nn.Sequential(
            nn.Dropout(0.3), 
            nn.Linear(input_size, 100), 
            nn.GELU(), 
            nn.Linear(100, 20), 
            nn.GELU()
        )
        # fmt:on
        self.spatial_out = nn.Linear(20, 3)
        self.label_out = nn.Linear(20, n_labels)

        # losses
        self.loss_mse = nn.MSELoss()
        self.loss_ce = nn.CrossEntropyLoss()

        # metrics
        self.metric_rmse = MeanSquaredError(squared=False, num_outputs=3)
        self.metric_multiclass_acc = MulticlassAccuracy(
            num_classes=n_labels,
            top_k=1,
            average=None,
            multidim_average="global",
            ignore_index=None,
            validate_args=True,
        )

    def forward(self, x):
        shared_rep = self.encoder(x)
        xyz = self.spatial_out(shared_rep)
        celltype = self.label_out(shared_rep)
        return xyz, celltype

    def training_step(self, batch, batch_idx):
        gene_exp, xyz, celltype = batch
        xyz_pred, celltype_pred = self.forward(gene_exp)

        # Calculate losses
        mse_loss = self.loss_mse(xyz_pred, xyz)
        ce_loss = self.loss_ce(celltype_pred, celltype.squeeze())
        total_loss = self.weight_mse * mse_loss + self.weight_ce * ce_loss

        # Calculate metrics
        train_metric_multiclass_acc = self.metric_multiclass_acc(preds=celltype_pred, target=celltype.reshape(-1))
        train_metric_rmse = self.metric_rmse(xyz_pred, xyz)

        # Log losses and metrics
        self.log("train_mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        log_dict = {"train_x_rmse": train_metric_rmse[0], "train_y_rmse": train_metric_rmse[1], "train_z_rmse": train_metric_rmse[2]}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_metric_multiclass_acc", train_metric_multiclass_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_metric_rmse", train_metric_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss  # this will be minimized

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        gene_exp, xyz, celltype = batch
        xyz_pred, celltype_pred = self.forward(gene_exp)

        # Calculate metrics
        val_metric_multiclass_acc = self.metric_multiclass_acc(preds=celltype_pred, target=celltype.reshape(-1))
        val_metric_rmse = self.metric_rmse(xyz_pred, xyz)
        log_dict = {"val_x_rmse": val_metric_rmse[0], "val_y_rmse": val_metric_rmse[1], "val_z_rmse": val_metric_rmse[2]}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Log metrics
        # self.log("val_metric_multiclass_acc", val_metric_multiclass_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # val_metric_rmse_dict = {f"val_metric_rmse_{i}": val_metric_rmse[i] for name, i in enumerate(('x','y','z'),range(3))}
        # self.log("val_metric_rmse", val_metric_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # self.logger.experiment.add_figure()

    def on_validation_epoch_end(self):
        pass

    def test_step(self):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
