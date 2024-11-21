# This is a implementation of the Fusemap graph autoencoder
# based on He et al. 2024: https://www.biorxiv.org/content/10.1101/2024.05.27.594872v1


import lightning as L
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch_geometric.nn.conv import GCNConv
from torchmetrics.classification import MulticlassAccuracy


# GNN encoder-decoder
class GNN_EncDec(nn.Module):  # noqa: N801
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN_EncDec, self).__init__()
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.conv2 = GCNConv(in_channels=output_dim, out_channels=hidden_dim)
        self.gelu = nn.GELU()

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.gelu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.gelu(x)
        return x


# Linear decoder for classification - returns logits
class LinearDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class LitFusemapGNNAE(L.LightningModule):
    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        latent_dim=64,
        n_region_labels=126,
        n_celltype_labels=16,
        weight_mse=1.0,
        weight_ce_regions=1.0,
        weight_ce_celltypes=1.0,
    ):
        super(LitFusemapGNNAE, self).__init__()

        self.weight_ce_regions = weight_ce_regions
        self.weight_ce_celltypes = weight_ce_celltypes
        self.logging_defaults = dict(on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # fmt:off
        self.gnnenc = GNN_EncDec(input_dim, hidden_dim, latent_dim)
        self.gnndec = GNN_EncDec(latent_dim, hidden_dim, input_dim)
        self.linear_dec_regions = LinearDecoder(latent_dim, n_region_labels)
        self.linear_dec_celltypes = LinearDecoder(latent_dim, n_celltype_labels)

        # losses for training
        self.loss_mse = MSELoss()
        self.loss_ce_regions = nn.CrossEntropyLoss()
        self.loss_ce_celltypes = nn.CrossEntropyLoss()

        # metrics for celltypes
        self.metric_overall_acc_celltypes = MulticlassAccuracy(num_classes=n_celltype_labels, top_k=1, average="weighted", multidim_average="global")
        self.metric_macro_acc_celltypes = MulticlassAccuracy(num_classes=n_celltype_labels, top_k=1, average="macro", multidim_average="global")
        self.metric_multiclass_acc_celltypes = MulticlassAccuracy(num_classes=n_celltype_labels, top_k=1, average=None, multidim_average="global")

        # metrics for regions
        self.metric_overall_acc_regions = MulticlassAccuracy(num_classes=n_region_labels, top_k=1, average="weighted", multidim_average="global")
        self.metric_macro_acc_regions = MulticlassAccuracy(num_classes=n_region_labels, top_k=1, average="macro", multidim_average="global")
        self.metric_multiclass_acc_regions = MulticlassAccuracy(num_classes=n_region_labels, top_k=1, average=None, multidim_average="global")
        # fmt:on

    def forward(self, zc, edge_index, edge_weight, section_idx, brain_idx):
        # zc is cell embedding, zt is region embedding
        zt = self.gnnenc(zt, edge_index, edge_weight)
        zc_recon = self.gnndec(zc, edge_index, edge_weight)
        region_logits = self.linear_dec_regions(zt)
        celltype_logits = self.linear_dec_celltypes(zt)
        return zc_recon, region_logits, celltype_logits

    def training_step(self, batch, batch_idx):
        # for GNN, batch size should be 1, and there isn't a batch dimension.
        steptype = "train"
        zc, edge_index, edge_weight, section_idx, brain_idx, label_celltypes, label_regions = batch
        zc_recon, region_logits, celltype_logits = self.forward(zc, edge_index, edge_weight, section_idx, brain_idx)

        # Calculate losses
        mse_loss = 0.5 * self.loss_mse(zc_recon, zc)
        ce_loss_regions = self.loss_ce_regions(region_logits, label_regions)
        ce_loss_celltypes = self.loss_ce_celltypes(celltype_logits, label_celltypes)
        total_loss = (
            self.weight_mse * mse_loss
            + self.weight_ce_regions * ce_loss_regions
            + self.weight_ce_celltypes * ce_loss_celltypes
        )

        # Log losses
        
        self.log(f"{steptype}_mse_loss", mse_loss, **self.logging_defaults)
        self.log(f"{steptype}_ce_regions_loss", ce_loss_regions, **self.logging_defaults)
        self.log(f"{steptype}_ce_celltypes_loss", ce_loss_celltypes, **self.logging_defaults)
        self.log(f"{steptype}_total_loss", total_loss, **self.logging_defaults)

        # Metrics for regions
        for name, metric in [
            ("overall_acc", self.metric_overall_acc_regions),
            ("macro_acc", self.metric_macro_acc_regions),
            ("multiclass_acc", self.metric_multiclass_acc_regions),
        ]:
            metric_value = metric(preds=region_logits, target=label_regions)
            self.log(
                f"{steptype}_{name}_regions", metric_value, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )

        # Metrics for celltypes
        for name, metric in [
            ("overall_acc", self.metric_overall_acc_celltypes),
            ("macro_acc", self.metric_macro_acc_celltypes),
            ("multiclass_acc", self.metric_multiclass_acc_celltypes),
        ]:
            metric_value = metric(preds=celltype_logits, target=label_celltypes)
            self.log(
                f"{steptype}_{name}_celltypes", metric_value, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )

        return total_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        steptype = "validation"
        
        zc, edge_index, edge_weight, section_idx, brain_idx, label_celltypes, label_regions = batch
        zc_recon, region_logits, celltype_logits = self.forward(zc, edge_index, edge_weight, section_idx, brain_idx)

        # Calculate losses
        mse_loss = 0.5 * self.loss_mse(zc_recon, zc)

        # Log losses
        self.log(f"{steptype}_mse_loss", mse_loss, **defaults)

        # Metrics for regions
        
        for name, metric in [
            ("overall_acc", self.metric_overall_acc_regions),
            ("macro_acc", self.metric_macro_acc_regions),
            ("multiclass_acc", self.metric_multiclass_acc_regions),
        ]:
            metric_value = metric(preds=region_logits, target=label_regions)
            self.log(f"{steptype}_{name}_regions", metric_value, **self.logging_defaults)

        # Metrics for celltypes
        for name, metric in [
            ("overall_acc", self.metric_overall_acc_celltypes),
            ("macro_acc", self.metric_macro_acc_celltypes),
            ("multiclass_acc", self.metric_multiclass_acc_celltypes),
        ]:
            metric_value = metric(preds=celltype_logits, target=label_celltypes)
            self.log(f"{steptype}_{name}_celltypes", metric_value, **self.logging_defaults)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
