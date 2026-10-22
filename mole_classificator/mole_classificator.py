import logging
import math

import lightning as L
import torch
import torch.nn.functional as F
import torchvision
from torchmetrics.classification import MulticlassAUROC, MulticlassF1Score

log = logging.getLogger(__name__)


class Classificator(L.LightningModule):
    def __init__(self, **params):
        super().__init__()
        # Older Lightning versions may have saved the variadic arguments nested.
        params = {**params.pop("params", {}), **params}
        self.lr = params["lr"]
        self.weight_decay = params["weight_decay"]
        self.num_classes = params["num_classes"]
        self.d_model = params.get("d_model", 256)
        self.freeze_backbone_epochs = params.get("freeze_backbone_epochs", 3)
        self.backbone_lr_mult = params.get("backbone_lr_mult", 0.1)
        self.diagnostics_every_n_steps = params.get("diagnostics_every_n_steps", 100)
        self._last_inspected_step = -1
        self._last_gradient_step = -1
        # Old checkpoints used ToTensor only. New training config enables normalization.
        params.setdefault("normalize", False)
        params.setdefault("max_image_size", 600)
        if params.get("class_weights") is not None:
            params["class_weights"] = torch.as_tensor(params["class_weights"], dtype=torch.float32).detach().clone()
        self.save_hyperparameters(params)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=params.get("class_weights"))
        # TorchMetrics states are non-persistent: older checkpoints still load strictly.
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average=None)
        self.val_auroc = MulticlassAUROC(num_classes=self.num_classes, average=None)
        self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average=None)
        self.test_auroc = MulticlassAUROC(num_classes=self.num_classes, average=None)

        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if params.get("pretrained", True) else None
        self.backbone = torchvision.models.convnext_tiny(weights=weights).features
        # DDP must register the backbone before it unfreezes. Frozen epochs still
        # compute its backward pass; on_before_optimizer_step discards its gradients.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
        self.token_proj = torch.nn.Conv2d(768, self.d_model, kernel_size=1)

        # Keep the old parameter shape/key so native-resolution checkpoints still load.
        num_tokens = params.get("num_tokens", 252)
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        height = math.isqrt(num_tokens)
        while num_tokens % height:
            height -= 1
        self.pos_grid_size = tuple(params.get("pos_grid_size", (height, num_tokens // height)))
        if (len(self.pos_grid_size) != 2 or min(self.pos_grid_size) < 1
                or math.prod(self.pos_grid_size) != num_tokens):
            raise ValueError("pos_grid_size must contain num_tokens positions")
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_tokens, self.d_model) * 0.02)

        layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=params.get("nhead", 8),
            dim_feedforward=self.d_model * 2, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=params.get("num_layers", 2))
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.d_model), torch.nn.Dropout(0.2),
            torch.nn.Linear(self.d_model, self.num_classes),
        )
        self.example_input_array = torch.zeros(1, 3, 64, 64)

    def positional_embedding(self, height, width):
        positions = self.pos_embedding.reshape(1, *self.pos_grid_size, self.d_model).permute(0, 3, 1, 2)
        if (height, width) != self.pos_grid_size:
            positions = F.interpolate(positions.float(), size=(height, width), mode="bicubic", align_corners=False)
        return positions.flatten(2).transpose(1, 2)

    def forward(self, x, inspect=False):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 32:
            raise ValueError("Expected RGB input [batch, 3, height, width] with both sides >= 32")
        feat = x
        for index, stage in enumerate(self.backbone):
            feat = stage(feat)
            if inspect:
                self._inspect_tensor(f"backbone/stage_{index}", feat)
        feat = self.token_proj(feat)
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = tokens + self.positional_embedding(*feat.shape[-2:]).to(tokens.dtype)
        if inspect:
            self._inspect_tensor("projection", feat)
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        logits = self.classifier(pooled)
        if inspect:
            self._inspect_tensor("transformer", tokens)
            self._inspect_tensor("pooled", pooled)
            self._inspect_tensor("logits", logits)
        return logits

    def _writer(self):
        if self._trainer is None or not self.trainer.is_global_zero:
            return None
        for logger in self.loggers:
            experiment = logger.experiment
            if hasattr(experiment, "add_histogram"):
                return experiment
        return None

    def _inspect_tensor(self, name, tensor):
        writer = self._writer()
        if writer is None:
            return
        # Bound diagnostic transfer/serialization cost even for large feature maps.
        values = tensor.detach().float().flatten()
        values = values[::max(1, values.numel() // 2048)][:2048]
        writer.add_histogram(f"activations/{name}", values.cpu(), self.global_step)
        writer.add_scalar(f"activation_mean/{name}", values.mean(), self.global_step)
        writer.add_scalar(f"activation_std/{name}", values.std(unbiased=False), self.global_step)

    def on_train_epoch_start(self):
        frozen = self.current_epoch < self.freeze_backbone_epochs
        self.backbone.train(not frozen)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
        if self.current_epoch == self.freeze_backbone_epochs:
            log.info("Epoch %d: backbone unfrozen", self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        inspect = (self.diagnostics_every_n_steps > 0
                   and self.global_step % self.diagnostics_every_n_steps == 0
                   and self.global_step != self._last_inspected_step)
        logits = self(x, inspect=inspect)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_accuracy", (logits.argmax(-1) == y).float().mean(),
                 on_step=True, on_epoch=True, sync_dist=True)
        if inspect:
            self._last_inspected_step = self.global_step
            writer = self._writer()
            if writer is not None:
                images = x[:4].detach().cpu()
                if self.hparams.normalize:
                    from mole_classificator.preprocessing import MEAN, STD
                    images = images * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
                writer.add_images("inputs", images.clamp(0, 1), self.global_step)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Lightning calls this after AMP unscaling and gradient accumulation.
        if self.current_epoch < self.freeze_backbone_epochs:
            for parameter in self.backbone.parameters():
                # None skips AdamW's update and weight decay during frozen epochs.
                parameter.grad = None
        if (self.diagnostics_every_n_steps <= 0
                or self.global_step % self.diagnostics_every_n_steps
                or self.global_step == self._last_gradient_step):
            return
        writer = self._writer()
        if writer is None:
            return
        self._last_gradient_step = self.global_step
        gradients = [p.grad.detach().float().norm() for p in self.parameters() if p.grad is not None]
        if gradients:
            writer.add_scalar("gradients/global_norm", torch.stack(gradients).norm(), self.global_step)
        for name, parameter in self.named_parameters():
            if name in {"pos_embedding", "token_proj.weight", "classifier.2.weight"}:
                writer.add_histogram(f"weights/{name}", parameter.detach().float().cpu(), self.global_step)
                if parameter.grad is not None:
                    writer.add_histogram(f"gradients/{name}", parameter.grad.detach().float().cpu(), self.global_step)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("val_loss", self.loss_fn(logits, y), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", (logits.argmax(-1) == y).float().mean(), on_epoch=True, sync_dist=True)
        self.val_f1.update(logits.argmax(-1), y)
        self.val_auroc.update(logits.float().softmax(dim=-1), y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("test_loss", self.loss_fn(logits, y), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_accuracy", (logits.argmax(-1) == y).float().mean(), on_epoch=True, sync_dist=True)
        self.test_f1.update(logits.argmax(-1), y)
        self.test_auroc.update(logits.float().softmax(dim=-1), y)

    def on_validation_epoch_start(self):
        self.val_f1.reset()
        self.val_auroc.reset()

    def on_test_epoch_start(self):
        self.test_f1.reset()
        self.test_auroc.reset()

    def _log_classification_metrics(self, stage):
        f1_metric = getattr(self, f"{stage}_f1")
        auroc_metric = getattr(self, f"{stage}_auroc")
        # All ranks call compute(): TorchMetrics synchronizes global states in DDP.
        f1 = f1_metric.compute()
        auroc = auroc_metric.compute()
        self.log(f"{stage}_f1_macro", f1.mean(), prog_bar=True, sync_dist=False)
        self.log(f"{stage}_auroc_macro", auroc.mean(), prog_bar=True, sync_dist=False)
        names = self.hparams.get("class_names") or [str(i) for i in range(self.num_classes)]
        for index, name in enumerate(names):
            self.log(f"{stage}_f1/{name}", f1[index], sync_dist=False)
            self.log(f"{stage}_auroc/{name}", auroc[index], sync_dist=False)
        f1_metric.reset()
        auroc_metric.reset()

    def on_validation_epoch_end(self):
        self._log_classification_metrics("val")

    def on_test_epoch_end(self):
        self._log_classification_metrics("test")

    def configure_optimizers(self):
        decay, no_decay, backbone_params = [], [], []
        for name, parameter in self.named_parameters():
            # Frozen parameters must already be in the optimizer when later unfrozen.
            if name.startswith("backbone."):
                backbone_params.append(parameter)
            elif parameter.ndim <= 1 or "bias" in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": backbone_params, "lr": self.lr * self.backbone_lr_mult,
             "weight_decay": self.weight_decay},
        ], lr=self.lr)
        epochs = self.trainer.max_epochs if self._trainer is not None else 50
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
