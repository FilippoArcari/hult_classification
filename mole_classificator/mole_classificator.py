import lightning as L
import torch
import torchvision


class Classificator(L.LightningModule):
    def __init__(self, **params):
        super().__init__()

        try:
            self.lr = params["lr"]
            self.weight_decay = params["weight_decay"]
            self.num_classes = params["num_classes"]
        except KeyError as e:
            raise KeyError(f"Missing hyperparameters: {e}")


        # d_model del Transformer: proiettiamo i 768 canali dell'ultimo
        # stage di ConvNeXt-Tiny a una dimensione più piccola per non
        # gonfiare inutilmente il Transformer con soli 7k esempi
        self.d_model = params.get("d_model", 256)
        self.freeze_backbone_epochs = params.get("freeze_backbone_epochs", 3)
        self.backbone_lr_mult = params.get("backbone_lr_mult", 0.1)

        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=params.get("class_weights", None)
        )

        # --------------------------------------------------
        # BACKBONE: ConvNeXt-Tiny pretrained ImageNet-1k
        # --------------------------------------------------
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        convnext = torchvision.models.convnext_tiny(weights=weights)

        # `.features` è la parte fully-convolutional (stem + 4 stage),
        # esclude avgpool e classifier: output [B, 768, H/32, W/32]
        self.backbone = convnext.features
        backbone_out_channels = 768  # canali dell'ultimo stage di ConvNeXt-Tiny

        # Congeliamo il backbone all'inizio, lo sblocchiamo dopo N epoche
        # (vedi on_train_epoch_start)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # --------------------------------------------------
        # PROIEZIONE CANALI -> TOKEN + POSITIONAL EMBEDDING
        # --------------------------------------------------
        self.token_proj = torch.nn.Conv2d(
            backbone_out_channels, self.d_model, kernel_size=1
        )

        # num_tokens dipende dalla risoluzione di input: con 384x512
        # e downsample totale 32x -> 12x16 = 192 token.
        # Va ricalcolato se cambi la risoluzione in preprocessing.
        num_tokens = params.get("num_tokens", 252)
        self.pos_embedding = torch.nn.Parameter(
            torch.randn(1, num_tokens, self.d_model) * 0.02
        )

        # --------------------------------------------------
        # TRANSFORMER
        # --------------------------------------------------
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --------------------------------------------------
        # CLASSIFIER
        # --------------------------------------------------
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.d_model),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.d_model, self.num_classes),
        )

        self.save_hyperparameters()

    def forward(self, x):
        feat = self.backbone(x)               # [B, 768, H', W']
        feat = self.token_proj(feat)           # [B, d_model, H', W']
        tokens = feat.flatten(2).transpose(1, 2)  # [B, N, d_model]

        n = tokens.shape[1]
        tokens = tokens + self.pos_embedding[:, :n, :]

        tokens = self.transformer(tokens)      # [B, N, d_model]
        pooled = tokens.mean(dim=1)            # global average pooling
        return self.classifier(pooled)

    def on_train_epoch_start(self):
        # Sblocco progressivo del backbone dopo `freeze_backbone_epochs`
        if self.current_epoch == self.freeze_backbone_epochs:
            for p in self.backbone.parameters():
                p.requires_grad = True
            print(f"[epoch {self.current_epoch}] backbone sbloccato")

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Learning rate ridotto per il backbone pretrained (fine-tuning
        # più delicato), LR pieno per Transformer + classifier + proiezione,
        # ed escludiamo bias/LayerNorm dal weight decay
        decay, no_decay, backbone_params = [], [], []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(p)
            elif p.ndim <= 1 or "bias" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        param_groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": backbone_params, "lr": self.lr * self.backbone_lr_mult,
             "weight_decay": self.weight_decay},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 50
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}