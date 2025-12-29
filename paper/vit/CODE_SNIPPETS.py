"""
VIT Paper - Key Code Snippets
=============================

This file contains the essential code snippets for the paper's Methods section.
All code is extracted from the VIT repository at ~/VIT.

"""

# ==============================================================================
# 1. MODEL ARCHITECTURE (src/models/specvit.py)
# ==============================================================================

class MyViT(ViTPreTrainedModel, BaseModel):
    """Vision Transformer adapted for 1D spectral regression.
    
    Key modifications from standard ViT:
    - Custom SpectraEmbeddings for 1D input
    - Conv1D patch tokenizer instead of 2D patches
    - Regression head instead of classification
    """
    
    config_class = ViTConfig

    def __init__(
        self,
        config: ViTConfig,
        loss_name: str = "",
        model_name: str = "ViT",
        preprocessor: nn.Module | None = None,
        full_config: dict = None,
    ) -> None:
        ViTPreTrainedModel.__init__(self, config)
        self.config = config
        self.vit = ViTModel(config)
        
        # Replace embeddings with custom SpectraEmbeddings
        self.vit.embeddings = SpectraEmbeddings(config)
        
        self.preprocessor = preprocessor
        
        # Setup regression head
        self.task_type = config.task_type  # 'reg'
        self.regressor = nn.Linear(config.hidden_size, config.num_labels)
        
        # Loss function: L1 for MAE
        self.loss_fct = nn.L1Loss()
        
        self.init_weights()

    def forward(self, pixel_values, labels=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None):
        return_dict = return_dict or self.config.use_return_dict

        # Apply preprocessor if exists (e.g., ZCA whitening)
        if self.preprocessor is not None:
            pixel_values = self.preprocessor(pixel_values)

        # ViT forward pass
        outputs = self.vit(pixel_values, output_attentions=output_attentions, 
                          output_hidden_states=output_hidden_states, 
                          return_dict=return_dict)
        
        # Extract CLS token
        cls_token = outputs[0][:, 0, :]

        # Regression head
        logits = self.regressor(cls_token)

        # Compute loss
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1).float())

        return SequenceClassifierOutput(
            loss=loss, 
            logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions
        )


# ==============================================================================
# 2. PATCH TOKENIZATION (src/models/tokenization.py)
# ==============================================================================

class Conv1DPatchTokenizer(nn.Module):
    """Tokenize 1D spectra using Conv1d kernel.
    
    Converts a spectrum of length L into a sequence of N patch embeddings,
    where N = (L - patch_size) // stride + 1.
    """

    def __init__(self, input_length: int, patch_size: int, 
                 hidden_size: int, stride: int | None = None) -> None:
        super().__init__()
        stride_size = stride if stride and stride > 0 else int(patch_size)
        
        self.image_size = input_length      # 4096
        self.patch_size = patch_size        # 16
        self.stride_size = stride_size      # 16
        self.num_channels = 1
        
        # Number of patches: (4096 - 16) // 16 + 1 = 256
        self.num_patches = ((self.image_size - self.patch_size) // self.stride_size) + 1
        
        # Conv1d projection: (1, L) -> (hidden_size, N)
        self.projection = nn.Conv1d(
            in_channels=self.num_channels,
            out_channels=hidden_size,
            kernel_size=self.patch_size,
            stride=self.stride_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 4096)
        x = x.reshape(-1, 1, self.image_size)  # (batch, 1, 4096)
        x = self.projection(x)                  # (batch, 256, 256)
        return x.transpose(1, 2)                # (batch, 256, 256)


# ==============================================================================
# 3. EMBEDDINGS (src/models/embedding.py)
# ==============================================================================

class SpectraEmbeddings(nn.Module):
    """Patch + positional embeddings for 1D spectral inputs.
    
    Supports different position encoding types:
    - 'learned': Learned absolute position embeddings (used in best model)
    - 'rope': Rotary Position Embedding
    - None: No position encoding
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        stride_size = getattr(config, "stride_size", None)
        stride = stride_size if stride_size and stride_size > 0 else int(config.stride_ratio * config.patch_size)
        
        # Conv1D patch tokenizer
        self.patch_embeddings = Conv1DPatchTokenizer(
            input_length=config.image_size,    # 4096
            patch_size=config.patch_size,      # 16
            hidden_size=config.hidden_size,    # 256
            stride=stride,                     # 16
        )

        self.num_patches = self.patch_embeddings.num_patches  # 256
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Position encoding: learned absolute embeddings
        self.pos_encoding_type = getattr(config, "pos_encoding_type", None)
        
        if self.pos_encoding_type == "learned":
            # Learned position embeddings: (1, num_patches + 1, hidden_size)
            self.position_embeddings = nn.Parameter(
                torch.randn(1, self.num_patches + 1, config.hidden_size)
            )

    def forward(self, x: torch.Tensor, bool_masked_pos=None, 
                interpolate_pos_encoding=False) -> torch.Tensor:
        # Patch embedding: (batch, 256, 256)
        tokens = self.patch_embeddings(x)
        batch_size = tokens.size(0)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, 256)
        tokens = torch.cat((cls_tokens, tokens), dim=1)         # (batch, 257, 256)
        
        # Add position embeddings
        if self.pos_encoding_type == "learned":
            tokens = tokens + self.position_embeddings
        
        return self.dropout(tokens)


# ==============================================================================
# 4. TRAINING STEP (src/base/vit.py)
# ==============================================================================

class ViTLModule(BaseLightningModule):
    """PyTorch Lightning module for ViT training."""
    
    def __init__(self, model=None, config={}):
        model = model or get_model(config)
        super().__init__(model=model, config=config)
        self.save_hyperparameters(ignore=['model'])
        
        self.task_type = 'reg'
        self.noise_level = config.get('noise', {}).get('noise_level', 0.0)
        
        # Metrics
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.r2 = R2Score()
        self.monitor_metric = 'mae'

    def training_step(self, batch, batch_idx):
        # Training: batch is (flux, error, labels)
        flux, error, labels = batch
        
        # On-the-fly noise augmentation
        if self.noise_level > 0:
            noisy = flux + torch.randn_like(flux) * error * self.noise_level
            loss = self.model(noisy, labels=labels).loss
        else:
            loss = self.model(flux, labels=labels).loss
        
        self.log('mae_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Validation: batch is (noisy, flux, error, labels) with pre-generated noise
        noisy, flux, error, labels = batch
        
        if self.noise_level > 0:
            outputs = self.model(noisy, labels=labels)
        else:
            outputs = self.model(flux, labels=labels)
        
        loss = outputs.loss
        preds = outputs.logits.squeeze()
        
        # Log metrics
        self.log('val_mae_loss', loss, on_step=False, on_epoch=True)
        self.log('val_mae', self.mae(preds, labels), on_step=False, on_epoch=True)
        self.log('val_mse', self.mse(preds, labels), on_step=False, on_epoch=True)
        self.log('val_r2', self.r2(preds, labels), on_step=False, on_epoch=True)
        
        return loss


# ==============================================================================
# 5. DATA LOADING (src/dataloader/base.py, src/dataloader/spec_datasets.py)
# ==============================================================================

class BaseSpecDataset(MaskMixin, NoiseMixin, BaseDataset):
    """Base dataset for spectral data with noise augmentation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.noisy = None  # Pre-generated noisy data for val/test

    def load_data(self, stage: Optional[str] = None) -> None:
        load_path, num_samples = self.get_path_and_samples(stage)
        
        with h5py.File(load_path, "r") as f:
            self.wave = torch.tensor(f["spectrumdataset/wave"][()], dtype=torch.float32)
            self.flux = torch.tensor(
                f["dataset/arrays/flux/value"][:num_samples], dtype=torch.float32
            )
            self.error = torch.tensor(
                f["dataset/arrays/error/value"][:num_samples], dtype=torch.float32
            )

        self.flux = self.flux.clip(min=0.0)
        self.num_samples = self.flux.shape[0]
        self.num_pixels = len(self.wave)

    def _set_noise(self, seed: int = 42) -> None:
        """Pre-generate noisy data with fixed seed for reproducible validation/test."""
        if self.noise_level > 0:
            torch.manual_seed(seed)
            noise = torch.randn_like(self.flux) * self.error * self.noise_level
            self.noisy = self.flux + noise


class RegSpecDataset(BaseSpecDataset):
    """Regression dataset for stellar parameter prediction."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_mean = None
        self.label_std = None

    def load_data(self, stage=None):
        super().load_data(stage)
        self.load_params(stage)
        
        # Load labels
        self.labels = torch.tensor(self.param_values).float()
        
        # Normalize labels (z-score for training)
        self._maybe_normalize_labels(stage)
        
        # Pre-generate noisy data for validation/test
        if stage in ('val', 'test', 'validate'):
            self._set_noise()

    def _maybe_normalize_labels(self, stage=None):
        """Apply z-score normalization to labels."""
        kind = getattr(self, "label_norm", "none")
        if kind not in ("standard", "zscore"):
            return
        
        is_train = stage in (None, "fit", "train")
        if is_train or (self.label_mean is None or self.label_std is None):
            self.label_mean = self.labels.mean()
            self.label_std = self.labels.std()
        
        self.labels = (self.labels - self.label_mean) / self.label_std

    def __getitem__(self, idx):
        flux, error = super().__getitem__(idx)
        
        # For val/test: return pre-generated noisy; for train: return flux
        if self.noisy is not None:
            return self.noisy[idx], flux, error, self.labels[idx]
        else:
            return flux, error, self.labels[idx]


# ==============================================================================
# 6. OPTIMIZER CONFIGURATION (src/opt/optimizer.py)
# ==============================================================================

class OptModule():
    """Optimizer and learning rate scheduler configuration."""
    
    def __init__(self, lr, monitor_metric='loss', opt_type='adamw', 
                 weight_decay=0.01, lr_scheduler_name='cosine', **kwargs):
        self.lr = float(lr)
        self.opt_type = opt_type
        self.weight_decay = weight_decay
        self.lr_scheduler_name = lr_scheduler_name
        self.kwargs = kwargs

    def __call__(self, model):
        # Create AdamW optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Create Cosine Annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.kwargs.get('T_max', 200),
            eta_min=self.kwargs.get('eta_min', 1e-5)
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": f"val_{self.monitor_metric}",
                "interval": "epoch",
                "frequency": 1
            }
        }


# ==============================================================================
# 7. INFERENCE EXAMPLE
# ==============================================================================

def load_model_for_inference(checkpoint_path, config_path):
    """Load trained model for inference."""
    from src.models import get_model
    from src.utils.utils import load_config
    
    # Load config
    config = load_config(config_path)
    
    # Build model
    model = get_model(config)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict (remove 'model.' prefix from Lightning)
    state_dict = {k.replace('model.', ''): v 
                  for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # Get normalization stats for de-normalizing predictions
    label_mean = ckpt['hyper_parameters']['config']['data'].get('label_mean')
    label_std = ckpt['hyper_parameters']['config']['data'].get('label_std')
    
    return model, label_mean, label_std


def predict(model, spectrum, label_mean=None, label_std=None):
    """Make prediction on a single spectrum."""
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension if needed
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
        
        # Forward pass
        output = model(spectrum, labels=None)
        pred_normalized = output.logits.squeeze()
        
        # De-normalize if stats available
        if label_mean is not None and label_std is not None:
            pred = pred_normalized * label_std + label_mean
        else:
            pred = pred_normalized
    
    return pred


# ==============================================================================
# 8. ViT CONFIG FACTORY (src/models/builder.py)
# ==============================================================================

def get_vit_config(config):
    """Build ViTConfig from config dict."""
    m = config["model"]
    
    return ViTConfig(
        task_type='reg',
        image_size=m["image_size"],           # 4096
        patch_size=m["patch_size"],           # 16
        num_channels=1,                        # 1D input
        hidden_size=m["hidden_size"],         # 256
        num_hidden_layers=m["num_hidden_layers"],  # 6
        num_attention_heads=m["num_attention_heads"],  # 8
        intermediate_size=4 * m["hidden_size"],  # 1024
        stride_size=m.get("stride_size", None),  # 16
        proj_fn=m["proj_fn"],                 # 'C1D'
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        num_labels=1,                          # Single output
        pos_encoding_type='learned',           # Learned positional embeddings
        max_position_embeddings=512,
    )
