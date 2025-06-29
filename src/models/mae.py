import jax
import jax.numpy as jnp
from flax.nnx import (
    Dropout,
    Linear,
    Module,
    LayerNorm,
)
from .vit import ViT


class MAE(Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float,
        mask_ratio: float,
        *,
        rngs,
    ):
        self.mask_ratio = mask_ratio
        self.encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        self.decoder = Linear(
            in_features=embed_dim,
            out_features=patch_size * patch_size * 3,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, *, train: bool):
        x = self.encoder.patch_embedding(x)
        b, n, _ = x.shape

        if train:
            # Masking
            num_masked = int(self.mask_ratio * n)
            masked_indices = jax.random.permutation(self.make_rng('params'), n)[:num_masked]
            unmasked_indices = jnp.setdiff1d(jnp.arange(n), masked_indices)

            # Encoder
            x_unmasked = x[:, unmasked_indices, :]
            cls_tokens = jnp.tile(self.encoder.cls_token, (b, 1, 1))
            x_unmasked = jnp.concatenate([cls_tokens, x_unmasked], axis=1)
            x_unmasked = x_unmasked + self.encoder.pos_embedding[:, :n - num_masked + 1]
            x_unmasked = self.encoder.dropout(x_unmasked, deterministic=not train)

            for block in self.encoder.encoder_blocks:
                x_unmasked = block(x_unmasked, train=train)

            x_unmasked = self.encoder.norm(x_unmasked)

            # Decoder
            masked_tokens = jnp.zeros((b, num_masked, x.shape[-1]))
            x_masked = jnp.concatenate([x_unmasked[:, 1:, :], masked_tokens], axis=1)
            x_reconstructed = self.decoder(x_masked)

            return x_reconstructed, masked_indices
        else:
            return self.encoder(x, train=False)
