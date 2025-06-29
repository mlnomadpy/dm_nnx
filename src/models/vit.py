import jax
import jax.numpy as jnp
from flax.nnx import (
    Dropout,
    Embed,
    Linear,
    Module,
    MultiHeadAttention,
    LayerNorm,
    vmap,
)
from typing import Tuple


class PatchEmbedding(Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        *,
        rngs,
    ):
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.projection = Linear(
            in_features=patch_size * patch_size * 3,
            out_features=embed_dim,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array):
        x = x.transpose((0, 2, 3, 1))
        x = jnp.reshape(
            x,
            (
                x.shape[0],
                x.shape[1] // self.patch_size,
                self.patch_size,
                x.shape[2] // self.patch_size,
                self.patch_size,
                3,
            ),
        )
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = jnp.reshape(
            x, (x.shape[0], self.num_patches, self.patch_size * self.patch_size * 3)
        )
        x = self.projection(x)
        return x


class EncoderBlock(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout_rate: float,
        *,
        rngs,
    ):
        self.norm1 = LayerNorm(num_features=embed_dim, rngs=rngs)
        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            query_size=embed_dim,
            use_bias=True,
            rngs=rngs,
        )
        self.dropout = Dropout(rate=dropout_rate)
        self.norm2 = LayerNorm(num_features=embed_dim, rngs=rngs)
        self.linear1 = Linear(in_features=embed_dim, out_features=mlp_dim, rngs=rngs)
        self.linear2 = Linear(in_features=mlp_dim, out_features=embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x)
        x = self.dropout(x, deterministic=not train)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = jax.nn.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x, deterministic=not train)
        x = x + residual
        return x


class ViT(Module):
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
        *,
        rngs,
    ):
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            rngs=rngs,
        )
        self.cls_token = self.param(
            "cls_token", lambda rng: jnp.zeros((1, 1, embed_dim))
        )
        self.pos_embedding = self.param(
            "pos_embedding",
            lambda rng: jnp.zeros(
                (1, self.patch_embedding.num_patches + 1, embed_dim)
            ),
        )
        self.dropout = Dropout(rate=dropout_rate)

        self.encoder_blocks = [
            EncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]
        self.norm = LayerNorm(num_features=embed_dim, rngs=rngs)
        self.head = Linear(in_features=embed_dim, out_features=num_classes, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool):
        x = self.patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = jnp.tile(self.cls_token, (b, 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + self.pos_embedding[:, : n + 1]
        x = self.dropout(x, deterministic=not train)

        for block in self.encoder_blocks:
            x = block(x, train=train)

        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits
