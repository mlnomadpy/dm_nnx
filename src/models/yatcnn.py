import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from .models import register_model


@register_model("yatcnn")
class YatCNN(nnx.Module):
    """YAT CNN model with custom layers."""
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(input_channels, 8, kernel_size=(7, 7), rngs=rngs)
        self.conv2 = nnx.Conv(8, 8, kernel_size=(5, 5), rngs=rngs)
        self.conv3 = nnx.Conv(8, 8, kernel_size=(5, 5), rngs=rngs)
        self.conv4 = nnx.Conv(8, 8, kernel_size=(5, 5), rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout4 = nnx.Dropout(rate=0.3, rngs=rngs)
        
        self.projector = nnx.Sequential(
            nnx.Linear(8, 128, rngs=rngs),
        )

        self.out_linear = nnx.Linear(8, num_classes, use_bias=False, rngs=rngs)

        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

    def __call__(self, x, training: bool = False, return_activations_for_layer: tp.Optional[str] = None):
        activations = {}
        x = self.conv1(x)
        x = jax.nn.relu(x)
        activations['conv1'] = x
        if return_activations_for_layer == 'conv1': return x
        x = self.dropout1(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv2(x)
        x = jax.nn.relu(x)
        activations['conv2'] = x
        if return_activations_for_layer == 'conv2': return x
        x = self.dropout2(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv3(x)
        x = jax.nn.relu(x)
        activations['conv3'] = x
        if return_activations_for_layer == 'conv3': return x
        x = self.dropout3(x, deterministic=not training)
        x = self.avg_pool(x)
        
        x = self.conv4(x)
        x = jax.nn.relu(x)
        activations['conv4'] = x
        if return_activations_for_layer == 'conv4': return x
        x = self.dropout4(x, deterministic=not training)
        
        # The representation used for contrastive learning is the output
        # of the conv backbone before global pooling for the classification head.
        representation = jnp.mean(x, axis=(1, 2))
        activations['representation'] = representation
        if return_activations_for_layer == 'representation': return representation

        projection = self.projector(representation)
        activations['projection'] = projection
        if return_activations_for_layer == 'projection': return projection

        # x = self.avg_pool(x)
        x = jnp.mean(x, axis=(1, 2))
        x = self.out_linear(x)
        activations['global_avg_pool'] = x
        if return_activations_for_layer == 'global_avg_pool': return x
        
        if return_activations_for_layer is not None and return_activations_for_layer not in activations:
            print(f"Warning: Layer '{return_activations_for_layer}' not found in YatCNN. Available: {list(activations.keys())}")
        return x
