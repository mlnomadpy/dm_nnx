import jax
import jax.numpy as jnp
from flax import nnx

class BYOL(nnx.Module):
    def __init__(self, encoder, predictor, *, rngs: nnx.Rngs):
        self.online_encoder = encoder
        self.target_encoder = encoder.copy()
        self.predictor = predictor

    def __call__(self, x, training: bool, return_activations_for_layer: str):
        return self.online_encoder(x, training=training, return_activations_for_layer=return_activations_for_layer)

    def update_target_network(self, momentum: float):
        # Update target network weights using exponential moving average
        online_params, online_state = self.online_encoder.split()
        target_params, target_state = self.target_encoder.split()

        new_target_params = jax.tree_map(
            lambda target, online: target * momentum + online * (1 - momentum),
            target_params,
            online_params
        )

        self.target_encoder = nnx.merge(new_target_params, target_state)
