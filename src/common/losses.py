import jax
import jax.numpy as jnp
import optax
from flax import nnx

def supervised_loss_fn(model, batch):
    logits = model(batch['image'], training=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()
    return loss, logits

def contrastive_loss_fn(model: nnx.Module, batch, temperature: float):
    """
    NT-Xent loss for contrastive learning.
    is no longer bounded like cosine similarity.
    """
    img1, img2 = batch['image1'], batch['image2']
    images = jnp.concatenate([img1, img2], axis=0)
    batch_size = img1.shape[0]

    # Get representations directly from the encoder backbone
    representations = model(images, training=True, return_activations_for_layer='representation')

    # Calculate similarity matrix based on the user-provided formula:
    # (u.v)^2 / ||u-v||^2 = (u.v)^2 / (||u||^2 + ||v||^2 - 2*u.v)
    dot_product_sim = jnp.matmul(representations, representations.T)
    
    squared_norms = jnp.sum(representations**2, axis=1, keepdims=True)
    # Calculate ||u-v||^2 for all pairs
    denominator = squared_norms + squared_norms.T - 2 * dot_product_sim
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    similarity_matrix = (dot_product_sim**2) / (denominator + epsilon)


    # --- Self-Similarity Masking ---
    # We mask out the diagonal elements (self-similarity) so they are not treated as negatives.
    identity_mask = jax.nn.one_hot(jnp.arange(2 * batch_size), num_classes=2 * batch_size, dtype=jnp.bool_)
    similarity_matrix = jnp.where(identity_mask, -1e9, similarity_matrix)

    # --- Temperature Scaling ---
    # Scale the logits by the temperature. This controls the sharpness of the
    # distribution, helping the model learn from hard negative examples.
    similarity_matrix /= temperature

    # Create labels for positive pairs.
    labels = jnp.arange(batch_size)
    positive_indices = jnp.concatenate([labels + batch_size, labels], axis=0)
    
    loss = optax.softmax_cross_entropy(
        logits=similarity_matrix,
        labels=jax.nn.one_hot(positive_indices, num_classes=2 * batch_size)
    ).mean()
    
    return loss


def regression_loss(p, z):
    # The authors of the paper encourage to not use a L2-normalized predictor.
    # However, they do use a L2-normalized target projection.
    # See page 4 of the paper.
    p = p / jnp.linalg.norm(p, axis=-1, keepdims=True)
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    return 2 - 2 * (p * z).sum(axis=-1)


def byol_loss_fn(online_model, target_model, batch):
    """
    BYOL loss function.
    """
    img1, img2 = batch['image1'], batch['image2']

    # Online network predictions for both views
    online_proj_one = online_model(img1, training=True, return_activations_for_layer='projector')
    online_proj_two = online_model(img2, training=True, return_activations_for_layer='projector')

    # Target network predictions for both views
    # Stop gradient to prevent gradients from flowing to the target network
    target_proj_one = jax.lax.stop_gradient(target_model(img1, training=True, return_activations_for_layer='projector'))
    target_proj_two = jax.lax.stop_gradient(target_model(img2, training=True, return_activations_for_layer='projector'))

    # Symmetrized loss
    loss_one = regression_loss(online_proj_one, target_proj_two)
    loss_two = regression_loss(online_proj_two, target_proj_one)

    loss = (loss_one + loss_two).mean()
    return loss
