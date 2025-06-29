import os
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as orbax
import tensorflow as tf
from flax import nnx
from tqdm import tqdm

from ..common.data import create_image_folder_dataset, get_image_processor
from ..models import create_model


@nnx.jit
def mae_pretrain_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: tp.Dict[str, np.ndarray]):
    """Performs a single pretraining step for the MAE model."""

    def loss_fn(model):
        reconstructed_patches, masked_indices, unmasked_indices = model(batch["image"], train=True)
        
        # Get original patches
        patches = model.encoder.patch_embedding.unfold(batch["image"])
        
        # Get only the masked patches from the original image
        original_masked_patches = jnp.take(patches, masked_indices, axis=1)

        # Calculate reconstruction loss (MSE) on the masked patches
        loss = jnp.mean(jnp.square(reconstructed_patches - original_masked_patches))
        return loss

    grad_fn = nnx.grad(loss_fn)
    grads = grad_fn(model)
    optimizer.update(grads)
    # We don't have a separate metrics object here, just return loss for logging
    return loss_fn(model) # Rerun to get loss with updated params for logging


def mae_pretrain_model_loop(
    model_name: str,
    dataset_name: str,
    rng_seed: int,
    learning_rate: float,
    optimizer_constructor: tp.Callable,
    dataset_configs: dict,
    fallback_configs: dict,
):
    """Helper function to pretrain a MAE model."""
    print(f"\n🚀 Starting MAE Pretraining for {model_name} on {dataset_name}...")

    # Data loading for pretraining
    config = dataset_configs.get("custom_folder", fallback_configs)
    image_size = config.get("input_dim", (32, 32))
    current_batch_size = config.get(
        "pretrain_batch_size", fallback_configs["pretrain_batch_size"]
    )
    train_ds, _, _, train_size = create_image_folder_dataset(
        dataset_name, validation_split=0.01, seed=42
    )
    input_channels = config.get("input_channels", 3)
    current_num_epochs = config.get(
        "pretrain_epochs", fallback_configs["pretrain_epochs"]
    )
    
    # For MAE, we only need the images, no special contrastive augmentations
    image_processor = get_image_processor(
        image_size=image_size, num_channels=input_channels
    )
    train_ds = train_ds.map(image_processor, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_size = train_size

    # Initialize MAE model
    mae_config = {
        "image_size": image_size[0],
        "patch_size": 4, 
        "embed_dim": 256,
        "num_heads": 4,
        "mlp_dim": 512,
        "num_layers": 4,
        "num_classes": 10, # Dummy for encoder, not used in pretraining
        "dropout_rate": 0.1,
        "mask_ratio": config.get("mask_ratio", 0.75),
    }
    pretrain_model = create_model(
        "mae",
        config=mae_config,
        mesh=None,
        rngs=nnx.Rngs(rng_seed),
    )

    # --- Learning Rate Schedule & Optimizer Setup ---
    steps_per_epoch = dataset_size // current_batch_size
    total_steps = current_num_epochs * steps_per_epoch

    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=total_steps, alpha=0.0
    )

    tx = optimizer_constructor(learning_rate=schedule)
    optimizer = nnx.Optimizer(pretrain_model, tx)

    # Pretraining loop
    print(f"Pre-training for {current_num_epochs} epochs ({total_steps} steps)...")

    for epoch in range(current_num_epochs):
        with tqdm(
            total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{current_num_epochs}"
        ) as pbar:
            for batch in train_ds.batch(current_batch_size).as_numpy_iterator():
                loss = mae_pretrain_step(pretrain_model, optimizer, batch)
                pbar.set_postfix(loss=f"{loss:.4f}")
                pbar.update(1)

    # Save the pretrained encoder state
    save_dir = os.path.abspath(f"./models/{model_name}_pretrained_encoder")
    # We only want to save the state of the encoder, not the whole MAE model
    model_state = nnx.state(pretrain_model.encoder, nnx.Param)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"\n💾 Saving pretrained model state to {save_dir}...")
    checkpointer.save(save_dir, model_state, force=True)
    print("✅ Pretrained model saved successfully!")

    return save_dir
