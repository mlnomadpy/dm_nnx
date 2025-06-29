import os
import typing as tp

import jax.numpy as jnp
import optax
import orbax.checkpoint as orbax
import tensorflow as tf
from flax import nnx
from tqdm import tqdm

from ..common.data import create_image_folder_dataset, get_contrastive_image_processor
from ..common.losses import contrastive_loss_fn
from ..models import create_model


@nnx.jit
def pretrain_step(model: nnx.Module, optimizer: nnx.Optimizer, batch, temperature: float):
    """Performs a single pretraining step."""
    grad_fn = nnx.value_and_grad(contrastive_loss_fn)
    loss, grads = grad_fn(model, batch, temperature)
    optimizer.update(grads)
    return loss


def pretrain_model_loop(
    model_name: str,
    dataset_name: str,
    rng_seed: int,
    learning_rate: float,
    momentum: float,
    optimizer_constructor: tp.Callable,
    dataset_configs: dict,
    fallback_configs: dict,
):
    """Helper function to pretrain a model using contrastive learning."""
    print(f"\n🚀 Starting Contrastive Pretraining for {model_name} on {dataset_name}...")

    # Data loading for pretraining
    config = dataset_configs.get("custom_folder", fallback_configs)
    image_size = config.get("input_dim", (64, 64))
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
    contrastive_processor = get_contrastive_image_processor(
        image_size=image_size, num_channels=input_channels
    )
    train_ds = train_ds.map(contrastive_processor, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_size = train_size

    # Initialize model
    pretrain_model = create_model(
        model_name,
        config={"num_classes": 10, "input_channels": input_channels},
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
    tx_with_clipping = optax.chain(optax.clip_by_global_norm(1.0), tx)
    optimizer = nnx.Optimizer(pretrain_model, tx_with_clipping)
    temperature = config.get("temperature", 0.1)

    # Pretraining loop
    global_step_counter = 0
    print(f"Pre-training for {current_num_epochs} epochs ({total_steps} steps)...")

    for epoch in range(current_num_epochs):
        epoch_train_iter = (
            train_ds.shuffle(1024)
            .batch(current_batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )

        pbar = tqdm(
            epoch_train_iter,
            total=steps_per_epoch,
            desc=f"Pretrain Epoch {epoch + 1}/{current_num_epochs}",
        )
        for batch_data in pbar:
            loss = pretrain_step(pretrain_model, optimizer, batch_data, temperature)
            if global_step_counter % 20 == 0:
                pbar.set_postfix({"loss": f"{loss:.4f}"})
            global_step_counter += 1
            if jnp.isnan(loss):
                print(
                    "\n❗️ Loss is NaN. Stopping pretraining. This might be due to unstable gradients or data issues."
                )
                return None

    # Save the pretrained model state (only parameters, as per convention)
    save_dir = os.path.abspath(f"./models/{model_name}_pretrained_encoder")
    model_state = nnx.state(pretrain_model, nnx.Param)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"\n💾 Saving pretrained model state to {save_dir}...")
    checkpointer.save(save_dir, model_state, force=True)
    print("✅ Pretrained model saved successfully!")

    return save_dir
