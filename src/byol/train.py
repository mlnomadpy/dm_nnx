import os
import typing as tp

import numpy as np
import optax
import orbax.checkpoint as orbax
import tensorflow as tf
from flax import nnx
from tqdm import tqdm

from ..common.data import create_image_folder_dataset, get_contrastive_image_processor
from ..common.losses import byol_loss_fn
from ..models import create_model
from .model import BYOL


@nnx.jit
def byol_train_step(
    model: BYOL,
    optimizer: nnx.Optimizer,
    batch: tp.Dict[str, np.ndarray],
):
    """Performs a single training step for the BYOL model."""

    def loss_fn(model: BYOL):
        # Forward pass for both views
        online_proj_one = model.online_network(batch["image1"], train=True)
        online_proj_two = model.online_network(batch["image2"], train=True)

        with nnx.flags(frozen=True):  # Target network is not updated by gradients
            target_proj_one = model.target_network(batch["image1"], train=True)
            target_proj_two = model.target_network(batch["image2"], train=True)

        loss = byol_loss_fn(online_proj_one, target_proj_two) + byol_loss_fn(
            online_proj_two, target_proj_one
        )
        return loss

    # Update online network
    grad_fn = nnx.grad(loss_fn)
    grads = grad_fn(model)
    optimizer.update(grads)

    # Update target network using EMA
    model.update_target_network()


def byol_pretrain_model_loop(
    model_name: str,
    dataset_name: str,
    rng_seed: int,
    learning_rate: float,
    optimizer_constructor: tp.Callable,
    dataset_configs: dict,
    fallback_configs: dict,
    momentum: float = 0.99,  # BYOL-specific momentum for EMA
):
    """Helper function to pretrain a model using BYOL."""
    print(f"\n🚀 Starting BYOL Pretraining for {model_name} on {dataset_name}...")

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

    # Initialize BYOL model
    rngs = nnx.Rngs(rng_seed)

    # The underlying encoder
    encoder = create_model(
        model_name,
        # num_classes is a dummy for YatCNN, not used in pretraining
        config={"num_classes": 10, "input_channels": input_channels},
        mesh=None,
        rngs=rngs,
    )

    # BYOL wrapper model
    byol_model = BYOL(
        encoder=encoder,
        momentum=momentum,
        rngs=rngs,
    )

    # --- Learning Rate Schedule & Optimizer Setup ---
    steps_per_epoch = dataset_size // current_batch_size
    total_steps = current_num_epochs * steps_per_epoch

    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=total_steps, alpha=0.0
    )

    tx = optimizer_constructor(learning_rate=schedule)
    # We only optimize the online network's parameters
    optimizer = nnx.Optimizer(byol_model.online_network, tx)

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
            desc=f"BYOL Pretrain Epoch {epoch + 1}/{current_num_epochs}",
        )
        for batch_data in pbar:
            loss = byol_train_step(byol_model, optimizer, batch_data)
            if global_step_counter % 20 == 0:
                pbar.set_postfix({"loss": f"{loss:.4f}"})
            global_step_counter += 1
            if np.isnan(loss):
                print(
                    "\n❗️ Loss is NaN. Stopping pretraining. This might be due to unstable gradients or data issues."
                )
                return None

    # Save the pretrained online encoder's state
    save_dir = os.path.abspath(f"./models/{model_name}_byol_pretrained_encoder")
    # We only want to save the state of the encoder, not the whole BYOL model
    model_state = nnx.state(byol_model.online_network.encoder, nnx.Param)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"\n💾 Saving pretrained model state to {save_dir}...")
    checkpointer.save(save_dir, model_state, force=True)
    print("✅ Pretrained model saved successfully!")

    return save_dir
