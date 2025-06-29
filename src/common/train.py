import os
import typing as tp

import jax
import numpy as np
import optax
import orbax.checkpoint as orbax
import tensorflow as tf
from flax import nnx
from tqdm import tqdm

from ..byol.model import BYOL
from ..models import create_model
from .data import (
    create_image_folder_dataset,
    get_contrastive_image_processor,
    get_image_processor,
)
from .losses import byol_loss_fn, contrastive_loss_fn, supervised_loss_fn


@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(supervised_loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])
    optimizer.update(grads)


@nnx.jit
def eval_step(model, metrics: nnx.MultiMetric, batch):
    loss, logits = supervised_loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


@nnx.jit
def pretrain_step(model: nnx.Module, optimizer: nnx.Optimizer, batch, temperature: float):
    """Performs a single pretraining step."""
    grad_fn = nnx.value_and_grad(contrastive_loss_fn)
    loss, grads = grad_fn(model, batch, temperature)
    optimizer.update(grads)
    return loss


@nnx.jit
def byol_train_step(
    model: BYOL,
    optimizer: nnx.Optimizer,
    batch: tp.Dict[str, np.ndarray],
):
    """Performs a single BYOL pretraining step."""
    # Note: Flax NNX's jit and value_and_grad require pure functions.
    # The model and optimizer state are handled explicitly, so we can define
    # the loss function inside the step where the model is available.
    def loss_fn(model):
        return byol_loss_fn(model, batch)

    grad_fn = nnx.value_and_grad(loss_fn)

    # Compute loss and gradients for the online network
    loss, grads = grad_fn(model.online_network)

    # Update online network
    optimizer.update(grads)

    # Update target network using exponential moving average
    model.update_target_network()

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


def train_model_loop(
    model_name: str,
    dataset_name: str,
    rng_seed: int,
    learning_rate: float,
    momentum: float,
    optimizer_constructor: tp.Callable,
    dataset_configs: dict,
    fallback_configs: dict,
    pretrained_encoder_path: tp.Optional[str] = None,
):
    """Helper function to train a model and return it with its metrics history."""
    stage = (
        "Fine-tuning with Pretrained Weights"
        if pretrained_encoder_path
        else "Training from Scratch"
    )
    print(f"\n🚀 Initializing {model_name} for {stage} on dataset {dataset_name}...")

    # Data loading logic
    if os.path.isdir(dataset_name):
        config = dataset_configs.get("custom_folder", fallback_configs)
        image_size = config.get("input_dim", (64, 64))
        split_percentage = config.get("test_split_percentage", 0.2)
        current_batch_size = config.get(
            "batch_size", fallback_configs["batch_size_folder"]
        )
        train_ds, test_ds, class_names, train_size = create_image_folder_dataset(
            dataset_name, validation_split=split_percentage, seed=42
        )
        num_classes = len(class_names)
        input_channels = config.get("input_channels", 3)
        current_num_epochs = config.get("num_epochs", fallback_configs["num_epochs"])
        current_eval_every = config.get("eval_every", fallback_configs["eval_every"])
        image_processor = get_image_processor(
            image_size=image_size, num_channels=input_channels
        )
        train_ds = train_ds.map(image_processor, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(image_processor, num_parallel_calls=tf.data.AUTOTUNE)
        dataset_size = train_size
    else:  # TFDS logic
        config = dataset_configs.get(dataset_name)
        if not config:
            raise ValueError(
                f"Dataset '{dataset_name}' not in configs and not a valid directory."
            )
        num_classes, input_channels = config["num_classes"], config["input_channels"]
        (
            current_num_epochs,
            current_eval_every,
            current_batch_size,
        ) = (
            config["num_epochs"],
            config["eval_every"],
            config["batch_size"],
        )
        image_key, label_key, train_split_name, test_split_name = (
            config["image_key"],
            config["label_key"],
            config["train_split"],
            config["test_split"],
        )
        preprocess_fn = lambda s: {
            "image": tf.cast(s[image_key], tf.float32) / 255.0,
            "label": s[label_key],
        }
        train_ds = tfds.load(
            dataset_name, split=train_split_name, as_supervised=False
        ).map(preprocess_fn)
        test_ds = tfds.load(
            dataset_name, split=test_split_name, as_supervised=False
        ).map(preprocess_fn)
        dataset_size = train_ds.cardinality().numpy()

    # Initialize model
    model = create_model(
        model_name,
        config={"num_classes": num_classes, "input_channels": input_channels},
        mesh=None,
        rngs=nnx.Rngs(rng_seed),
    )

    # Load pretrained weights if a path is provided
    if pretrained_encoder_path:
        print(f"💾 Loading pretrained model weights from {pretrained_encoder_path}...")
        checkpointer = orbax.PyTreeCheckpointer()

        # Create an abstract version of the current model's state to use as a
        # template for restoring. This helps guide the checkpointer.
        abstract_state = jax.eval_shape(lambda: nnx.state(model, nnx.Param))

        # Restore the checkpoint, using the abstract state as a target.
        restored_params = checkpointer.restore(pretrained_encoder_path, item=abstract_state)

        # We must not load the 'out_conv' layer from the pretrained model,
        # as its shape is different in the fine-tuning model.
        # We can filter it out from the restored state before updating.

        # Find the path to the 'out_conv' layer in the model's state
        out_conv_path = None
        # Iterate directly over the flat_state object
        for path, _ in nnx.state(model, nnx.Param).flat_state():
            if "out_conv" in path[0]:
                out_conv_path = path
                break

        if out_conv_path and out_conv_path in restored_params:
            # Pop the out_conv from the restored params to avoid shape mismatches
            del restored_params[out_conv_path]

        # Update the model with the filtered restored state.
        # This will load all weights from the pretrained encoder backbone.
        nnx.update(model, restored_params)

        print("✅ Pretrained weights loaded successfully!")

    # Initialize optimizer and metrics for fine-tuning
    optimizer = nnx.Optimizer(model, optimizer_constructor(learning_rate, momentum))
    metrics_computer = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average("loss")
    )
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    global_step_counter = 0

    # Training loop
    steps_per_epoch = dataset_size // current_batch_size
    total_steps = current_num_epochs * steps_per_epoch
    print(f"Training for {total_steps} steps...")

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
            desc=f"Epoch {epoch + 1}/{current_num_epochs} ({stage})",
        )
        for batch_data in pbar:
            train_step(model, optimizer, metrics_computer, batch_data)

            if (
                global_step_counter > 0
                and (global_step_counter % current_eval_every == 0)
                or global_step_counter == total_steps - 1
            ):
                train_metrics = metrics_computer.compute()
                metrics_history["train_loss"].append(train_metrics["loss"])
                metrics_history["train_accuracy"].append(train_metrics["accuracy"])
                metrics_computer.reset()

                current_test_iter = (
                    test_ds.batch(current_batch_size, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE)
                    .as_numpy_iterator()
                )
                for test_batch in current_test_iter:
                    eval_step(model, metrics_computer, test_batch)
                test_metrics = metrics_computer.compute()
                metrics_history["test_loss"].append(test_metrics["loss"])
                metrics_history["test_accuracy"].append(test_metrics["accuracy"])
                metrics_computer.reset()
                pbar.set_postfix(
                    {
                        "Train Acc": f"{train_metrics['accuracy']:.4f}",
                        "Test Acc": f"{test_metrics['accuracy']:.4f}",
                    }
                )

            global_step_counter += 1
            if global_step_counter >= total_steps:
                break
        if global_step_counter >= total_steps:
            break

    print(f"✅ {stage} complete on {dataset_name} after {global_step_counter} steps!")
    if metrics_history["test_accuracy"]:
        print(f"   Final Test Accuracy: {metrics_history['test_accuracy'][-1]:.4f}")

    save_dir = os.path.abspath(f"./models/{model_name}_{dataset_name.replace('/', '_')}")
    state = nnx.state(model)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"💾 Saving final model state to {save_dir}...")
    checkpointer.save(save_dir, state, force=True)
    print(f"   Model saved successfully!")

    return model, metrics_history
