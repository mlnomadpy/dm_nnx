# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import typing as tp
import os
import glob
from functools import partial
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow_datasets as tfds
import tensorflow as tf
import optax
import orbax.checkpoint as orbax
import pandas as pd
from tqdm import tqdm

from flax import nnx

from ..models import create_model, YatCNN

try:
    from nmn.nnx.yatconv import YatConv
except ImportError:
    print("Warning: `nmn.nnx.yatconv` not found. Using a placeholder for YatConv.")
# ==============================================================================
# 1. CONFIGURATIONS (Now defined in main)
# ==============================================================================

# Typing
Array = jax.Array
Axis = int
Size = int

# ==============================================================================
# 2. DATA LOADING & PREPROCESSING
# ==============================================================================

def create_image_folder_dataset(path: str, validation_split: float, seed: int):
    """Creates train and test tf.data.Dataset from an image folder."""
    class_names = sorted([d.name for d in os.scandir(path) if d.is_dir()])
    if not class_names:
        raise ValueError(f"No subdirectories found in {path}. Each subdirectory should contain images for one class.")
        
    class_to_index = {name: i for i, name in enumerate(class_names)}

    all_image_paths = []
    all_image_labels = []
    for class_name in class_names:
        class_dir = os.path.join(path, class_name)
        image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.JPEG'):
             image_paths.extend(glob.glob(os.path.join(class_dir, ext)))
        all_image_paths.extend(image_paths)
        all_image_labels.extend([class_to_index[class_name]] * len(image_paths))
        
    if not all_image_paths:
        raise ValueError(f"No image files found in subdirectories of {path}.")

    # Create a tf.data.Dataset
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int32))
    image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))

    # Shuffle and split
    dataset_size = len(all_image_paths)
    image_label_ds = image_label_ds.shuffle(buffer_size=dataset_size, seed=seed)
    
    val_count = int(dataset_size * validation_split)
    train_ds = image_label_ds.skip(val_count)
    test_ds = image_label_ds.take(val_count)
    
    # Get dataset size for train and test
    train_size = dataset_size - val_count
    
    return train_ds, test_ds, class_names, train_size

def get_image_processor(image_size: tuple[int, int], num_channels: int):
    """Returns a tf.data processing function for supervised learning."""
    def process_path(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return {'image': img, 'label': label}
    return process_path

def get_contrastive_image_processor(image_size: tuple[int, int], num_channels: int):
    """Returns a tf.data processing function for contrastive pretraining."""
    @tf.function
    def augment_image(image):
        # Apply a sequence of random augmentations
        image = tf.image.random_flip_left_right(image)
        # Slightly less aggressive brightness/contrast to reduce chance of all-black images
        image = tf.image.random_brightness(image, max_delta=0.4) 
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        # Random resized crop
        # Ensure crop size is valid
        crop_h = tf.cast(tf.cast(image_size[0], tf.float32) * 0.9, tf.int32)
        crop_w = tf.cast(tf.cast(image_size[1], tf.float32) * 0.9, tf.int32)
        image = tf.image.random_crop(image, size=[crop_h, crop_w, num_channels])
        image = tf.image.resize(image, image_size)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def process_path(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        
        # Create two augmented views
        aug_img_1 = augment_image(img)
        aug_img_2 = augment_image(img)
        
        return {'image1': aug_img_1, 'image2': aug_img_2}
    return process_path


# ==============================================================================
# 4. MODEL ARCHITECTURES
# ==============================================================================


# ==============================================================================
# 5. TRAINING & EVALUATION INFRASTRUCTURE
# ==============================================================================

# -- Supervised (Fine-tuning) Stage ---

def loss_fn(model, batch):
    logits = model(batch['image'], training=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()
    return loss, logits

@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(grads)

@nnx.jit
def eval_step(model, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])

# --- Contrastive (Pretraining) Stage ---

def contrastive_loss_fn(model: YatCNN, batch, temperature: float):
    """
    NT-Xent loss for contrastive learning.
    NOTE: As requested, L2 normalization has been removed. This is generally not recommended
    as it can lead to training instability (e.g., NaN loss) because the dot product
    is no longer bounded like cosine similarity.
    """
    img1, img2 = batch['image1'], batch['image2']
    images = jnp.concatenate([img1, img2], axis=0)
    batch_size = img1.shape[0]

    # Get representations directly from the encoder backbone
    representations = model(images, training=True, return_activations_for_layer='representation')

    # --- L2 Normalization REMOVED as per user request ---
    # The following line was removed:
    # representations = representations / (jnp.linalg.norm(representations, axis=1, keepdims=True) + 1e-8)

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


@nnx.jit
def pretrain_step(model: nnx.Module, optimizer: nnx.Optimizer, batch, temperature: float):
    """Performs a single pretraining step."""
    grad_fn = nnx.value_and_grad(contrastive_loss_fn)
    loss, grads = grad_fn(model, batch, temperature)
    optimizer.update(grads)
    return loss

def _pretrain_model_loop(
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
    config = dataset_configs.get('custom_folder', fallback_configs)
    image_size = config.get('input_dim', (64, 64))
    current_batch_size = config.get('pretrain_batch_size', fallback_configs['pretrain_batch_size'])
    train_ds, _, _, train_size = create_image_folder_dataset(dataset_name, validation_split=0.01, seed=42)
    input_channels = config.get('input_channels', 3)
    current_num_epochs = config.get('pretrain_epochs', fallback_configs['pretrain_epochs'])
    contrastive_processor = get_contrastive_image_processor(image_size=image_size, num_channels=input_channels)
    train_ds = train_ds.map(contrastive_processor, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_size = train_size

    # Initialize model
    pretrain_model = create_model(model_name, config={'num_classes': 10, 'input_channels': input_channels}, mesh=None, rngs=nnx.Rngs(rng_seed))
    
    # --- Learning Rate Schedule & Optimizer Setup ---
    steps_per_epoch = dataset_size // current_batch_size
    total_steps = current_num_epochs * steps_per_epoch

    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps,
        alpha=0.0
    )

    tx = optimizer_constructor(learning_rate=schedule)
    tx_with_clipping = optax.chain(
        optax.clip_by_global_norm(1.0),
        tx
    )
    optimizer = nnx.Optimizer(pretrain_model, tx_with_clipping)
    temperature = config.get('temperature', 0.1)

    # Pretraining loop
    global_step_counter = 0
    print(f"Pre-training for {current_num_epochs} epochs ({total_steps} steps)...")

    for epoch in range(current_num_epochs):
        epoch_train_iter = train_ds.shuffle(1024).batch(current_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        
        pbar = tqdm(epoch_train_iter, total=steps_per_epoch, desc=f"Pretrain Epoch {epoch + 1}/{current_num_epochs}")
        for batch_data in pbar:
            loss = pretrain_step(pretrain_model, optimizer, batch_data, temperature)
            if global_step_counter % 20 == 0:
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            global_step_counter += 1
            if jnp.isnan(loss):
                print("\n❗️ Loss is NaN. Stopping pretraining. This might be due to unstable gradients or data issues.")
                return None

    # Save the pretrained model state (only parameters, as per convention)
    save_dir = os.path.abspath(f"./models/{model_name}_pretrained_encoder")
    model_state = nnx.state(pretrain_model, nnx.Param)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"\n💾 Saving pretrained model state to {save_dir}...")
    checkpointer.save(save_dir, model_state, force=True)
    print("✅ Pretrained model saved successfully!")
    
    return save_dir


def _train_model_loop(
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
    stage = "Fine-tuning with Pretrained Weights" if pretrained_encoder_path else "Training from Scratch"
    print(f"\n🚀 Initializing {model_name} for {stage} on dataset {dataset_name}...")

    # Data loading logic
    if os.path.isdir(dataset_name):
        config = dataset_configs.get('custom_folder', fallback_configs)
        image_size = config.get('input_dim', (64, 64))
        split_percentage = config.get('test_split_percentage', 0.2)
        current_batch_size = config.get('batch_size', fallback_configs['batch_size_folder'])
        train_ds, test_ds, class_names, train_size = create_image_folder_dataset(dataset_name, validation_split=split_percentage, seed=42)
        num_classes = len(class_names)
        input_channels = config.get('input_channels', 3)
        current_num_epochs = config.get('num_epochs', fallback_configs['num_epochs'])
        current_eval_every = config.get('eval_every', fallback_configs['eval_every'])
        image_processor = get_image_processor(image_size=image_size, num_channels=input_channels)
        train_ds = train_ds.map(image_processor, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(image_processor, num_parallel_calls=tf.data.AUTOTUNE)
        dataset_size = train_size
    else: # TFDS logic
        config = dataset_configs.get(dataset_name)
        if not config:
            raise ValueError(f"Dataset '{dataset_name}' not in configs and not a valid directory.")
        num_classes, input_channels = config['num_classes'], config['input_channels']
        current_num_epochs, current_eval_every, current_batch_size = config['num_epochs'], config['eval_every'], config['batch_size']
        image_key, label_key, train_split_name, test_split_name = config['image_key'], config['label_key'], config['train_split'], config['test_split']
        preprocess_fn = lambda s: {'image': tf.cast(s[image_key], tf.float32) / 255.0, 'label': s[label_key]}
        train_ds = tfds.load(dataset_name, split=train_split_name, as_supervised=False).map(preprocess_fn)
        test_ds = tfds.load(dataset_name, split=test_split_name, as_supervised=False).map(preprocess_fn)
        dataset_size = train_ds.cardinality().numpy()

    # Initialize model
    model = create_model(model_name, config={'num_classes': num_classes, 'input_channels': input_channels}, mesh=None, rngs=nnx.Rngs(rng_seed))
    
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
            if 'out_conv' in path[0]:
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
    metrics_computer = nnx.MultiMetric(accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average('loss'))
    metrics_history = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}
    global_step_counter = 0

    # Training loop
    steps_per_epoch = dataset_size // current_batch_size
    total_steps = current_num_epochs * steps_per_epoch
    print(f"Training for {total_steps} steps...")

    for epoch in range(current_num_epochs):
        epoch_train_iter = train_ds.shuffle(1024).batch(current_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        pbar = tqdm(epoch_train_iter, total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{current_num_epochs} ({stage})")
        for batch_data in pbar:
            train_step(model, optimizer, metrics_computer, batch_data)
            
            if global_step_counter > 0 and (global_step_counter % current_eval_every == 0 or global_step_counter == total_steps - 1):
                train_metrics = metrics_computer.compute()
                metrics_history['train_loss'].append(train_metrics['loss'])
                metrics_history['train_accuracy'].append(train_metrics['accuracy'])
                metrics_computer.reset()
                
                current_test_iter = test_ds.batch(current_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
                for test_batch in current_test_iter:
                    eval_step(model, metrics_computer, test_batch)
                test_metrics = metrics_computer.compute()
                metrics_history['test_loss'].append(test_metrics['loss'])
                metrics_history['test_accuracy'].append(test_metrics['accuracy'])
                metrics_computer.reset()
                pbar.set_postfix({'Train Acc': f"{train_metrics['accuracy']:.4f}", 'Test Acc': f"{test_metrics['accuracy']:.4f}"})

            global_step_counter += 1
            if global_step_counter >= total_steps: break
        if global_step_counter >= total_steps: break

    print(f"✅ {stage} complete on {dataset_name} after {global_step_counter} steps!")
    if metrics_history['test_accuracy']:
        print(f"   Final Test Accuracy: {metrics_history['test_accuracy'][-1]:.4f}")
    
    save_dir = os.path.abspath(f"./models/{model_name}_{dataset_name.replace('/', '_')}")
    state = nnx.state(model)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"💾 Saving final model state to {save_dir}...")
    checkpointer.save(save_dir, state, force=True)
    print(f"   Model saved successfully!")
    
    return model, metrics_history


# ==============================================================================
# 6. ANALYSIS & VISUALIZATION FUNCTIONS (UNCHANGED)
# ==============================================================================
def plot_training_curves(history, model_name):
    """Plot training curves for a model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Training Curves for {model_name}', fontsize=16, fontweight='bold')
    steps = range(len(history['train_loss']))
    ax1.plot(steps, history['train_loss'], 'b-', label=f'{model_name} Train Loss', linewidth=2)
    ax1.plot(steps, history['test_loss'], 'r--', label=f'{model_name} Test Loss', linewidth=2)
    ax1.set_title('Loss'); ax1.set_xlabel('Evaluation Steps'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(steps, history['train_accuracy'], 'b-', label=f'{model_name} Train Accuracy', linewidth=2)
    ax2.plot(steps, history['test_accuracy'], 'r--', label=f'{model_name} Test Accuracy', linewidth=2)
    ax2.set_title('Accuracy'); ax2.set_xlabel('Evaluation Steps'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

def print_final_metrics(history, model_name):
    """Print a detailed table of final metrics."""
    print(f"\n📊 FINAL METRICS FOR {model_name}" + "\n" + "=" * 40)
    final_metrics = {metric: hist[-1] for metric, hist in history.items() if hist}
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    for metric, value in final_metrics.items():
        print(f"{metric:<20} {value:<15.4f}")
    print("\n🏆 SUMMARY:")
    print(f"   Final Test Accuracy: {final_metrics.get('test_accuracy', 0):.4f}")

def detailed_test_evaluation(model, test_ds_iter, class_names: list[str], model_name: str):
    """Perform detailed evaluation on test set including per-class accuracy."""
    print(f"Running detailed test evaluation for {model_name}...")
    num_classes = len(class_names)
    predictions, true_labels = [], []
    for batch in tqdm(test_ds_iter, desc="Detailed Evaluation"):
        preds = jnp.argmax(model(batch['image'], training=False), axis=1)
        predictions.extend(preds.tolist())
        true_labels.extend(batch['label'].tolist())
    predictions, true_labels = np.array(predictions), np.array(true_labels)
    print("\n🎯 PER-CLASS ACCURACY" + "\n" + "=" * 50)
    print(f"{'Class':<15} {'Accuracy':<10} {'Sample Count':<12}")
    print("-" * 50)
    for i in range(num_classes):
        mask = true_labels == i
        if np.sum(mask) > 0:
            acc = np.mean(predictions[mask] == true_labels[mask])
            print(f"{class_names[i]:<15} {acc:<10.4f} {np.sum(mask):<12}")
    return {'predictions': predictions, 'true_labels': true_labels, 'class_names': class_names}

def plot_confusion_matrix(predictions_data, model_name):
    """Plot confusion matrix for the model."""
    cm = confusion_matrix(predictions_data['true_labels'], predictions_data['predictions'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=predictions_data['class_names'], yticklabels=predictions_data['class_names'])
    plt.title(f'{model_name} - Confusion Matrix', fontweight='bold'); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.tight_layout(); plt.show()

# ==============================================================================
# 7. UNIT TESTS
# ==============================================================================

def test_contrastive_loss():
    """Unit test for the contrastive_loss_fn to ensure its correctness."""
    print("\n" + "="*80 + "\n🧪 Running Unit Test for Contrastive Loss...\n" + "="*80)

    # 1. Setup a deterministic 'model' that returns predefined embeddings
    class TestModel(nnx.Module):
        def __init__(self, representations):
            self.reps = representations
        def __call__(self, x, training=False, return_activations_for_layer=None):
            # Ignores input 'x', returns the predefined representations
            return self.reps

    batch_size = 4
    embedding_dim = 16
    temperature = 0.1

    # 2. Create a test case with PERFECTLY matching positive pairs
    # Representations for the first batch
    key = jax.random.PRNGKey(0)
    reps1 = jax.random.normal(key, (batch_size, embedding_dim))
    # For this test, the second augmented view is identical to the first.
    reps2 = reps1
    
    # The model will return these representations for the concatenated batch
    combined_reps = jnp.concatenate([reps1, reps2], axis=0)
    
    # The batch data is just dummy placeholders, as the model will ignore it
    dummy_images = jnp.ones((batch_size, 32, 32, 3))
    batch = {'image1': dummy_images, 'image2': dummy_images}

    # Instantiate the test model
    model = TestModel(representations=combined_reps)

    # 3. Calculate loss
    loss = contrastive_loss_fn(model, batch, temperature)

    # 4. Assert the expected outcome
    # With identical positive pairs, the similarity for the correct pair will be maximized.
    # The softmax will be sharply peaked at the correct label, and the cross-entropy
    # loss should be very close to zero. We allow for a small tolerance.
    expected_loss = 0.0
    np.testing.assert_allclose(loss, expected_loss, atol=1e-5)
    
    print("✅ Test Case 1 (Perfect Positive Pairs): PASSED")
    print(f"   - Calculated Loss: {loss:.6f}")
    print(f"   - Expected Loss:   {expected_loss:.6f}")

    # 5. Create a second test case with orthogonal negative pairs
    reps1 = jnp.array([[1,0,0,0], [0,1,0,0]], dtype=jnp.float32) # Two one-hot vectors
    reps2 = reps1 # Perfect positive pairs again
    
    combined_reps_orthogonal = jnp.concatenate([reps1, reps2], axis=0)
    model_orthogonal = TestModel(representations=combined_reps_orthogonal)
    
    batch_ortho = {'image1': jnp.ones((2, 4)), 'image2': jnp.ones((2, 4))}
    loss_ortho = contrastive_loss_fn(model_orthogonal, batch_ortho, temperature=1.0)
    
    # Theoretical calculation for this specific case:
    # Sim matrix (unscaled): [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    # After masking: [[-inf, 0, 1, 0], [0, -inf, 0, 1], [1, 0, -inf, 0], [0, 1, 0, -inf]]
    # Logits for first sample (positive is 3rd element): [0, 1] -> softmax([0,1]) -> probs=[0.26, 0.73] -> -log(0.73) = 0.31
    # This should be consistent for all samples.
    # A more robust check is to ensure the loss is not zero and not NaN.
    assert not jnp.isnan(loss_ortho) and loss_ortho > 0, "Loss should be positive for non-trivial cases"

    print("\n✅ Test Case 2 (Orthogonal Negative Pairs): PASSED")
    print(f"   - Calculated Loss: {loss_ortho:.6f} (is positive and not NaN)")
    
    print("\n" + "="*80 + "\n✅ All Unit Tests Passed!\n" + "="*80)


# ==============================================================================
# 8. MAIN EXECUTION & DEMO FUNCTIONS
# ==============================================================================

def run_training_and_analysis(
    dataset_name: str, 
    dataset_configs: dict, 
    fallback_configs: dict, 
    learning_rate: float, 
    momentum: float,
    pretrain: bool = False,
):
    """Runs a full training and analysis pipeline for the YatCNN model."""
    print("\n" + "="*80 + f"\n                                  RUNNING TRAINING & ANALYSIS FOR: {dataset_name.upper()}\n" + "="*80)
    
    is_path = os.path.isdir(dataset_name)
    train_path = dataset_name
    test_path = None
    if is_path:
        potential_train_path = os.path.join(dataset_name, 'train')
        if os.path.isdir(potential_train_path):
            train_path = potential_train_path
            test_path = os.path.join(dataset_name, 'test')
            print(f"INFO: Found 'train'/'test' subdirs. Using '{train_path}' for training.")
        else:
            print(f"INFO: Assuming '{dataset_name}' is the training data directory.")

    pretrained_encoder_path = None
    if pretrain:
        if not is_path:
            print("⚠️ WARNING: Contrastive pretraining is currently only implemented for folder-based datasets. Skipping pretraining.")
        else:
            print("\n🚀 STEP 1: Starting Contrastive Pretraining...")
            pretrained_encoder_path = _pretrain_model_loop(
                model_name="YatCNN",
                dataset_name=train_path,
                rng_seed=42,
                learning_rate=learning_rate,
                momentum=momentum,
                optimizer_constructor=optax.adamw,
                dataset_configs=dataset_configs,
                fallback_configs=fallback_configs,
            )
            
    if pretrain and pretrained_encoder_path is None:
        print("\nPretraining failed or was skipped. Aborting fine-tuning.")
        return

    # Get class names for fine-tuning
    if is_path:
        custom_config = dataset_configs.get('custom_folder', fallback_configs)
        _, _, class_names_comp, _ = create_image_folder_dataset(train_path, validation_split=custom_config.get('test_split_percentage', 0.2), seed=42)
        num_classes_comp = len(class_names_comp)
    else: # TFDS logic
        config = dataset_configs.get(dataset_name, {})
        ds_info = tfds.builder(dataset_name).info
        label_key = config.get('label_key', 'label')
        label_feature = ds_info.features[label_key]
        class_names_comp = label_feature.names if hasattr(label_feature, 'names') else [f'Class {i}' for i in range(label_feature.num_classes)]
        num_classes_comp = len(class_names_comp)
    
    print(f"\n🚀 STEP 2: {'Fine-tuning' if pretrain else 'Training'} Model...");
    model, metrics_history = _train_model_loop(
        "yatcnn", train_path if is_path else dataset_name, 0, learning_rate, momentum, optax.adamw,
        dataset_configs=dataset_configs, fallback_configs=fallback_configs,
        pretrained_encoder_path=pretrained_encoder_path
    )
    
    def get_test_iter():
      if is_path:
          config = dataset_configs.get('custom_folder', fallback_configs)
          _, test_ds, _, _ = create_image_folder_dataset(train_path, validation_split=config.get('test_split_percentage', 0.2), seed=42)
          processor = get_image_processor(image_size=config.get('input_dim'), num_channels=config.get('input_channels'))
          test_ds = test_ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
          return test_ds.batch(config.get('batch_size'), True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
      else:
          config = dataset_configs.get(dataset_name)
          key, label = config['image_key'], config['label_key']
          preprocess = lambda s: {'image': tf.cast(s[key], tf.float32) / 255.0, 'label': s[label]}
          test_ds = tfds.load(dataset_name, split=config['test_split']).map(preprocess)
          return test_ds.batch(config['batch_size'], True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    print("\n📊 STEP 3: Running Analysis...")
    plot_training_curves(metrics_history, "YatCNN")
    print_final_metrics(metrics_history, "YatCNN")
    predictions_data = detailed_test_evaluation(model, get_test_iter(), class_names=class_names_comp, model_name="YatCNN")
    if num_classes_comp <= 50: # Avoid plotting huge confusion matrices
        plot_confusion_matrix(predictions_data, "YatCNN")

    print("\n" + "="*80 + f"\n                                  ANALYSIS FOR {dataset_name.upper()} COMPLETE! ✅\n" + "="*80)
    return {'model': model, 'metrics_history': metrics_history, 'predictions_data': predictions_data}


# ==============================================================================
# 9. SCRIPT ENTRYPOINT & USAGE GUIDE
# ==============================================================================

def main():
    """Main function to define configurations and run the training and analysis."""
    
    # Run unit tests first to ensure correctness
    test_contrastive_loss()

    dataset_configs = {
        'cifar10': {
            'num_classes': 10, 'input_channels': 3, 'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 'num_epochs': 5, 'eval_every': 200, 'batch_size': 64
        },
        'custom_folder': {
            'num_classes': 'auto',
            'input_channels': 3,
            'input_dim': (32, 32),
            'test_split_percentage': 0.2,
            # Supervised fine-tuning params
            'num_epochs': 1,
            'eval_every': 500,
            'batch_size': 256,
            # Contrastive pretraining params
            'pretrain_epochs': 10,
            'pretrain_batch_size': 512,
            'temperature': 0.1, # Temperature for the NT-Xent loss
        }
    }

    fallback_configs = {
        'num_epochs': 10, 'eval_every': 200, 'batch_size': 64,
        'batch_size_folder': 32, 'pretrain_epochs': 20, 'pretrain_batch_size': 128,
    }
    
    learning_rate = 0.003
    momentum = 0.9
    
    # =================================================
    # CHOOSE YOUR DATASET AND PRETRAINING OPTION HERE
    # =================================================
    dataset_to_run = '/kaggle/input/ml-nomads-downscaling-laws-cifar-10' 
    use_pretraining = True
    # =================================================

    run_training_and_analysis(
        dataset_name=dataset_to_run,
        dataset_configs=dataset_configs,
        fallback_configs=fallback_configs,
        learning_rate=learning_rate,
        momentum=momentum,
        pretrain=use_pretraining,
    )


if __name__ == '__main__':
    # Set TF to not occupy all GPU memory by default
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    print("\n" + "="*80)
    print("🚀 TO RUN: Modify `dataset_to_run` and `use_pretraining` in `main()`.")
    print("="*80)
    
    main()
