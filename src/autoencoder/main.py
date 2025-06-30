# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import typing as tp
import os
import glob
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import tensorflow_datasets as tfds
import tensorflow as tf
import optax
import orbax.checkpoint as orbax
from tqdm import tqdm
from flax import nnx


# ==============================================================================
# 1. CONFIGURATIONS
# ==============================================================================
# Typing
Array = jax.Array
Axis = int
Size = int

# ==============================================================================
# 2. DATA LOADING & PREPROCESSING
# ==============================================================================

# --- For Local Image Folders ---
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

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int32))
    image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))

    dataset_size = len(all_image_paths)
    image_label_ds = image_label_ds.shuffle(buffer_size=dataset_size, seed=seed)

    val_count = int(dataset_size * validation_split)
    train_ds = image_label_ds.skip(val_count)
    test_ds = image_label_ds.take(val_count)

    train_size = dataset_size - val_count

    return train_ds, test_ds, class_names, train_size

def get_image_processor(image_size: tuple[int, int], num_channels: int):
    """Returns a tf.data processing function for supervised learning from a path."""
    def process_path(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return {'image': img, 'label': label}
    return process_path

# --- For TFDS Datasets ---
def get_tfds_processor(image_size: tuple[int, int], image_key: str, label_key: str):
    """Returns a processing function for TFDS datasets."""
    def _process(sample):
        img = tf.cast(sample[image_key], tf.float32)
        img = tf.image.resize(img, image_size)
        img = img / 255.0
        return {'image': img, 'label': sample[label_key]}
    return _process

# --- Universal Pretraining Processors ---
def get_autoencoder_pretrain_processor(image_size: tuple[int, int]):
    """Returns a processing function for autoencoder pretraining that works on already-loaded images."""
    def _process(data):
        # For autoencoders, we just need the image itself.
        return {'image': data['image']}
    return _process

# ==============================================================================
# 3. MODEL ARCHITECTURES
# ==============================================================================
class YatCNN(nnx.Module):
    """YAT CNN model (Encoder)."""
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(input_channels, 32, kernel_size=(7, 7), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(5, 5), rngs=rngs)
        self.conv3 = nnx.Conv(64, 128, kernel_size=(5, 5), rngs=rngs)
        self.conv4 = nnx.Conv(128, 256, kernel_size=(5, 5), rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout4 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.out_linear = nnx.Linear(256, num_classes, use_bias=False, rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        # Learnable mask token for MAE-style pretraining.
        self.mask_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, 1, 8)))
        # This model now manages its own RNG state for masking and dropout
        self.rngs = rngs

    def __call__(self, x, training: bool = False, return_activations_for_layer: tp.Optional[str] = None, 
                 apply_masking: bool = False, mask_ratio: float = 0.75):
        x = self.conv1(x)
        x = jax.nn.relu(x)
        if return_activations_for_layer == 'conv1': return x
        x = self.dropout1(x, deterministic=not training)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = jax.nn.relu(x)
        if return_activations_for_layer == 'conv2': return x
        x = self.dropout2(x, deterministic=not training)
        x = self.avg_pool(x)

        x = self.conv3(x)
        x = jax.nn.relu(x)
        if return_activations_for_layer == 'conv3': return x
        x = self.dropout3(x, deterministic=not training)
        x = self.avg_pool(x)

        x = self.conv4(x)
        x = jax.nn.relu(x)
        if return_activations_for_layer == 'bottleneck': return x
        x = self.dropout4(x, deterministic=not training)

        representation = jnp.mean(x, axis=(1, 2))
        x = self.out_linear(representation)
        return x

class Decoder(nnx.Module):
    """Decoder with ConvTranspose to reconstruct an image from a bottleneck."""
    def __init__(self, *, output_channels: int, rngs: nnx.Rngs):
        self.deconv1 = nnx.ConvTranspose(256, 128, kernel_size=(5, 5), strides=(2, 2), padding='SAME', rngs=rngs)
        self.deconv2 = nnx.ConvTranspose(128, 64, kernel_size=(5, 5), strides=(2, 2), padding='SAME', rngs=rngs)
        self.deconv3 = nnx.ConvTranspose(64, 32, kernel_size=(7, 7), strides=(2, 2), padding='SAME', rngs=rngs)
        self.final_conv = nnx.Conv(32, output_channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)

    def __call__(self, x, training: bool = False):
        x = self.deconv1(x)
        x = jax.nn.relu(x)
        x = self.deconv2(x)
        x = jax.nn.relu(x)
        x = self.deconv3(x)
        x = jax.nn.relu(x)
        x = self.final_conv(x)
        x = jax.nn.sigmoid(x)
        return x

class ConvAutoencoder(nnx.Module):
    """A Convolutional Autoencoder model."""
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        self.encoder = YatCNN(num_classes=num_classes, input_channels=input_channels, rngs=rngs)
        self.decoder = Decoder(output_channels=input_channels, rngs=rngs)

    def __call__(self, x, training: bool = False):
        bottleneck = self.encoder(
            x, 
            training=training, 
            return_activations_for_layer='bottleneck', 
            apply_masking=True
        )
        reconstructed_image = self.decoder(bottleneck, training=training)
        return reconstructed_image

# ==============================================================================
# 4. TRAINING & EVALUATION INFRASTRUCTURE
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
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(grads)

@nnx.jit
def eval_step(model, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])

# --- Autoencoder (Pretraining) Stage ---
def autoencoder_loss_fn(model: ConvAutoencoder, batch):
    original_image = batch['image']
    reconstructed_image = model(original_image, training=True)
    loss = jnp.mean(jnp.square(original_image - reconstructed_image))
    return loss

@nnx.jit
def pretrain_autoencoder_step(model: ConvAutoencoder, optimizer: nnx.Optimizer, batch):
    grad_fn = nnx.value_and_grad(autoencoder_loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss


# ==============================================================================
# 5. ANALYSIS & VISUALIZATION FUNCTIONS
# ==============================================================================
def plot_training_curves(history, model_name):
    """Plot training curves for a model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Fine-Tuning Curves for {model_name}', fontsize=16, fontweight='bold')
    steps = range(len(history['train_loss']))
    ax1.plot(steps, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(steps, history['test_loss'], 'r--', label='Test Loss', linewidth=2)
    ax1.set_title('Loss')
    ax1.set_xlabel('Evaluation Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(steps, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(steps, history['test_accuracy'], 'r--', label='Test Accuracy', linewidth=2)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Evaluation Steps')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_final_metrics(history, model_name):
    """Print a detailed table of final metrics."""
    print(f"\n📊 FINAL METRICS FOR {model_name}" + "\n" + "=" * 40)
    final_metrics = {metric: hist[-1] for metric, hist in history.items() if hist}
    if not final_metrics:
        print("No metrics recorded.")
        return
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    for metric, value in final_metrics.items():
        print(f"{metric:<20} {value:<15.4f}")
    print("\n🏆 SUMMARY:")
    print(f"   Final Test Accuracy: {final_metrics.get('test_accuracy', 0):.4f}")

def detailed_test_evaluation(model, test_ds_iter, class_names: list[str], model_name: str):
    """Perform detailed evaluation on test set including per-class accuracy."""
    print(f"\n🔬 Running detailed test evaluation for {model_name}...")
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
    plt.title(f'{model_name} - Confusion Matrix', fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def visualize_tsne(encoder: YatCNN, dataset_iter, class_names: list[str], title: str, num_samples: int = 1000):
    """Extracts embeddings, runs t-SNE, and plots the result."""
    print(f"\n🎨 Generating t-SNE plot: {title}...")
    
    all_embeddings = []
    all_labels = []
    
    batch_size = 32 # Assume a default batch size for the iterator
    for batch in tqdm(dataset_iter, desc="Extracting embeddings for t-SNE", total=int(np.ceil(num_samples / batch_size))):
        embeddings = encoder(batch['image'], training=False, return_activations_for_layer='bottleneck')
        flat_embeddings = jnp.reshape(embeddings, (embeddings.shape[0], -1))
        
        all_embeddings.append(np.array(flat_embeddings))
        all_labels.append(np.array(batch['label']))
        
        if len(np.concatenate(all_labels)) >= num_samples:
            break

    all_embeddings = np.concatenate(all_embeddings, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]

    if all_embeddings.shape[0] < 2:
        print("Not enough samples for t-SNE plot.")
        return

    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=all_labels, cmap=plt.cm.get_cmap("jet", len(class_names)))
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(handles=scatter.legend_elements(prop="colors")[0], labels=class_names)
    plt.grid(True, alpha=0.3)
    plt.show()

# ==============================================================================
# 6. PRETRAINING & FINE-TUNING LOOPS
# ==============================================================================
def _pretrain_autoencoder_loop(
    model_name: str,
    dataset_name: str,
    rng_seed: int,
    learning_rate: float,
    optimizer_constructor: tp.Callable,
    dataset_configs: dict,
    fallback_configs: dict,
):
    """Helper function to pretrain a model using a Convolutional Autoencoder."""
    print(f"\n🚀 Starting Autoencoder Pretraining for {model_name} on {dataset_name}...")
    is_path = os.path.isdir(dataset_name)
    config = dataset_configs.get(dataset_name) if not is_path else dataset_configs.get('custom_folder', fallback_configs)

    image_size = config.get('input_dim', (64, 64))
    input_channels = config.get('input_channels', 3)
    current_batch_size = config.get('pretrain_batch_size', fallback_configs['pretrain_batch_size'])
    current_num_epochs = config.get('pretrain_epochs', fallback_configs['pretrain_epochs'])
    
    if is_path:
        train_ds, _, class_names, train_size = create_image_folder_dataset(dataset_name, validation_split=0.01, seed=42)
        processor = get_image_processor(image_size=image_size, num_channels=input_channels)
        train_ds = train_ds.map(processor)
    else: # TFDS
        train_ds, ds_info = tfds.load(
            dataset_name,
            split=config['train_split'],
            shuffle_files=True,
            as_supervised=False,
            with_info=True,
        )
        class_names = ds_info.features[config['label_key']].names
        train_size = ds_info.splits[config['train_split']].num_examples
        processor = get_tfds_processor(image_size, config['image_key'], config['label_key'])
        train_ds = train_ds.map(processor)

    autoencoder_model = ConvAutoencoder(num_classes=len(class_names), input_channels=input_channels, rngs=nnx.Rngs(rng_seed))
    
    steps_per_epoch = train_size // current_batch_size
    total_steps = current_num_epochs * steps_per_epoch
    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=total_steps)
    tx = optimizer_constructor(learning_rate=schedule)
    optimizer = nnx.Optimizer(autoencoder_model, tx)

    print(f"Pre-training for {current_num_epochs} epochs ({total_steps} steps)...")
    for epoch in range(current_num_epochs):
        epoch_train_iter = train_ds.shuffle(1024).batch(current_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        pbar = tqdm(epoch_train_iter, total=steps_per_epoch, desc=f"Autoencoder Epoch {epoch + 1}/{current_num_epochs}")
        for batch_data in pbar:
            loss = pretrain_autoencoder_step(autoencoder_model, optimizer, batch_data)
            pbar.set_postfix({'reconstruction_loss': f'{loss:.6f}'})
            if jnp.isnan(loss):
                print("\n❗️ Loss is NaN. Stopping pretraining.")
                return None
    
    save_dir = os.path.abspath(f"./models/{model_name}_autoencoder_pretrained_encoder")
    encoder_state = nnx.state(autoencoder_model.encoder, nnx.Param)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"\n💾 Saving pretrained ENCODER state to {save_dir}...")
    checkpointer.save(save_dir, encoder_state, force=True)
    
    visualize_tsne(autoencoder_model.encoder, train_ds.batch(32).as_numpy_iterator(), class_names, title="t-SNE of Autoencoder Pretrained Embeddings")
    
    return save_dir

def _train_model_loop(
    model_class: tp.Type[YatCNN],
    model_name: str,
    dataset_name: str,
    rng_seed: int,
    learning_rate: float,
    optimizer_constructor: tp.Callable,
    dataset_configs: dict,
    fallback_configs: dict,
    pretrained_encoder_path: tp.Optional[str] = None,
    freeze_encoder: bool = False,
):
    """Helper function to fine-tune a model and return it with its metrics history."""
    stage = "Fine-tuning with Pretrained Weights" if pretrained_encoder_path else "Training from Scratch"
    if freeze_encoder and pretrained_encoder_path:
        stage += " (Encoder Frozen)"
    print(f"\n🚀 Initializing {model_name} for {stage} on dataset {dataset_name}...")
    
    is_path = os.path.isdir(dataset_name)
    config = dataset_configs.get(dataset_name) if not is_path else dataset_configs.get('custom_folder', fallback_configs)

    image_size = config.get('input_dim', (64, 64))
    input_channels = config.get('input_channels', 3)
    current_num_epochs = config.get('num_epochs', fallback_configs['num_epochs'])
    current_eval_every = config.get('eval_every', fallback_configs['eval_every'])
    current_batch_size = config.get('batch_size', fallback_configs['batch_size'])

    if is_path:
        split_percentage = config.get('test_split_percentage', 0.2)
        train_ds, test_ds, class_names, train_size = create_image_folder_dataset(dataset_name, validation_split=split_percentage, seed=42)
        num_classes = len(class_names)
        processor = get_image_processor(image_size=image_size, num_channels=input_channels)
        train_ds = train_ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
    else: # TFDS logic
        (train_ds, test_ds), ds_info = tfds.load(
            dataset_name,
            split=[config['train_split'], config['test_split']],
            shuffle_files=True,
            as_supervised=False,
            with_info=True,
        )
        num_classes = ds_info.features[config['label_key']].num_classes
        class_names = ds_info.features[config['label_key']].names
        train_size = ds_info.splits[config['train_split']].num_examples
        
        processor = get_tfds_processor(image_size, config['image_key'], config['label_key'])
        
        train_ds = train_ds.map(processor)
        test_ds = test_ds.map(processor)

    model = model_class(num_classes=num_classes, input_channels=input_channels, rngs=nnx.Rngs(rng_seed))

    if pretrained_encoder_path:
        print(f"💾 Loading pretrained encoder weights from {pretrained_encoder_path}...")
        checkpointer = orbax.PyTreeCheckpointer()
        abstract_state = jax.eval_shape(lambda: nnx.state(model, nnx.Param))
        restored_params = checkpointer.restore(pretrained_encoder_path, item=abstract_state)
        nnx.update(model, restored_params)
        print("✅ Pretrained weights loaded successfully!")

    # Optimizer setup with optional freezing
    if freeze_encoder and pretrained_encoder_path:
        print("❄️ Freezing encoder weights. Only the final classification layer will be trained.")
        
        # FIX: Remove the problematic type hint from the partition function.
        def path_partition_fn(path: tp.Sequence[tp.Any], value: tp.Any):
            """Partitions parameters based on their path in the model's state."""
            # The path is a sequence of keys. We check if the first key is 'out_linear'.
            # The `hasattr` check adds robustness for different path entry types.
            if path and hasattr(path[0], 'key') and path[0].key == 'out_linear':
                return 'trainable'
            return 'frozen'

        # Get the parameter pytree from the model.
        params = nnx.state(model, nnx.Param)
        # Create the label pytree by mapping the partitioning function over the params.
        param_labels = jax.tree_util.tree_map_with_path(path_partition_fn, params)

        # Create different optimizers for each partition.
        trainable_tx = optimizer_constructor(learning_rate)
        frozen_tx = optax.set_to_zero()

        # Combine them using multi_transform, passing the generated label pytree.
        tx = optax.multi_transform(
            {'trainable': trainable_tx, 'frozen': frozen_tx},
            param_labels
        )
        optimizer = nnx.Optimizer(model, tx)
    else:
        # Default behavior: fine-tune all weights.
        optimizer = nnx.Optimizer(model, optimizer_constructor(learning_rate))

    metrics_computer = nnx.MultiMetric(accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average('loss'))
    metrics_history = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}
    
    steps_per_epoch = train_size // current_batch_size
    total_steps = current_num_epochs * steps_per_epoch
    print(f"Starting training for {total_steps} steps...")

    global_step_counter = 0
    for epoch in range(current_num_epochs):
        epoch_train_iter = train_ds.shuffle(1024).batch(current_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        pbar = tqdm(epoch_train_iter, total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{current_num_epochs} ({stage})")
        for batch_data in pbar:
            train_step(model, optimizer, metrics_computer, batch_data)
            
            if global_step_counter > 0 and (global_step_counter % current_eval_every == 0 or global_step_counter >= total_steps -1):
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

    print(f"✅ {stage} complete on {dataset_name} after {global_step_counter} steps!")
    if metrics_history['test_accuracy']:
        print(f"   Final Test Accuracy: {metrics_history['test_accuracy'][-1]:.4f}")
    
    save_dir = os.path.abspath(f"./models/{model_name}_{dataset_name.replace('/', '_')}")
    state = nnx.state(model)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"💾 Saving final model state to {save_dir}...")
    checkpointer.save(save_dir, state, force=True)
    return model, metrics_history, test_ds, class_names

# ==============================================================================
# 7. MAIN EXECUTION & DEMO FUNCTIONS
# ==============================================================================
def run_training_and_analysis(
    dataset_name: str,
    dataset_configs: dict,
    fallback_configs: dict,
    learning_rate: float,
    pretrain_method: tp.Optional[tp.Literal['contrastive', 'autoencoder']] = None,
    freeze_encoder: bool = False,
):
    """Runs a full training and analysis pipeline for the YatCNN model."""
    print("\n" + "="*80 + f"\nRUNNING TRAINING & ANALYSIS FOR: {dataset_name.upper()}\n" + "="*80)

    is_path = os.path.isdir(dataset_name)
    if not is_path and dataset_name not in dataset_configs:
        raise ValueError(f"Dataset '{dataset_name}' is not a valid path and has no configuration.")

    pretrained_encoder_path = None
    if pretrain_method:
        print(f"\n🚀 STEP 1: Starting Pretraining (Method: {pretrain_method.upper()})...")
        if pretrain_method == 'autoencoder':
            pretrained_encoder_path = _pretrain_autoencoder_loop(
                model_name="YatCNN", dataset_name=dataset_name,
                rng_seed=42, learning_rate=learning_rate, optimizer_constructor=optax.adamw,
                dataset_configs=dataset_configs, fallback_configs=fallback_configs,
            )
        else:
            print(f"Pretraining method '{pretrain_method}' is not supported. Skipping pretraining.")

    if pretrain_method and pretrained_encoder_path is None:
        print("\nPretraining failed or was skipped. Aborting fine-tuning.")
        return

    print(f"\n🚀 STEP 2: {'Fine-tuning' if pretrain_method else 'Training'} Model...");
    model, metrics_history, test_ds, class_names = _train_model_loop(
        YatCNN, "YatCNN", dataset_name, 0, learning_rate, optax.adamw,
        dataset_configs=dataset_configs, fallback_configs=fallback_configs,
        pretrained_encoder_path=pretrained_encoder_path,
        freeze_encoder=freeze_encoder
    )
    
    print("\n📊 STEP 3: Running Final Analysis...")
    plot_training_curves(metrics_history, "YatCNN")
    print_final_metrics(metrics_history, "YatCNN")
    
    config = dataset_configs.get(dataset_name) if not is_path else dataset_configs.get('custom_folder', fallback_configs)
    test_iter = test_ds.batch(config.get('batch_size')).as_numpy_iterator()
    predictions_data = detailed_test_evaluation(model, test_iter, class_names=class_names, model_name="YatCNN")
    
    if len(class_names) <= 50: # Avoid plotting huge confusion matrices
        plot_confusion_matrix(predictions_data, "YatCNN")

    print("\n" + "="*80 + f"\nANALYSIS FOR {dataset_name.upper()} COMPLETE! ✅\n" + "="*80)
    return {'model': model, 'metrics_history': metrics_history, 'predictions_data': predictions_data}

# ==============================================================================
# 8. SCRIPT ENTRYPOINT & USAGE GUIDE
# ==============================================================================
def main():
    """Main function to define configurations and run the training and analysis."""
    dataset_configs = {
        'cifar10': {
            'input_channels': 3, 'input_dim': (32, 32),
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 10, 'eval_every': 300, 'batch_size': 128,
            'pretrain_epochs': 100, 'pretrain_batch_size': 256, 'temperature': 0.1,
        },
        'custom_folder': {
            'input_channels': 3, 'input_dim': (32, 32),
            'test_split_percentage': 0.2,
            'num_epochs': 10, 'eval_every': 200, 'batch_size': 128,
            'pretrain_epochs': 50, 'pretrain_batch_size': 256, 'temperature': 0.1,
        }
    }

    fallback_configs = {
        'num_epochs': 10, 'eval_every': 200, 'batch_size': 64,
        'pretrain_epochs': 20, 'pretrain_batch_size': 128,
    }

    learning_rate = 0.003

    # =================================================
    # CHOOSE YOUR DATASET AND PRETRAINING OPTION HERE
    # =================================================
    # To use a TFDS dataset, provide its name (e.g., 'cifar10').
    # To use a local folder, provide the path (e.g., '/path/to/your/dataset').
    dataset_to_run = 'cifar10'

    # Choose the pretraining method:
    #   - 'autoencoder': Use the ConvTranspose autoencoder.
    #   - None: Train from scratch without pretraining.
    pretrain_method_to_use = 'autoencoder'
    
    # NEW: Set to True to freeze the encoder during fine-tuning.
    # Set to False to fine-tune the entire model.
    freeze_encoder_during_finetune = True
    # =================================================
    
    is_path = os.path.isdir(dataset_to_run)
    if not is_path and dataset_to_run not in dataset_configs:
        print("="*80)
        print(f"ERROR: Dataset '{dataset_to_run}' not found as a directory or in `dataset_configs`.")
        print("Please update the `dataset_to_run` variable or add a configuration for it.")
        print("="*80)
        return
    if is_path and 'custom_folder' not in dataset_configs:
         print("ERROR: `dataset_to_run` is a path, but no 'custom_folder' config found.")
         return


    run_training_and_analysis(
        dataset_name=dataset_to_run,
        dataset_configs=dataset_configs,
        fallback_configs=fallback_configs,
        learning_rate=learning_rate,
        pretrain_method=pretrain_method_to_use,
        freeze_encoder=freeze_encoder_during_finetune,
    )

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    main()
