import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import jax.numpy as jnp

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
