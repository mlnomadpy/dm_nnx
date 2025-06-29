import os
import typing as tp

import jax
import numpy as np
import optax
import tensorflow_datasets as tfds
from flax import nnx

from ..common.analysis import (
    detailed_test_evaluation,
    plot_confusion_matrix,
    plot_training_curves,
    print_final_metrics,
)
from ..common.data import create_image_folder_dataset
from ..common.train import byol_pretrain_model_loop, train_model_loop


def run_training_and_analysis(
    dataset_name: str,
    dataset_configs: dict,
    fallback_configs: dict,
    learning_rate: float,
    momentum: float,
    pretrain: bool = False,
):
    """Runs a full training and analysis pipeline for the BYOL model."""
    print(
        "\n" + "=" * 80 + f"\n                                  RUNNING TRAINING & ANALYSIS FOR: {dataset_name.upper()}\n" + "=" * 80
    )

    is_path = os.path.isdir(dataset_name)
    train_path = dataset_name

    if is_path:
        potential_train_path = os.path.join(dataset_name, "train")
        if os.path.isdir(potential_train_path):
            train_path = potential_train_path
            os.path.join(dataset_name, "test")
            print(f"INFO: Found 'train'/'test' subdirs. Using '{train_path}' for training.")
        else:
            print(f"INFO: Assuming '{dataset_name}' is the training data directory.")

    pretrained_encoder_path = None
    if pretrain:
        if not is_path:
            print(
                "⚠️ WARNING: Contrastive pretraining is currently only implemented for folder-based datasets. Skipping pretraining."
            )
        else:
            print("\n🚀 STEP 1: Starting BYOL Pretraining...")
            pretrained_encoder_path = byol_pretrain_model_loop(
                model_name="yatcnn",
                dataset_name=train_path,
                rng_seed=42,
                learning_rate=learning_rate,
                optimizer_constructor=optax.adamw,
                dataset_configs=dataset_configs,
                fallback_configs=fallback_configs,
                momentum=0.99,  # Typical momentum for BYOL EMA
            )

    if pretrain and pretrained_encoder_path is None:
        print("\nPretraining failed or was skipped. Aborting fine-tuning.")
        return

    # Get class names for fine-tuning
    if is_path:
        custom_config = dataset_configs.get("custom_folder", fallback_configs)
        _, _, class_names_comp, _ = create_image_folder_dataset(
            train_path, validation_split=custom_config.get("test_split_percentage", 0.2), seed=42
        )
        num_classes_comp = len(class_names_comp)
    else:  # TFDS logic
        config = dataset_configs.get(dataset_name, {})
        ds_info = tfds.builder(dataset_name).info
        label_key = config.get("label_key", "label")
        label_feature = ds_info.features[label_key]
        class_names_comp = (
            label_feature.names
            if hasattr(label_feature, "names")
            else [f"Class {i}" for i in range(label_feature.num_classes)]
        )
        num_classes_comp = len(class_names_comp)

    print(f"\n🚀 STEP 2: {'Fine-tuning' if pretrain else 'Training'} Model...")
    model, metrics_history = train_model_loop(
        "yatcnn",
        train_path if is_path else dataset_name,
        0,
        learning_rate,
        momentum,
        optax.adamw,
        dataset_configs=dataset_configs,
        fallback_configs=fallback_configs,
        pretrained_encoder_path=pretrained_encoder_path,
    )

    def get_test_iter():
        if is_path:
            config = dataset_configs.get("custom_folder", fallback_configs)
        else:
            pass

    print("\n📊 STEP 3: Running Analysis...")
    plot_training_curves(metrics_history, "YatCNN")
    print_final_metrics(metrics_history, "YatCNN")
    predictions_data = detailed_test_evaluation(
        model, get_test_iter(), class_names=class_names_comp, model_name="YatCNN"
    )
    if num_classes_comp <= 50:
        plot_confusion_matrix(predictions_data, "YatCNN")

    print(
        "\n" + "=" * 80 + f"\n                                  ANALYSIS FOR {dataset_name.upper()} COMPLETE! ✅\n" + "=" * 80
    )
    return {
        "model": model,
        "metrics_history": metrics_history,
        "predictions_data": predictions_data,
    }


def main():
    """Main function to define configurations and run the training and analysis."""

    dataset_configs = {
        "cifar10": {
            "num_classes": 10,
            "input_channels": 3,
            "train_split": "train",
            "test_split": "test",
            "image_key": "image",
            "label_key": "label",
            "num_epochs": 5,
            "eval_every": 200,
            "batch_size": 64,
        },
        "custom_folder": {
            "num_classes": "auto",
            "input_channels": 3,
            "input_dim": (32, 32),
            "test_split_percentage": 0.2,
            "num_epochs": 1,
            "eval_every": 500,
            "batch_size": 256,
            "pretrain_epochs": 10,
            "pretrain_batch_size": 512,
            "temperature": 0.1,
        },
    }

    fallback_configs = {
        "num_epochs": 1,
        "eval_every": 100,
        "batch_size_folder": 128,
        "pretrain_epochs": 10,
        "pretrain_batch_size": 256,
    }

    # --- DEMO ON CIFAR-10 ---
    run_training_and_analysis(
        dataset_name="cifar10",
        dataset_configs=dataset_configs,
        fallback_configs=fallback_configs,
        learning_rate=1e-3,
        momentum=0.9,
        pretrain=False,  # Set to True to run BYOL pretraining
    )


if __name__ == "__main__":
    main()
