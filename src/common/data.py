import os
import glob
import tensorflow as tf

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
