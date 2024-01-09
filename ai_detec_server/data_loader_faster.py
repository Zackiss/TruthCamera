import tensorflow as tf
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

# Define the parent directory for the training and validation data
parent_dir = "Dataset"

# Define a list of tag names
tags = ["ai", "nature"]

AUTOTUNE = tf.data.experimental.AUTOTUNE


# Define a function to load and preprocess the images
def load_and_preprocess_image(file_path, label_map, image_size):
    label = tf.strings.split(file_path, os.path.sep)[-2]
    label = label_map.lookup(label)
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# Define a function to create a dataset from a directory
def create_dataset(dir_path, label_map, image_size):
    file_paths = tf.io.gfile.glob(os.path.join(dir_path, "*", "*.*"))
    labels = [label_map.lookup(tf.strings.split(file_path, os.path.sep)[-2]) for file_path in tqdm(file_paths)]
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(lambda x, y: load_and_preprocess_image(x, label_map, image_size))
    return dataset


def build_data_generator(image_size, batch_size, test_mode=False):
    # Get a list of all the directories that have a "train" and "val" subdirectory
    dir_list = []
    for dir_name in os.listdir(parent_dir):
        if os.path.isdir(os.path.join(parent_dir, dir_name, "train")) and os.path.isdir(
                os.path.join(parent_dir, dir_name, "val")):
            if test_mode:
                if "Test" in dir_name:
                    dir_list.append(dir_name)
            else:
                # Not enough space to perform full-data training
                if "Test" not in dir_name and "Stable_diffusion_5" in dir_name:
                    dir_list.append(dir_name)

    # Create a label map for each directory
    label_maps = []
    for dir_name in dir_list:
        table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(tags),
                values=tf.constant([0, 1]),
            ),
            default_value=tf.constant(-1),
            name="class_weight"
        )
        label_maps.append(table)

    # Load the training data from the directories
    train_datasets = []
    for dir_name, label_map in zip(dir_list, label_maps):
        train_dir = os.path.join(parent_dir, dir_name, "train")
        train_ds = create_dataset(train_dir, label_map, image_size)
        train_datasets.append(train_ds)
    train_ds = train_datasets[0]
    for ds in train_datasets[1:]:
        train_ds = train_ds.concatenate(ds)
    train_ds = train_ds.shuffle(buffer_size=10000, seed=42)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    # Load the validation data from the directories
    val_datasets = []
    for dir_name, label_map in zip(dir_list, label_maps):
        val_dir = os.path.join(parent_dir, dir_name, "val")
        val_ds = create_dataset(val_dir, label_map, image_size)
        val_datasets.append(val_ds)
    val_ds = val_datasets[0]
    for ds in val_datasets[1:]:
        val_ds = val_ds.concatenate(ds)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    if test_mode:
        for image, label in train_ds.take(10):
            # Display the image and label
            print("Label:", label.numpy())
            for i in range(batch_size):
                plt.imshow(image.numpy()[i])
                plt.show()

    return train_ds, val_ds

    # train_data_gen = list(train_data_gen.as_numpy_iterator())
    # train_data_len = len(train_data_gen)
    # x_train = np.concatenate([train_data_gen[i][0] for i in trange(train_data_len)])
    # y_train = np.concatenate([train_data_gen[i][1] for i in trange(train_data_len)])
    # y_train = to_categorical(y_train, num_classes=2)
    # print(y_train)
    # vali_data_gen = list(vali_data_gen.as_numpy_iterator())
    # vali_data_len = len(vali_data_gen)
    # x_val = np.concatenate([vali_data_gen[i][0] for i in trange(vali_data_len)])
    # y_val = np.concatenate([vali_data_gen[i][1] for i in trange(vali_data_len)])
    # y_val = to_categorical(y_val, num_classes=2)
    # print("Data generator length (Batch count):", x_train.shape)
