import math
import random

import os
import imghdr
import shutil
import tensorflow as tf


def split_val_img_to_test(path):
    src_path = "./" + path + "/val"
    dest_path = "./" + path + "/test"

    for tag in ["ai", "nature"]:
        files = os.listdir(src_path + "/{0}/".format(tag))

        for file in files[:len(files) // 2]:
            print(src_path + "/{0}/".format(tag) + file)
            print(dest_path + "/{0}/".format(tag) + file)
            shutil.move(src_path + "/{0}/".format(tag) + file, dest_path + "/{0}/".format(tag) + file)


# Split the validation data equally to test data
# for dataset_name in ["ADM", "BigGAN", "glide", "Midjourney",
#                      "Stable_diffusion_4", "Stable_diffusion_5", "VQDM", "wukong"]:
#     split_val_img_to_test("Dataset/{0}".format(dataset_name))


def build_data_generator(image_size, batch_size, dataset_spec="*"):
    train_path = tf.io.gfile.glob("Dataset/{0}/train".format(dataset_spec))[0]
    vali_path = tf.io.gfile.glob("Dataset/{0}/val".format(dataset_spec))[0]
    test_path = tf.io.gfile.glob("Dataset/{0}/test".format(dataset_spec))[0]

    def check_img_valid(_path):
        # path = "./dataset/PetImages/dog/"
        dir_ = os.listdir(_path)

        for image in dir_:
            file = os.path.join(_path, image)
            if not imghdr.what(file):
                print("Image removed: " + file)
                os.remove(file)

        print("Path scanned: " + _path)

    for path in ["ai", "nature"]:
        check_img_valid(train_path + "\\" + path)
        check_img_valid(vali_path + "\\" + path)
        check_img_valid(test_path + "\\" + path)

    train_ds, train_len = make_dataset(
        train_path,
        image_size=image_size,
        batch_size=batch_size
    )
    # class_names = train_ds.class_names
    train_steps_num = math.ceil(train_len / batch_size)

    vali_ds, vali_len = make_dataset(
        vali_path,
        image_size=image_size,
        batch_size=batch_size
    )

    vali_steps_num = math.ceil(vali_len / batch_size)

    test_ds, test_len = make_dataset(
        test_path,
        image_size=image_size,
        batch_size=batch_size
    )

    test_steps_num = math.ceil(test_len / batch_size)

    return train_ds, vali_ds, test_ds, train_steps_num, vali_steps_num, test_steps_num


def make_dataset(path, batch_size, image_size):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [image_size, image_size])
        image = data_augmentation(image)
        return image

    def configure_for_performance(_ds):
        _ds = _ds.shuffle(buffer_size=1000)
        _ds = _ds.batch(batch_size)
        _ds = _ds.repeat()
        _ds = _ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return _ds

    classes = os.listdir(path)
    filenames = tf.io.gfile.glob(path + '/*/*')
    random.shuffle(filenames)
    labels = [classes.index(name.split('\\')[-2]) for name in filenames]
    # print(labels)

    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(tf.one_hot(labels, depth=2))
    ds = tf.data.Dataset.zip((images_ds, labels_ds))
    ds = configure_for_performance(ds)

    return ds, len(filenames)


# build_data_generator(128, 32, "Test")

