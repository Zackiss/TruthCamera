import sys

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from data_loader_even_faster import build_data_generator
from helper_func import DropPath, window_partition, window_reverse
from patch_modify import PatchExtract, PatchEmbedding, PatchMerging

import keras
from keras import layers
from keras import losses
from keras import metrics
from keras import activations

import warnings


class WindowAttention(layers.Layer):
    def __init__(
            self, dim, _window_size, _num_heads, _qkv_bias=True, _dropout_rate=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.relative_position_index = None
        self.relative_position_bias_table = None
        self.dim = dim
        self.window_size = _window_size
        self.num_heads = _num_heads
        self.scale = (dim // _num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=_qkv_bias)
        self.dropout = layers.Dropout(_dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, _input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
                2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            name="magic_weight",
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, _inputs, mask=None):
        x = _inputs
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                    tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                    + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv


class SwinTransformer(layers.Layer):
    def __init__(
            self,
            dim,
            num_patch,
            _num_heads,
            _window_size=7,
            _shift_size=0,
            _num_mlp=1024,
            _qkv_bias=True,
            _dropout_rate=0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.attn_mask = None
        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = _num_heads  # number of attention heads
        self.window_size = _window_size  # size of window
        self.shift_size = _shift_size  # size of window shift
        self.num_mlp = _num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            _window_size=(self.window_size, self.window_size),
            _num_heads=_num_heads,
            _qkv_bias=_qkv_bias,
            _dropout_rate=_dropout_rate,
        )
        self.drop_path = DropPath(_dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(_num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(_dropout_rate),
                layers.Dense(dim),
                layers.Dropout(_dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, _input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x


image_size = 128
input_shape = (image_size, image_size, 3)
patch_size = (2, 2)  # 4-by-4 sized patches
dropout_rate = 0.03  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 256  # MLP layer size
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = image_size  # Initial image size

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

# learning_rate = 1e-3
learning_rate = 1e-5
batch_size = 32
num_epochs = 40
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1
num_classes = 2
k_accuracy = 5


model_input = layers.Input(input_shape)
x = layers.RandomCrop(image_dimension, image_dimension)(model_input)
x = layers.RandomFlip("horizontal")(x)
x = PatchExtract(patch_size)(x)
x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
x = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    _num_heads=num_heads,
    _window_size=window_size,
    _shift_size=0,
    _num_mlp=num_mlp,
    _qkv_bias=qkv_bias,
    _dropout_rate=dropout_rate,
)(x)
x = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    _num_heads=num_heads,
    _window_size=window_size,
    _shift_size=shift_size,
    _num_mlp=num_mlp,
    _qkv_bias=qkv_bias,
    _dropout_rate=dropout_rate,
)(x)
x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
x = layers.GlobalAveragePooling1D()(x)
output = layers.Dense(num_classes, activation="softmax")(x)


if __name__ == "__main__":

    AUTOTUNE = tf.data.AUTOTUNE
    warnings.filterwarnings('ignore')

    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    if len(tf.config.list_physical_devices("GPU")) == 0:
        sys.exit(0)

    # Main loop
    for dataset_name in ["ADM", "BigGAN", "glide", "Midjourney",
                         "Stable_diffusion_4", "Stable_diffusion_5", "VQDM", "wukong"]:

        print("Working on dataset {0}".format(dataset_name))

        train_data_gen, vali_data_gen, test_data_gen, train_steps, vali_steps, test_steps = build_data_generator(
            image_size=image_size,
            batch_size=batch_size,
            dataset_spec=dataset_name
        )

        model = keras.Model(model_input, output)
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            optimizer=tfa.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            ),
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                # keras.metrics.TopKCategoricalAccuracy(k_accuracy, name="top_{0}_accuracy".format(k_accuracy)),
            ],
            run_eagerly=True
        )

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        history = model.fit_generator(
            train_data_gen,
            steps_per_epoch=train_steps,
            validation_steps=vali_steps,
            validation_data=vali_data_gen,
            epochs=num_epochs,
            callbacks=[callback],
            verbose=1
        )

        model.save('./checkpoints/{0}'.format(dataset_name))
        loaded_model = keras.models.load_model('./checkpoints/{0}'.format(dataset_name))
        loss, accuracy = loaded_model.evaluate_generator(
            test_data_gen,
            steps=test_steps
        )

        print(f"Test loss: {round(loss, 2)}")
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        # print(f"Test top K accuracy: {round(top_k_accuracy * 100, 2)}%")

        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train and Validation Losses Over Epochs", fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()
