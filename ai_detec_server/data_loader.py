from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import matplotlib.pyplot as plt
import numpy as np
import os


class MergedGenerators(Sequence):

    def __init__(self, batch_size, generators=[], sub_batch_size=[]):
        self.generators = generators
        self.sub_batch_size = sub_batch_size
        self.batch_size = batch_size

    def __len__(self):
        return int(
            sum([(len(self.generators[idx]) * self.sub_batch_size[idx])
                 for idx in range(len(self.sub_batch_size))]) /
            self.batch_size)

    def __getitem__(self, index):
        """Getting items from the generators and packing them"""

        X_batch = []
        Y_batch = []
        for generator in self.generators:
                x1, y1 = generator[index % len(generator)]
                X_batch = [*X_batch, *x1]
                Y_batch = [*Y_batch, *y1]

        return np.array(X_batch), np.array(Y_batch)

    def next(self):
        X_batch = []
        Y_batch = []
        for generator in self.generators:
            x1, y1 = generator.next()
            X_batch = [*X_batch, *x1]
            Y_batch = [*Y_batch, *y1]

        return np.array(X_batch), np.array(Y_batch)


def build_data_generator(image_size: int, dirs: list[str], batch_size=32):
    train_batches = []
    test_batches = []

    train_img_nums = []
    test_img_nums = []

    train_generators = []
    test_generators = []

    for i in range(len(dirs)):
        locals()["train_dir_" + str(i)] = os.path.join(dirs[i], 'train')
        locals()["test_dir_" + str(i)] = os.path.join(dirs[i], 'val')

        locals()["train_data_gen_" + str(i)] = ImageDataGenerator(
            rescale=1. / 255,
            # shear_range=0.2,
            # zoom_range=0.2,
            # rotation_range=5.,
            # horizontal_flip=True,
        )
        locals()["test_data_gen_" + str(i)] = ImageDataGenerator(
            rescale=1. / 255,
        )

        print("Querying on folder path: " + locals()["train_dir_" + str(i)])
        locals()["train_gen_" + str(i)] = locals()["train_data_gen_" + str(i)].flow_from_directory(
            locals()["train_dir_" + str(i)],
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='binary',
            seed=42,
        )
        train_img_nums.append(sum([len(files) for r, d, files in os.walk(locals()["train_dir_" + str(i)])]))

        print("Querying on folder path: " + locals()["test_dir_" + str(i)])
        locals()["test_gen_" + str(i)] = locals()["test_data_gen_" + str(i)].flow_from_directory(
            locals()["test_dir_" + str(i)],
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='binary',
            seed=42,
        )
        test_img_nums.append(sum([len(files) for r, d, files in os.walk(locals()["test_dir_" + str(i)])]))

        train_generators.append(locals()["train_gen_" + str(i)])
        test_generators.append(locals()["test_gen_" + str(i)])

    print("Data generator build finish :)")

    train_img_sum = sum(train_img_nums)
    train_generator = MergedGenerators(
        batch_size=batch_size,
        generators=train_generators,
        sub_batch_size=[
            int((train_img_nums[i] * batch_size) / train_img_sum)
            for i in range(len(train_img_nums))
        ]
    )

    test_img_sum = sum(test_img_nums)
    test_generator = MergedGenerators(
        batch_size=batch_size,
        generators=test_generators,
        sub_batch_size=[
            int((test_img_nums[i] * batch_size) / test_img_sum)
            for i in range(len(test_img_nums))
        ]
    )

    return train_generator, test_generator


def test_datagen(batch_size=32):
    train, test = build_data_generator(
        image_size=1024,
        dirs=[
            "./Dataset/Midjourney",
        ],
        batch_size=batch_size
    )

    print("Data generator length (Batch count):", len(train))

    for i in range(len(train)):
        image_batch = train[i][0]
        label_batch = train[i][1]
        print("Images: ", image_batch.shape)
        plt.figure(figsize=(10, 10))
        for i in range(image_batch.shape[0]):
            plt.subplot(1, batch_size, i + 1)
            plt.imshow(image_batch[i], interpolation='nearest')
            plt.axis('off')
            plt.tight_layout()
        for i in range(label_batch.shape[0]):
            print(label_batch[i])
        plt.show()
        break

    # normalization_layer = rescaling.Rescaling(1. / 255)

    # norm_train_data_gen = train_data_gen.map(lambda a, y: (normalization_layer(a), y))
    # norm_vali_data_gen = vali_data_gen.map(lambda b, y: (normalization_layer(b), y))
    # train_image_batch,  train_labels_batch = next(iter(norm_train_data_gen))
    # vali_image_batch, vali_labels_batch = next(iter(norm_vali_data_gen))
