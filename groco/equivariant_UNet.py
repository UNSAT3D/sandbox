from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import GlobalMaxPooling3D
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dense, Input,
                                     Lambda, MaxPooling2D, UpSampling2D,
                                     UpSampling3D)

from groco.groups.wallpaper_groups import P4M, P4M_action
from groco.layers import GroupConv2D, GroupConv2DTranspose, GroupMaxPooling2D


class GroupUpSampling2D(UpSampling3D):
    def __init__(self, size, **kwargs):
        if isinstance(size, int):
            size = (size, size, 1)
        elif isinstance(size, tuple) and len(size) == 2:
            size = (size[0], size[1], 1)
        super().__init__(size=size, **kwargs)


def build_model(
    conv_layer,
    conv_layer_transpose,
    pool_layer,
    upsample_layer,
    invariant_layer,
    filters,
    input_shape=(28, 28, 1),
    group=False,
    name="model",
    num_layers=None,
):
    layers = [Input(shape=input_shape)]
    padding = "same"
    if group and padding == "same":
        padding = "same_equiv"
    for f in filters:
        layers.append(
            conv_layer(kernel_size=3, activation="relu", filters=f, padding=padding)
        )
        layers.append(pool_layer(strides=2, pool_size=2))

    for f in reversed(filters):
        layers.append(
            conv_layer_transpose(
                kernel_size=3, activation="relu", filters=f, padding=padding
            )
        )
        # todo: make this work with signal on group
        layers.append(upsample_layer(size=2))

    if group:
        group_axis = 3
        group_pool = Lambda(
            lambda x: tf.reduce_mean(x, axis=group_axis), name="group_pool"
        )
        layers.append(group_pool)

    layers.append(Dense(1, activation="sigmoid"))

    if num_layers:
        layers = layers[:num_layers]
    model = Sequential(layers, name=name)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"]
    )

    model.summary()

    return model


def main():
    # load mnist example data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)

    print(x_train.shape)

    # expand images with zeroes to be given size
    sz = 32
    x_train = tf.image.resize_with_crop_or_pad(x_train, sz, sz)
    x_test = tf.image.resize_with_crop_or_pad(x_test, sz, sz)
    print(x_train.shape)

    input_shape = (sz, sz, 1)
    num_classes = 10

    # prepare group layers
    tmp = False
    P4MConv2D = partial(GroupConv2D, group="P4M", allow_non_equivariance=tmp)
    P4MConv2DTranspose = partial(
        GroupConv2DTranspose, group="P4M", allow_non_equivariance=tmp
    )
    P4MMaxPooling2D = partial(
        GroupMaxPooling2D, group="P4M", allow_non_equivariance=tmp
    )
    P4Minvariant = GlobalMaxPooling3D
    P4MUpSampling2D = partial(GroupUpSampling2D)
    group_layers = (P4MConv2D, P4MConv2DTranspose, P4MMaxPooling2D, P4MUpSampling2D)

    regular_layers = (Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D)

    # define models
    features = [8, 16, 32]
    model = build_model(
        *regular_layers, Dense, features, input_shape=input_shape, name="regular"
    )
    model_group = build_model(
        *group_layers,
        Dense,
        features,
        group=True,
        input_shape=input_shape,
        name="equivariant"
    )

    test_image = x_train[2:3]
    check_equiv(test_image, model_group, P4M_action)
    plot_transformations(test_image, model, model_group, P4M_action)


def transform_image(image, group_action):
    group_order = 8
    transformed_images = group_action(image, spatial_axes=(1, 2), new_group_axis=0)
    transformed_images = tf.reshape(
        transformed_images, (group_order,) + image.shape[1:]
    )
    return transformed_images


def plot_transformations(image, model_regular, model_group, group_action):
    """
    Plot the transformations of an image under two models.

    Args:
        image: Image to transform.
        model_regular: First model.
        model_group: Second model.
    """
    print("image shape: ", image.shape)
    transformed_images = transform_image(image, group_action)
    print("new shape: ", transformed_images.shape)

    output_regular = model_regular(transformed_images)
    output_group = model_group(transformed_images)

    # for convenience, to be able to test with intermediate layers, take first feature
    output_regular = tf.expand_dims(tf.gather(output_regular, 0, axis=-1), -1)
    output_group = tf.expand_dims(tf.gather(output_group, 0, axis=-1), -1)
    print("regular output shape", output_regular.shape)
    print("group output shape", output_group.shape)

    # now plot all 8 transformed input images and their outputs
    fig, axs = plt.subplots(3, 8)
    # keep axes on but ticks and numbers off, just labels
    for ax in axs.flatten():
        ax.axis("on")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    fig.set_size_inches(9 * 3, 3 * 3)
    for i in range(8):
        for imtype, im in enumerate([transformed_images, output_regular, output_group]):
            if (
                imtype == 2 and output_regular.shape.rank != output_group.shape.rank
            ):  # the group output
                print("doing something weird")
                # Note: Up to the final layer which pools over the group, the equivariant model
                # has for each input image 8 outputs, one for each group element.
                # The group acts not only on the grid but also on this group element.
                # Taking  the ith in the batch dimension means the input image is already
                # transformed by group element i before going through the network, even though
                # it's given no explicit group argument.
                # The i index in the group axis compensates for this somehow
                # TODO: not really clear how this works
                im = im[i, :, :, i, :]
            else:
                im = im[i]
            vmin, vmax = np.min(im), np.max(im)
            axs[imtype, i].imshow(im, vmin=vmin, vmax=vmax)

    # set titles for rows, displaying to the left side
    for i, title in enumerate(["Input", model_regular.name, model_group.name]):
        axs[i, 0].set_ylabel(title, fontsize=20, labelpad=30)

    plt.tight_layout()
    # save to file
    plt.savefig("equivariant_UNet.png")
    plt.show()


def check_equiv(image, model, group_action):
    model_image = model(image)
    transform_model_image = transform_image(model_image, group_action)
    print("transformed model image shape: ", transform_model_image.shape)

    transformed_images = transform_image(image, group_action)
    model_transformed_images = model(transformed_images)
    print("model transformed images shape: ", model_transformed_images.shape)

    diff = tf.reduce_sum(tf.abs(model_transformed_images - transform_model_image))
    reldiff = diff / tf.reduce_sum(tf.abs(transform_model_image))
    print(
        "difference between transformed model image and model of transformed images: ",
        diff,
    )
    print("relative difference: ", reldiff)


if __name__ == "__main__":
    main()
