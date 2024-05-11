import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from easydict import EasyDict
from keras import Model
from keras.layers import AveragePooling2D, Conv2D

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

class CNN(Model):
    def __init__(self, nb_filters=64):
        super(CNN, self).__init__()
        img_size = 32
        log_resolution = int(round(math.log(img_size) / math.log(2)))
        conv_args = dict(activation=tf.nn.leaky_relu, kernel_size=3, padding="same")
        self.layers_obj = []
        for scale in range(log_resolution - 2):
            conv1 = Conv2D(nb_filters << scale, **conv_args)
            conv2 = Conv2D(nb_filters << (scale + 1), **conv_args)
            pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
            self.layers_obj.append(conv1)
            self.layers_obj.append(conv2)
            self.layers_obj.append(pool)
        conv = Conv2D(10, **conv_args)
        self.layers_obj.append(conv)

    def call(self, x):
        for layer in self.layers_obj:
            x = layer(x)
        return tf.reduce_mean(x, [1, 2])


def ld_cifar10():
    """Load training and test data."""

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 127.5
        image -= 1.0
        return image, label

    dataset, info = tfds.load("cifar10", with_info=True, as_supervised=True)

    cifar10_test =  dataset["test"]
    cifar10_test = cifar10_test.map(convert_types).batch(128)

    return cifar10_test


def main(_):
    # Load training and test data
    data = ld_cifar10()
    model = CNN()

    for x, y in data:

        x_fgm = fast_gradient_method(model, x, FLAGS.eps, np.inf)
        x_pgd = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
            
        # Adjust rows and columns based on the number of images (here: 1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # Customize figsize as needed

        # Visualize clean image
        axes[0].imshow(x[0])
        axes[0].set_title("Clean Image")
        axes[0].axis("off")

        # Visualize FGM adversarial image
        axes[1].imshow(x_fgm[0])
        axes[1].set_title("FGM Adversarial Image")
        axes[1].axis("off")

        # Visualize PGD adversarial image
        axes[2].imshow(x_pgd[0])  # Assuming x_pgd[0] represents the PGD adversarial image
        axes[2].set_title("PGD Adversarial Image")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
        

if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 1, "Number of epochs.")
    flags.DEFINE_float("eps", 0.05, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )
    app.run(main)
