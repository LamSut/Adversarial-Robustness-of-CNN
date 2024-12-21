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
    def __init__(self, nb_filters=64, input_shape=(32, 32, 3)):
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

    cifar10_test = dataset["test"]
    cifar10_test = cifar10_test.map(convert_types).batch(128)

    return cifar10_test


def main(_):
    # Load training and test data
    data = ld_cifar10()
    # Create a sample input (assuming your data has channels at the end)
    sample_input = np.random.rand(128, 32, 32, 3)  # Shape (batch_size, height, width, channels)
    # Call the model on the sample input
    model = CNN()
    model(sample_input)  # This triggers variable creation
    
    if FLAGS.adv_train:
        model.load_weights("model_weight_adv.h5")
        train_type="Adversarial"
        print("Loading model trained with adversarial examples...")
    else:
        model.load_weights("model_weight.h5")
        train_type="Standard"
        print("Loading model trained with clean examples...")
    data=data.skip(7)
    for x, y in data:
        
        y_pred = model(x)
        
        x_fgm = fast_gradient_method(model, x, FLAGS.eps, np.inf)
        y_pred_fgm = model(x_fgm)

        x_pgd = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
        y_pred_pgd = model(x_pgd)
        
        print("true label: ")
        print(y[0])
        print("probabilities: ")
        print(y_pred[0])
        print("predict label: ")
        print(tf.argmax(y_pred, axis=1)[0])
        
        true_label = tf.squeeze(y[0]).numpy()  # Extract label as a scalar (int)
        predicted_label = tf.squeeze(tf.argmax(y_pred, axis=1))[0].numpy()
        
        predicted_label_fgm = tf.squeeze(tf.argmax(y_pred_fgm, axis=1))[0].numpy()
        predicted_label_pgd = tf.squeeze(tf.argmax(y_pred_pgd, axis=1))[0].numpy()
        

        fig, axes = plt.subplots(2, 3, figsize=(10, 4), gridspec_kw={'height_ratios': [6, 1]})  # Adjust row heights
        class_names = tfds.builder("cifar10").info.features["label"].names

        fig.suptitle("Adversarial Robustness Of " + train_type + " Trainning Model", fontsize=18)

        # Visualize images and set titles in row 1
        axes[0][0].imshow(x[0])
        axes[0][0].set_title("Clean input")
        axes[0][0].axis("off")

        axes[0][1].imshow(x_fgm[0])
        axes[0][1].set_title("FGM input")
        axes[0][1].axis("off")

        axes[0][2].imshow(x_pgd[0])
        axes[0][2].set_title("PGD input")
        axes[0][2].axis("off")

        axes[1][0].set_title("Prediction: " + class_names[predicted_label])
        axes[1][0].axis("off")

        axes[1][1].set_title("Prediction: " + class_names[predicted_label_fgm])
        axes[1][1].axis("off")

        axes[1][2].set_title("Prediction: " + class_names[predicted_label_pgd])
        axes[1][2].axis("off")

        plt.figtext(0.5, 0.05, "True Label: " + class_names[true_label], ha="center", fontsize=16)

        plt.tight_layout()
        plt.show()
        

if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 1, "Number of epochs.")
    flags.DEFINE_float("eps", 0.05, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )
    app.run(main)
