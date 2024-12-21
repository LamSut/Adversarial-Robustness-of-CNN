import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from easydict import EasyDict
from keras import Model, metrics
from keras.layers import AveragePooling2D, Conv2D
import matplotlib.pyplot as plt

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

tf.config.run_functions_eagerly(True)

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

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 127.5
        image -= 1.0
        return image, label

    dataset, info = tfds.load("cifar10", with_info=True, as_supervised=True)

    def augment_mirror(x):
        return tf.image.random_flip_left_right(x)

    def augment_shift(x, w=4):
        y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode="REFLECT")
        return tf.image.random_crop(y, tf.shape(x))

    cifar10_train = dataset["train"]
    # Augmentation helps a lot in CIFAR10
    cifar10_train = cifar10_train.map(
        lambda x, y: (augment_mirror(augment_shift(x)), y)
    )
    cifar10_train = cifar10_train.map(convert_types).shuffle(10000).batch(128)
    return cifar10_train

def main(_):
    # Load training and test data
    data = ld_cifar10()
    model = CNN()
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    
    # Metrics to track the different accuracies.
    train_accuracy = metrics.SparseCategoricalAccuracy(name="train_accuracy") 
    train_loss = metrics.Mean(name="train_loss")
    train_accuracies = [] # Initialize an empty list
    train_losses = []  # Initialize an empty list

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_accuracy.update_state(y, predictions)  # Update accuracy metric
            train_loss(loss)

    # Train model with adversarial training
    for epoch in range(FLAGS.nb_epochs):
        # keras like display of progress
        progress_bar_train = tf.keras.utils.Progbar(50000)
        for (x, y) in data:
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for adversarial training
                x = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
            train_step(x, y)
            progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result())])
            train_accuracies.append(train_accuracy.result())
            train_losses.append(train_loss.result().numpy())
    
    if FLAGS.adv_train:
        model.save_weights("model_weight_adv.h5")
        print("CNN model with adversarial crafts saved successfully!")
    else:
        model.save_weights("model_weight.h5")
        print("CNN model saved successfully!")

    np.save("train_losses.npy", train_losses)  # Save losses as a NumPy array
    train_losses = np.load("train_losses.npy")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figure size as needed

    ax1.plot(train_losses, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Over Epochs")
    ax1.grid(True)

    ax2.plot(train_accuracies, label="Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy Over Epochs")
    ax2.grid(True)
    
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 10, "Number of epochs.")
    flags.DEFINE_float("eps", 0.05, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", True, "Use adversarial training (on PGD adversarial examples)."
    )
    app.run(main)
