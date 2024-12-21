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
    sample_input = np.random.rand(128, 32, 32 , 3)  # Shape (batch_size, height, width, channels)
    # Call the model on the sample input
    model = CNN()
    model(sample_input)  # This triggers variable creation
    if FLAGS.adv_train:
        model.load_weights("model_weight_adv.h5")
        print("Loading model trained with adversarial examples...")
    else:
        model.load_weights("model_weight.h5")
        print("Loading model trained with clean examples...")
    
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    
    # Metrics to track the different accuracies.
    train_loss = tf.metrics.Mean(name="train_loss")
    test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
    test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
    test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()
    
    # Metrics to track the different accuracies on graph plotted.
    test_acc_clean_graph = metrics.SparseCategoricalAccuracy(name='test_acc_clean')
    test_acc_fgsm_graph = metrics.SparseCategoricalAccuracy(name='test_acc_fgsm')
    test_acc_pgd_graph = metrics.SparseCategoricalAccuracy(name='test_acc_pgd')
    
    # Initialize empty lists to store history:
    test_acc_clean_history = []
    test_acc_fgsm_history = []
    test_acc_pgd_history = []

    # Evaluate on clean and adversarial data
    progress_bar_test = tf.keras.utils.Progbar(10000)
    for x, y in data:
        y_pred = model(x)
        test_acc_clean(y, y_pred)
        test_acc_clean_graph.update_state(y, y_pred)
        
        x_fgm = fast_gradient_method(model, x, FLAGS.eps, np.inf)
        y_pred_fgm = model(x_fgm)
        test_acc_fgsm(y, y_pred_fgm)
        test_acc_fgsm_graph.update_state(y, y_pred_fgm)

        x_pgd = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
        y_pred_pgd = model(x_pgd)
        test_acc_pgd(y, y_pred_pgd)
        test_acc_pgd_graph.update_state(y, y_pred_pgd)
        
        progress_bar_test.add(x.shape[0])
        
        # Append metrics to history lists after each test batch
        test_acc_clean_history.append(test_acc_clean_graph.result())
        test_acc_fgsm_history.append(test_acc_fgsm_graph.result())
        test_acc_pgd_history.append(test_acc_pgd_graph.result())

        # Reset metrics for the next test batch
        test_acc_clean_graph.reset_states()
        test_acc_fgsm_graph.reset_states()
        test_acc_pgd_graph.reset_states()

    print(
        "test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100)
    )
    print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            test_acc_fgsm.result() * 100
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            test_acc_pgd.result() * 100
        )
    )

    # Example plot after each epoch:
    plt.figure(figsize=(9, 6))

    plt.subplot(1, 1, 1)
    plt.plot(test_acc_clean_history, label='Clean Acc')
    plt.plot(test_acc_fgsm_history, label='FGM Acc')
    plt.plot(test_acc_pgd_history, label='PGD Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy History (Clean, FGM, PGD)')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")
    flags.DEFINE_float("eps", 0.05, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )
    app.run(main)
