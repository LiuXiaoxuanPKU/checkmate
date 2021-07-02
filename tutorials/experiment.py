from checkmate.tf2 import get_keras_model
import time

import logging
import numpy as np
import tensorflow as tf
from checkmate.tf2 import get_keras_model
from tqdm import tqdm
from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn as chen_solver
# logging.basicConfig(level=logging.DEBUG)

DEBUG_SPEED = True

# load cifar10 dataset
batch_size = 1024
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, y_train = x_train.astype(np.float32), y_train.astype(np.float32)
x_test, y_test = x_test.astype(np.float32), y_test.astype(np.float32)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# load TensorFlow model from Keras applications along with loss function and optimizer
model = get_keras_model("VGG16", input_shape=x_train[0].shape, num_classes=10)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss)

from checkmate.tf2.wrapper import compile_tf2
element_spec = train_ds.__iter__().__next__()
train_iteration = compile_tf2(
    model,
    loss=loss,
    optimizer=optimizer,
    input_spec=element_spec[0],  # retrieve first element of dataset
    label_spec=element_spec[1],
    scheduler=chen_solver
)

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

for epoch in range(10):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    train_step_ct = 0
    end = None
    train_ips_list = []
    with tqdm(total=x_train.shape[0]) as pbar:
        for images, labels in train_ds:
            predictions, loss_value = train_iteration(images, labels)
            train_step_ct += 1
            if DEBUG_SPEED and end is not None:
                batch_total_time = time.time() - end
                train_ips = batch_size / batch_total_time
                train_ips_list.append(train_ips)
            
            end = time.time()

            if train_step_ct >= 4:
                train_ips = np.median(train_ips_list)
                res = "BatchSize: %d\tIPS: %.2f\tCost: %.2f ms" % (batch_size, train_ips, 1000.0 / train_ips)
                print(res)
                exit(0)

            # if not DEBUG_SPEED:
            train_loss(loss_value)
            train_accuracy(labels, predictions)
            pbar.update(images.shape[0])
            pbar.set_description('Train epoch {}; loss={:0.4f}, acc={:0.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))


    with tqdm(total=x_test.shape[0]) as pbar:
        for images, labels in test_ds:
            predictions = model(images)
            test_loss_value = loss(labels, predictions)
            test_loss(test_loss_value)
            test_accuracy(labels, predictions)
            pbar.update(images.shape[0])
            pbar.set_description('Valid epoch {}, loss={:0.4f}, acc={:0.4f}'.format(epoch + 1, test_loss.result(), test_accuracy.result()))
