from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
import os

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, height, width, channels]
    input_layer = tf.reshape(features["x"], [-1, 240, 800, 3])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 240, 800, 3]
    # Output Tensor Shape: [batch_size, 120, 400, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        strides=2,
        activation=tf.nn.relu)

    # Convolutional Layer #2
    # Computes 32 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 120, 400, 32]
    # Output Tensor Shape: [batch_size, 60, 200, 32]
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        strides=2,
        activation=tf.nn.relu)

    # Convolutional Layer #3
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 60, 200, 32]
    # Output Tensor Shape: [batch_size, 30, 100, 64]
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        strides=2,
        activation=tf.nn.relu)

    # Convolutional Layer #4
    # Computes 128 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 30, 100, 64]
    # Output Tensor Shape: [batch_size, 15, 50, 128]
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=[5, 5],
        padding='same',
        strides=2,
        activation=tf.nn.relu)

    # Convolutional Layer #5
    # Computes 128 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 15, 50, 128]
    # Output Tensor Shape: [batch_size, 8, 25, 128]
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=128,
        kernel_size=[5, 5],
        padding='same',
        strides=2,
        activation=tf.nn.relu)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 8, 25, 128]
    # Output Tensor Shape: [batch_size, 8 * 25 * 128]
    pool1_flat = tf.reshape(conv5, [-1, 8 * 25 * 128])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 8 * 25 * 128]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(
        inputs=pool1_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout, units=1024, activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 24]
    logits = tf.layers.dense(inputs=dropout2, units=24)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=24)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def encoder():
    enc = dict()
    counter = 0
    img_dir = os.listdir("./ARGOS_data/training/photos")

    for fileDir in img_dir:
        if fileDir == "DBinfo.txt" or fileDir == ".DS_Store":
            continue
        enc[fileDir] = counter
        counter += 1
    return enc


def get_train_data(encoder):
    counter = 0
    list_of_imgs = []
    list_of_labels = []
    img_dir_path = "./ARGOS_data/training/photos"
    img_dir = os.listdir(img_dir_path)

    for fileDir in img_dir:
        if fileDir == "DBinfo.txt" or fileDir == ".DS_Store":
            continue
        for img in os.listdir(img_dir_path + "/" + fileDir):
           # print("Photo " + str(counter) + " in directory " + fileDir + " loading photo " + img)
           # if(counter == 10):
            #    break
            img = Image.open(img_dir_path + "/" + fileDir + "/" + img)
            img.load()

            data = np.asarray(img, dtype=np.float32)
            data = data.flatten()
            list_of_imgs.append(data)
            list_of_labels.append(encoder[fileDir])
            counter += 1

       # break

    return np.array(list_of_imgs), np.array(list_of_labels)


def get_eval_data(encoder):
    counter = 0
    truth = dict()
    classes = []
    list_of_imgs = []
    list_of_labels = []

    img_dir_path = "./ARGOS_data/test/photos"

    for fileDir in os.listdir("ARGOS_data/training/photos"):
        if fileDir == "DBinfo.txt" or fileDir == ".DS_Store":
            continue
        classes.append(fileDir)
    with open("ARGOS_data/test/photos/ground_truth.txt", encoding="utf-8") as f:
        for line in f:
            splittedLine = line.strip().replace(":", "").split(";")
            img = splittedLine[0]
            label = splittedLine[1]
            if(label in classes):
                truth[img] = label

    for img in os.listdir(img_dir_path):
        if(img == "ground_truth.txt" or img == "Thumbs.db" or img == ".DS_Store"):
            continue
        if(img not in truth.keys()):
            continue
       # if(counter == 10):
        #    break
        image = Image.open(img_dir_path + "/" + img)
        image.load()
        data = np.asarray(image, dtype=np.float32)
        list_of_imgs.append(data)
        list_of_labels.append(encoder[truth[img]])
        counter += 1

    return np.array(list_of_imgs), np.array(list_of_labels)


def main(unused_argv):
    # Load training and eval data
    labels = encoder()
    train_data, train_labels = get_train_data(labels)

    print("INFO:tensorflow:Finished loading data")

    # Create the Estimator
    argos_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="D:\\argos_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=31,
        num_epochs=50,
        shuffle=True)
    argos_classifier.train(
        input_fn=train_input_fn,
        steps=7700,
        hooks=[logging_hook])

    eval_data, eval_labels = get_eval_data(labels)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = argos_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
