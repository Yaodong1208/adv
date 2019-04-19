from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
import stadv
import idx2numpy

mnist_data_dir = '.'
mnist_images = idx2numpy.convert_from_file(os.path.join(mnist_data_dir, 't10k-images-idx3-ubyte'))
mnist_labels = idx2numpy.convert_from_file(os.path.join(mnist_data_dir, 't10k-labels-idx1-ubyte'))
mnist_images = np.expand_dims(mnist_images, -1)

print("Shape of images:", mnist_images.shape)
print("Range of values: from {} to {}".format(np.min(mnist_images), np.max(mnist_images)))
print("Shape of labels:", mnist_labels.shape)
print("Range of values: from {} to {}".format(np.min(mnist_labels), np.max(mnist_labels)))

# definition of the inputs to the network
images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='images')
flows = tf.placeholder(tf.float32, [None, 2, 28, 28], name='flows')
targets = tf.placeholder(tf.int64, shape=[None], name='targets')
tau = tf.placeholder_with_default(
    tf.constant(0., dtype=tf.float32),
    shape=[], name='tau'
)

# flow-based spatial transformation layer
perturbed_images = stadv.layers.flow_st(images, flows, 'NHWC')

# definition of the CNN in itself
conv1 = tf.layers.conv2d(
    inputs=perturbed_images,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu
)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu
)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
logits = tf.layers.dense(inputs=pool2_flat, units=10)

# definition of the losses pertinent to our study
L_adv = stadv.losses.adv_loss(logits, targets)
L_flow = stadv.losses.flow_loss(flows, padding_mode='CONSTANT')
L_final = L_adv + tau * L_flow
grad_op = tf.gradients(L_final, flows, name='loss_gradient')[0]

init = tf.global_variables_initializer()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)
saver.restore(sess, os.path.join('saved_models', 'simple_mnist'))

i_random_image = np.random.randint(0, len(mnist_images))
test_image = mnist_images[i_random_image]
test_label = mnist_labels[i_random_image]
random_target = 7

print("Considering image #", i_random_image, "from the test set of MNIST")
print("Ground truth label:", test_label)
print("Randomly selected target label:", random_target)

# reshape so as to have a first dimension (batch size) of 1
test_image = np.expand_dims(test_image, 0)
test_label = np.expand_dims(test_label, 0)
random_target = np.expand_dims(random_target, 0)

# with no flow the flow_st is the identity
null_flows = np.zeros((1, 2, 28, 28))

pred_label = np.argmax(sess.run(
    [logits],
    feed_dict={images: test_image, flows: null_flows}
))

print("Predicted label (no perturbation):", pred_label)


results = stadv.optimization.lbfgs(
    L_final,
    flows,
    # random initial guess for the flow
    flows_x0=np.random.random_sample((1, 2, 28, 28)),
    feed_dict={images: test_image, targets: random_target, tau: 0.05},
    grad_op=grad_op,
    sess=sess
)

print("Final loss:", results['loss'])
print("Optimization info:", results['info'])

test_logits_perturbed, test_image_perturbed = sess.run(
    [logits, perturbed_images],
    feed_dict={images: test_image, flows: results['flows']}
)
pred_label_perturbed = np.argmax(test_logits_perturbed)

print("Predicted label after perturbation:", pred_label_perturbed)






