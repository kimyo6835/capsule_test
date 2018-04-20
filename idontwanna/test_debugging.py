from __future__ import division, print_function, unicode_literals

import matplotlib.pyplot as plt
from tada import wires_functions
from tada import capsule_functions
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import numpy as np

# ----------- Generate images of wires ----------- #

x_train, y_train = wires_functions.gen_wires(n_data=1000, length=20, n_wires=2, imsize=32)
x_test, y_test = wires_functions.gen_wires(n_data=1000, length=20, n_wires=2, imsize=32)

# ----------- Check one of images (train) ----------- #
check_idx = 15  # random number

# Color image
plt.figure()
plt.imshow(x_train[check_idx, :, :, 0])
plt.colorbar()
plt.show()

# Grey image
plt.figure()
plt.imshow(x_train[check_idx, :, :, 0], cmap="gray")
plt.show()

# ----------- Set networks  ----------- #
epsilon = 1e-5
batch_size = 4
imsize = 32

x = tf.placeholder(shape=[None, imsize, imsize, 1], dtype=tf.float32, name="X")

# ------- ReLU layer -------- #
# kernel size = [5, 5]
# out_channels = 16
# stride = 1 with padding ('same')
ReLU_conv1 = capsule_functions.conv2d(x, 5, 16, 1, 'SAME', 'ReLU_conv1', True, tf.nn.relu)

# ------- PrimaryCaps layer -------- #
# kernel size = [5, 5]
# out_capsules = 16
# stride = 1 with padding ('same')
# pose shape = [4, 4]
PrimaryCaps = capsule_functions.primary_caps(ReLU_conv1, 5, 16, 1, 'SAME', (4, 4), "PrimaryCaps")

# spatial dimension = image dimension
# ------- ConvCaps layer -------- #
# i = number of input capsules
# o = number of output capsules
# convolution operation kernel, [kh, kw, i, o] = (3, 3, 16, 1)
# stride = 1 (1,1,1,1) with padding ('same')
# iteration (EM routing) = 3
ConvCaps1, a, b = capsule_functions.conv_capsule(PrimaryCaps, (3, 3, 16, 8), (1, 1, 1, 1), 3, batch_size, "ConvCaps1",
                                                 'SAME')

nets = capsule_functions.class_capsules(ConvCaps1, 1, iterations=3, batch_size=batch_size, name='class_capsules')

y = tf.placeholder(shape=[batch_size, imsize, imsize, 17], dtype=tf.float32, name="y")
y_final_0 = tf.reshape(nets[0], [batch_size, imsize, imsize, -1, 16], name="y_final_0")
y_final_1 = tf.expand_dims(nets[1], -1, name="y_final_1")
y_pred_f = tf.concat([y_final_0, y_final_1], axis=4, name="y_pred_f")
y_pred = tf.squeeze(y_pred_f, axis=3, name="y_pred")

#y = tf.placeholder(shape=[batch_size, imsize, imsize, 17], dtype=tf.float32, name="y")
#y_final_0 = tf.reshape(ConvCaps1[0], [batch_size, imsize, imsize, -1, 16], name="y_final_0")
#y_final_1 = tf.expand_dims(ConvCaps1[1], -1, name="y_final_1")
#y_pred_f = tf.concat([y_final_0, y_final_1], axis=4, name="y_pred_f")
#y_pred = tf.squeeze(y_pred_f, axis=3, name="y_pred")

gstep = tf.placeholder(shape=(), dtype=tf.float32, name="gstep")

diff_y = tf.subtract(y, y_pred, name="diff_y")
squa_y = tf.square(diff_y, name="squa_y") + epsilon
cap_y = tf.minimum(squa_y, gstep ** 2, name="cap_y")

# loss = tf.reduce_mean(squa_y,name="loss") + epsilon
loss = tf.reduce_mean(squa_y, name="loss")

#y_group = tf.squeeze(ConvCaps[1] > 0.5, 3, name="y_group")
y_group = tf.squeeze(nets[1] > 0.5, 3, name="y_group")
y_real = y[:, :, :, 16] > 0.5

correct = tf.equal(y_real, y_group, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters

print(total_parameters)

tf.trainable_variables()

# Tried option: learning_rate = 0.001 / 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
training_op = optimizer.minimize(loss, name="training_op")

saver = tf.train.Saver()
var_init = tf.global_variables_initializer()

checkpoint_path = "./180419_2D_simple_cable"

init = tf.constant_initializer(0.0)

n_epochs = 2
restore_checkpoint = True

n_iterations_per_epoch = 2  # y_train.shape[0] // batch_size
n_iterations_validation = 2  # y_test.shape[0] // batch_size
best_acc_val = 0.

sess = tf.Session()

sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(n_epochs):

    gsteps = np.array((1. * epoch / (n_epochs - 1)) * 0.1 + .5)
    print(gsteps)

    for iteration in range(1, n_iterations_per_epoch + 1):
        # for iteration in range(1,3):
        X_batch = x_train[(iteration - 1) * batch_size:(iteration + 0) * batch_size, :, :, :]
        y_batch = y_train[(iteration - 1) * batch_size:(iteration + 0) * batch_size, :]
        # Run the training operation and measure the loss:
        _, loss_train, mah_pred, mah_real, mah_re = sess.run(
            [training_op, loss, a, b, ConvCaps1],
            feed_dict={x: X_batch,
                       y: y_batch,
                       gstep: gsteps})

        print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.4f}".format(
            iteration, n_iterations_per_epoch,
            iteration * 100 / n_iterations_per_epoch,
            loss_train),
            end="")
        if np.isnan(loss_train):
            break
    print()

    # At the end of each epoch,
    # measure the validation loss and accuracy:
    loss_vals = []
    acc_vals = []
    for iteration in range(1, n_iterations_validation + 1):
        X_batch = x_test[(iteration - 1) * batch_size:(iteration + 0) * batch_size, :, :, :]
        y_batch = y_test[(iteration - 1) * batch_size:(iteration + 0) * batch_size, :]
        loss_val, acc_val, big_bois_pred, big_bois_real = sess.run(
            [loss, accuracy, y_pred, y],
            feed_dict={x: X_batch,
                       y: y_batch,
                       gstep: gsteps})
        loss_vals.append(loss_val)
        acc_vals.append(acc_val)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
            iteration, n_iterations_validation,
            iteration * 100 / n_iterations_validation),
            end=" " * 10)
    loss_val = np.mean(loss_vals)
    acc_val = np.mean(acc_vals)
    print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
        epoch + 1, acc_val * 100, loss_val,
        " (improved)" if acc_val > best_acc_val else ""))

    # And save the model if it improved:
    if acc_val > best_acc_val:
        save_path = saver.save(sess, checkpoint_path)
        best_acc_val = acc_val

