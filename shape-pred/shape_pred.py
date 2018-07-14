import tensorflow as tf
from sklearn.datasets import load_files
import numpy as np
import cv2
from tqdm import tqdm


#function to load data
def load_data(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    #targets = tf.keras.utils.to_categorical(np.array(data['target']), 3)
    targets = tf.one_hot(np.array(data['target']), 3)
    return files, targets


#converts images to tensor
def tensor(img):
    imag = cv2.imread(img)
    imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
    imag = tf.image.resize_images(imag, size=(24, 24))
    imag = tf.image.convert_image_dtype(imag, dtype='float32')
    imag = tf.image.per_image_standardization(imag)
    #imag = tf.expand_dims(imag, axis=0)
    return imag


#convert total folder of images to 3d tensor
def file_tensor(img_fol):
    list_of_tensors = [tensor(img) for img in tqdm(img_fol)]
    return tf.stack(list_of_tensors, axis=0)


#creating a function for convolution layers
def conv(x, w):
    conv_layer = tf.nn.conv2d(x, filter=w, strides=[1, 2, 2, 1], padding='VALID')
    conv_layer = tf.nn.relu(conv_layer)
    return conv_layer


#creating a function for max_pool layer
def max_pool(a, k):
    mp = tf.nn.max_pool(a, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')
    return mp


#creating a model
def nn_model(x, weights):
    conv_1 = conv(x, weights['wc1'])
    pool_1 = max_pool(conv_1, 2)
    conv_2 = conv(pool_1, weights['wc2'])
    pool_2 = max_pool(conv_2, 2)
    flatten = tf.contrib.layers.flatten(pool_2)
    flatten = tf.nn.dropout(flatten, 0.3)
    dense_1 = tf.matmul(flatten, weights['wd1'])
    dense_1 = tf.nn.relu(dense_1)
    drop = tf.nn.dropout(dense_1, 0.3)
    dense_2 = tf.matmul(drop, weights['wd2'])
    return dense_2


#creating batches
def get_batches(batch_size, data, lab):
    batch_data = tf.data.Dataset.from_tensor_slices(data)
    batch_labels = tf.data.Dataset.from_tensor_slices(lab)
    #batch_labels = tf.reshape(batch_labels, shape=tar[0].shape)
    tot_data = tf.data.Dataset.zip((batch_data, batch_labels))
    batched_data = tot_data.repeat().batch(batch_size)
    iterator = batched_data.make_initializable_iterator()
    sess.run(iterator.initializer)
    next_batch = iterator.get_next()
    return next_batch


#loading the data
fil, tar = load_data('datasets/shapes')
tensors = file_tensor(fil)
x_train = tensors[:250]
y_train = tar[:250]
x_val = tensors[250:]
y_val = tar[250:]
#tar = tf.convert_to_tensor(tar, dtype='float32')
#tensors = [tensor[i].eval() for i in len(tensor)]
x = tf.placeholder(dtype='float32', shape=[None, 24, 24, 3], name='input')
labels = tf.placeholder(dtype='float64', shape=[None, 3], name='labels')
weights = {'wc1': tf.Variable(tf.random_normal([2, 2, 3, 32])),
           'wc2': tf.Variable(tf.random_normal([2, 2, 32, 64])),
           'wd1': tf.Variable(tf.random_normal([64, 240])),
           'wd2': tf.Variable(tf.random_normal([240, 3]))}

pred = nn_model(x, weights)
cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels)
cost = tf.reduce_mean(cost)
optimiser = tf.train.AdagradOptimizer(learning_rate=0.001)
train = optimiser.minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype='float32'))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for j in range(10):
        for i in range(5):
            batch_x, batch_y = get_batches(50, x_train, y_train)
            val_x, val_y = get_batches(50, x_val, y_val)
            try:
                batch_x = sess.run(batch_x)
                batch_y = sess.run(batch_y)
                val_x = sess.run(val_x)
                val_y = sess.run(val_y)
            except tf.errors.OutOfRangeError:
                None

            sess.run(train, feed_dict={x: batch_x, labels: batch_y})
            #sess.run(train, feed_dict={x: tensor[i].eval(), labels: tar[i]})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: val_x, labels: val_y})
            #loss, acc = sess.run([cost, accuracy], feed_dict={x: tensor[i].eval(), labels: tar[i]})
            print('epoch:', j, 'iter:',  i, 'loss:', loss, 'accuracy:', acc)








