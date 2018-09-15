import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
data = np.load('F:/my files/python files/neural networks/tensorflow_projects/sign-language-pred/Sign-language-digits-dataset/X.npy')
y = np.load('F:/my files/python files/neural networks/tensorflow_projects/sign-language-pred/Sign-language-digits-dataset/Y.npy')
x = np.expand_dims(data, axis=3)
print(x.shape)

x_train, x_test = train_test_split(x, test_size=0.3, random_state=42)
x_test, x_val = train_test_split(x_test, test_size=0.3, random_state=42)
y_train, y_test = train_test_split(y, test_size=0.3, random_state=42)
y_test, y_val = train_test_split(y_test, test_size=0.3, random_state=42)


inp = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name='input')
oup = tf.placeholder(tf.float32, shape=[None, 10], name='labels')


# creating convolution layer
def conv_layer(inp_layer, wts):
    layer = tf.nn.conv2d(inp_layer, wts, strides=[1, 1, 1, 1], padding='SAME')
    layers = tf.nn.relu(layer)
    return layers


# creating max-pooling layers
def max_pool(inp_layer, k):
    layer = tf.nn.max_pool(inp_layer, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    return layer

#creatinf the network
def nn_model(inputs, weight, bias):
    conv_1 = conv_layer(inputs, weight['wc1'])
    pool_1 = max_pool(conv_1, 2)
    conv_2 = conv_layer(pool_1, weight['wc2'])
    pool_2 = max_pool(conv_2, 2)
    conv_3 = conv_layer(pool_2, weight['wc3'])
    pool_3 = max_pool(conv_3, 2)
    conv_4 = conv_layer(pool_3, weight['wc4'])
    pool_4 = max_pool(conv_4, 2)
    flat = tf.contrib.layers.flatten(pool_4)
    drop_1 = tf.nn.dropout(flat, 1.0)
    dense_1 = tf.add(tf.matmul(drop_1, weight['wd1']), bias['wd1'])
    drop_2 = tf.nn.dropout(dense_1, 1.0)
    dense_2 = tf.add(tf.matmul(drop_2, weight['wd2']), bias['wd2'])
    drop_3 = tf.nn.dropout(dense_2, 1.0)
    dense_3 = tf.add(tf.matmul(drop_3, weight['wd3']), bias['wd3'])
    return dense_3


weights = {'wc1': tf.Variable(tf.truncated_normal(shape=[4, 4, 1, 32])),
           'wc2': tf.Variable(tf.truncated_normal(shape=[4, 4, 32, 64])),
           'wc3': tf.Variable(tf.truncated_normal(shape=[2, 2, 64, 128])),
           'wc4': tf.Variable(tf.truncated_normal(shape=[2, 2, 128, 256])),
           'wd1': tf.Variable(tf.truncated_normal(shape=[4096, 512])),
           'wd2': tf.Variable(tf.truncated_normal(shape=[512, 64])),
           'wd3': tf.Variable(tf.truncated_normal(shape=[64, 10]))
           }

biases = {'wd1': tf.Variable(tf.truncated_normal(shape=[512])),
          'wd2': tf.Variable(tf.truncated_normal(shape=[64])),
          'wd3': tf.Variable(tf.truncated_normal(shape=[10]))}

pred = nn_model(inp, weights, biases)
cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits= pred, labels=oup)
cost = tf.reduce_mean(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(oup, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
batch_size = 50
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(50):
        tot_batch = len(x_train)//batch_size
        for start,end in zip(range(0,len(x_train),batch_size),range(batch_size,len(x_train),batch_size)):
            batch_x = x_train[start:end]
            batch_y = y_train[start:end]
            
            sess.run(train, feed_dict={inp: batch_x, oup: batch_y})
        loss, acc = sess.run([cost, accuracy], feed_dict={inp: x_val, oup: y_val})
        print('epoch:', i, 'loss:', loss, 'accuracy', acc)






