import pandas as pd
import numpy as np
import tensorflow as tf
data = pd.read_csv('datasets/cancer.csv')
feat = data.iloc[:, 2:32]
feat = feat.values
feat = feat.astype('float32')
feat = np.reshape(feat, (569, 30))
list_feat = [arr for arr in feat]
list_feat = tf.stack(list_feat, axis=0)
labels = data['diagnosis']
labels = pd.get_dummies(labels, prefix='labels')
labels = labels.values
labels = labels.astype('float32')
labels = np.reshape(labels, (569, 2))
print(feat[0].shape)

def get_batches(inp1, oup, batch_size):
    data = tf.data.Dataset.from_tensor_slices(inp1)
    lab = tf.data.Dataset.from_tensor_slices(oup)
    batched_data = tf.data.Dataset.zip((data, lab))
    tot_data = batched_data.repeat().batch(batch_size)
    iterator = tot_data.make_initializable_iterator()
    sess.run(iterator.initializer)
    next_batch = iterator.get_next()
    return next_batch


def ann(ip, wts, bias):
    dense_1 = tf.add(tf.matmul(ip, wts['wd1']), bias['wd1'])
    dense_1 = tf.nn.relu(dense_1)
    dense_2 = tf.add(tf.matmul(dense_1, wts['wd2']), bias['wd2'])
    dense_2 = tf.nn.relu(dense_2)
    dense_3 = tf.add(tf.matmul(dense_2, wts['wd3']), bias['wd3'])
    dense_3 = tf.nn.relu(dense_3)
    dense_4 = tf.add(tf.matmul(dense_3, wts['wd4']), bias['wd4'])
    return dense_4


inp = tf.placeholder(dtype='float32', shape=[None, 30], name='inputs')
oup = tf.placeholder(dtype='float32', shape=[None, 2], name='labels')

weights = {'wd1': tf.Variable(tf.truncated_normal(shape=[30, 64])),
           'wd2': tf.Variable(tf.truncated_normal(shape=[64, 256])),
           'wd3': tf.Variable(tf.truncated_normal(shape=[256, 30])),
           'wd4': tf.Variable(tf.truncated_normal(shape=[30, 2]))}
biases = {'wd1': tf.Variable(tf.truncated_normal(shape=[64])),
          'wd2': tf.Variable(tf.truncated_normal(shape=[256])),
          'wd3': tf.Variable(tf.truncated_normal(shape=[30])),
          'wd4': tf.Variable(tf.truncated_normal(shape=[2]))}
print(inp.get_shape())
test_size = 500
x_train = feat[:test_size]
x_val = feat[test_size:]
y_train = labels[:test_size]
y_val = labels[test_size:]
batch_size = 20
num_epochs = 30
pred = ann(inp, weights, biases)
cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=oup)
cost = tf.reduce_mean(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(oup, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float32'))
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(num_epochs):
        num_batches = len(feat)//batch_size

        for j in range(num_batches):
            batch_x, batch_y = get_batches(feat, labels, 50)
            val_x, val_y = get_batches(x_val, y_val, 569-test_size)
            try:
                batch_x = sess.run(batch_x)
                batch_y = sess.run(batch_y)
                val_x = sess.run(val_x)
                val_y = sess.run(val_y)
            except tf.errors.OutOfRangeError:
                None
            sess.run(train, feed_dict={inp: batch_x, oup: batch_y })
            loss, acc = sess.run([cost, accuracy], feed_dict={inp: val_x, oup: val_y})
            print('epoch:', i, 'batch no:', j, 'loss:', loss, 'accuracy:', acc)


