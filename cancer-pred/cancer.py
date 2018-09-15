import pandas as pd
import numpy as np
import tensorflow as tf
data = pd.read_csv('F:/my files/python files/neural networks/tensorflow_projects/cancer-pred/cancer.csv')
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

def ann(ip, wts, bias):
    dense_1 = tf.add(tf.matmul(ip, wts['wd1']), bias['wd1'])
    dense_1 = tf.nn.relu(dense_1)
    #dense_1 = tf.nn.dropout(dense_1,keep_prob=0.3)
    dense_2 = tf.add(tf.matmul(dense_1, wts['wd2']), bias['wd2'])
    dense_2 = tf.nn.relu(dense_2)
    #dense_2 = tf.nn.dropout(dense_2,keep_prob=0.3)
    dense_3 = tf.add(tf.matmul(dense_2, wts['wd3']), bias['wd3'])
    dense_3 = tf.nn.relu(dense_3)
    #dense_3 = tf.nn.dropout(dense_3,keep_prob=0.3)
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
num_epochs = 50
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
        j = 0
        for start,end in zip(range(0,len(x_train),batch_size),range(batch_size,len(x_train),batch_size)):
            batch_x = x_train[start:end]
            batch_y = y_train[start:end]
            j = j+1
            sess.run(train, feed_dict={inp: batch_x, oup: batch_y })
        loss, acc = sess.run([cost, accuracy], feed_dict={inp:x_val, oup:y_val})
        print('epoch:', i, 'loss:', loss, 'accuracy:', acc)


