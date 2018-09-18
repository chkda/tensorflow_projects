# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=C0326
# pylint: disable=C0330
# pylint: disable=C0305
# pylint: disable=C0301
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
from tqdm import tqdm
import cv2 


def load_path(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    return files

def tensor(img):
    imag = cv2.imread(img)
    imag = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY)
    #imag = tf.image.convert_image_dtype(imag, dtype='float32')
    imag = imag/255
    imag = np.expand_dims(imag,axis=3)
    return imag

def file_tensor(img_fol):
    list_of_tensors = [tensor(im) for im in tqdm(img_fol)]
    return np.stack(list_of_tensors, axis=0)


fil = load_path('F:/my files/python files/neural networks/tensorflow_projects/image denoising/shapes')
tens = file_tensor(fil)
x_train = tens[:250]
x_val = tens[250:]
inp = tf.placeholder(dtype='float32',shape=[None,28,28,1],name='input')
oup = tf.placeholder(dtype='float32',shape=[None,28,28,1],name='output')
# encode
conv1 = tf.layers.conv2d(inp,32,(3,3),padding='same',
                        strides=(1,1),activation=tf.nn.relu)
max1 = tf.layers.max_pooling2d(conv1,(2,2),(2,2))
conv2 = tf.layers.conv2d(max1,32,(3,3),strides=(1,1),
                        padding='same',activation=tf.nn.relu)
max2 = tf.layers.max_pooling2d(conv2,(2,2),(2,2))
conv3 = tf.layers.conv2d(max2,16,(3,3),strides=(1,1),
                        padding='same',activation=tf.nn.relu)
encoded = tf.layers.max_pooling2d(conv3,(2,2),(2,2))

# Decoder
upsample1 = tf.image.resize_nearest_neighbor(encoded,size=(7,7))
conv4 = tf.layers.conv2d(upsample1,16,(3,3),strides=(1,1),
                        padding='same',activation=tf.nn.relu)
upsample2 = tf.image.resize_nearest_neighbor(conv4,size=(14,14))
conv5 = tf.layers.conv2d(upsample2,32,(3,3),strides=(1,1),
                        padding='same',activation=tf.nn.relu)
upsample3 = tf.image.resize_nearest_neighbor(conv5,size=(28,28))
conv6 = tf.layers.conv2d(upsample3,32,(3,3),strides=(1,1),
                        padding='same',activation=tf.nn.relu)
fin = tf.layers.conv2d(conv6,1,(3,3),strides=(1,1),
                        padding='same',activation=None)
decoded = tf.nn.sigmoid(fin)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=oup,logits=fin)
los = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate=0.001)
train = opt.minimize(los)
pred = tf.equal(tf.argmax(decoded,1),tf.argmax(oup,1))
acc = tf.reduce_mean(tf.cast(pred,dtype='float32'))
epochs = 500
bat = 10
noise_factor = 0.5
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for start,end in zip(range(0,len(x_train),bat),range(bat,len(x_train),bat)):
            xtr = x_train[start:end]
            #xtr = np.reshape(xtr[0],(-1,28,28,3))
            noi = xtr + noise_factor*np.random.randn(*xtr.shape)
            noi = np.clip(noi,0.,1.)
            lo,_ = sess.run([acc,train], feed_dict={inp:noi,oup:xtr})
            print('epoch:',epoch,'accuracy:',lo)
    nf = x_val + noise_factor*np.random.randn(*x_val[0].shape)
    nf = np.clip(nf,0.,1.)
    re = sess.run(decoded,feed_dict={inp:nf,oup:x_val})
    cv2.imshow('tab',x_val[0])
    cv2.imshow('nf',nf[0])
    cv2.imshow('real',re[0])
    cv2.waitKey()
    cv2.destroyAllWindows()

