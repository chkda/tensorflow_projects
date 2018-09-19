# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=C0326
# pylint: disable=C0330
# pylint: disable=C0305
# pylint: disable=C0301
import cv2
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
from tqdm import tqdm

def load_path(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    return files

def extract(img):
    imag = cv2.imread(img)
    imag = cv2.cvtColor(imag,cv2.COLOR_BGR2RGB)
    imag = imag/255
    return imag

def tensor_4d(img_fol):
    list_of_tensors = [extract(im) for im in tqdm(img_fol)]
    return np.stack(list_of_tensors,axis=0)

def model_inputs(real_dim,z_dim):
    real_inp = tf.placeholder(dtype='float32',shape=[None,real_dim[0],real_dim[1],real_dim[2]])
    z_inp = tf.placeholder(dtype='float32',shape=[None,z_dim])
    return real_inp,z_inp

def generator(z,out_dim,n_units,alp,reuse=False):
    with tf.variable_scope('generator',reuse=reuse):
        la1 = tf.layers.dense(z,16*16*2*n_units,activation=None)
        la1 = tf.reshape(la1,(-1,16,16,2*n_units))
        la1 = tf.layers.batch_normalization(la1,training=True)
        la1 = tf.nn.leaky_relu(la1,alpha=alp)

        la2 = tf.layers.conv2d_transpose(la1,n_units,strides=(2,2),kernel_size=3,activation=None,padding='same')
        la2 = tf.layers.batch_normalization(la2,training=True)
        la2 = tf.nn.leaky_relu(la2,alpha=alp)

        la3 = tf.layers.conv2d_transpose(la2,out_dim,kernel_size=3,strides=(2,2),activation=tf.nn.tanh,padding='same')
        return la3

def discriminator(x,n_units,alp,reuse=False):
    with tf.variable_scope('discriminator',reuse=reuse):
        lay1 = tf.layers.conv2d(x,n_units,kernel_size=3,strides=(2,2),padding='same')
        lay1 = tf.layers.flatten(lay1)
        lay1 = tf.layers.batch_normalization(lay1,training=True)
        lay1 = tf.nn.leaky_relu(lay1,alpha=alp)

        lay2 = tf.layers.dense(lay1,1,activation=None)
        out = tf.nn.sigmoid(lay2)
        return lay2,out

input_size = (64,64,3)
z_size = 100
g_units = 16
d_units = 4
alph = 0.01
smoothing = 0.1

fil = load_path('F:/my files/python files/neural networks/tensorflow_projects/gen-adv-netw/images')
tens = tensor_4d(fil)

tf.reset_default_graph()
input_real,z_real = model_inputs(input_size,z_size)
g_model = generator(z_real,input_size[2],n_units=g_units,alp=alph)
d_real_log,d_real_out = discriminator(input_real,n_units=d_units,alp=alph)
d_fake_log,d_fake_out = discriminator(g_model,n_units=d_units,alp=alph,reuse=True)

lab = tf.ones_like(d_real_log)*(1-smoothing)
d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_log,labels=lab)
d_loss_real = tf.reduce_mean(d_loss_real)
d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_log,labels=tf.zeros_like(d_fake_log))
d_loss_fake = tf.reduce_mean(d_loss_fake)
d_loss = d_loss_real + d_loss_fake
g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_log,labels=tf.ones_like(d_fake_log))
g_loss = tf.reduce_mean(g_loss)

lr = 0.001
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
d_train_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(d_loss,var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(g_loss,var_list=g_vars)

batch = 100
epochs = 150
samples = []
losses = []
saver = tf.train.Saver(var_list=g_vars)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for start,end in zip(range(0,len(tens),batch),range(batch,len(tens),batch)):
            im = tens[start:end]
            im = im*2 - 1
            bat_z = np.random.uniform(-1,1,size=(batch,z_size))
            _ = sess.run(d_train_opt,feed_dict={input_real:im,z_real:bat_z})
            _ = sess.run(g_train_opt,feed_dict={z_real:bat_z})

        train_loss_d = sess.run(d_loss,feed_dict={input_real:im,z_real:bat_z})
        train_loss_g = sess.run(g_loss,feed_dict={z_real:bat_z})
        print('Epoch:',epoch,'Dis loss:',train_loss_d,'Gen loss:',train_loss_g)
    sam_z = np.random.uniform(-1,1,size=(2,z_size))
    imj = sess.run(generator(z_real,input_size[2],n_units=g_units,alp=alph,reuse=True),feed_dict={z_real:sam_z})

imj[0] = cv2.cvtColor(imj[0],cv2.COLOR_RGB2BGR)
imj[1] = cv2.cvtColor(imj[1],cv2.COLOR_RGB2BGR)
cv2.imshow('tab',imj[0])
cv2.imshow('fd',imj[1])
cv2.imshow('fg',tens[0])
cv2.waitKey()
cv2.destroyAllWindows()
