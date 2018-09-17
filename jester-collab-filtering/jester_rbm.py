import numpy as np 
import pandas as pd 
import tensorflow as tf


data_1 = pd.read_csv('F:/my files/python files/neural networks/tensorflow_projects/jester-collab-filtering/UserRatings1.csv')
data_2 = pd.read_csv('F:/my files/python files/neural networks/tensorflow_projects/jester-collab-filtering/UserRatings2.csv')
data_1 = data_1.drop(['JokeId'],axis=1)
data_2 = data_2.drop(['JokeId'],axis=1)
data = pd.concat([data_1,data_2],axis=1)
data = data.values
data = np.nan_to_num(data)
data = data.astype('float32')
data = data.transpose()
print(data[73420,:])

inp = tf.placeholder(dtype='float32',shape=[None,100],name='input')
weights = tf.placeholder(dtype='float32',shape=[100,50],name='weights')
forw_bias = tf.placeholder(dtype='float32',shape=[50],name='forward_bias')
back_bias = tf.placeholder(dtype='float32',shape=[100],name='backward_bias') 

mat_forw = tf.add(tf.matmul(inp,weights),forw_bias,name='forward_matrix_operation')
mat_forw = tf.nn.sigmoid(mat_forw)
out = tf.nn.relu(tf.sign(mat_forw-tf.random_uniform(tf.shape(mat_forw))))

mat_back = tf.add(tf.matmul(out,tf.transpose(weights)),back_bias,name='backward_matrix_operation')
mat_back = tf.nn.sigmoid(mat_back)
inp_recon = tf.nn.relu(tf.sign(mat_back-tf.random_uniform(tf.shape(mat_back))),name='input_reconstruction')
out_recon = tf.nn.sigmoid(tf.add(tf.matmul(inp_recon,weights),forw_bias),name='output_reconstruction')

alpha = 1.0
w_pos_grad = tf.matmul(tf.transpose(inp),out)
w_neg_grad = tf.matmul(tf.transpose(inp_recon),out_recon)
CD = (w_pos_grad - w_neg_grad)/ tf.to_float(tf.shape(inp)[0])
upadte_w = weights + alpha*CD
up_fo_bia = forw_bias + alpha*tf.reduce_mean(out-out_recon,0)
up_ba_bia = back_bias + alpha*tf.reduce_mean(inp-inp_recon,0)

err = tf.reduce_mean(tf.square(inp-inp_recon))
cur_wts = np.zeros([100,50],dtype='float32')
cur_for_bias = np.zeros([50],dtype='float32')
cur_back_bias = np.zeros([100],dtype='float32')
prv_wts = np.zeros([100,50],dtype='float32')
prv_for_bias = np.zeros([50],dtype='float32')
prv_back_bias = np.zeros([100],dtype='float32')
init = tf.global_variables_initializer()
epochs = 150
batch = 500
print(prv_for_bias)
with tf.Session() as sess:
    for epoch in range(epochs):
        for start,end in zip(range(0,len(data),batch),range(batch,len(data),batch)):
            xtr = data[start:end]
            cur_wts = sess.run(upadte_w,feed_dict={inp:xtr,weights:prv_wts,forw_bias:prv_for_bias,back_bias:prv_back_bias})
            cur_for_bias = sess.run(up_fo_bia,feed_dict={inp:xtr,weights:prv_wts,forw_bias:prv_for_bias,back_bias:prv_back_bias})
            cur_back_bias = sess.run(up_ba_bia,feed_dict={inp:xtr,weights:prv_wts,forw_bias:prv_for_bias,back_bias:prv_back_bias})
            prv_wts = cur_wts
            prv_for_bias = cur_for_bias
            prv_back_bias = cur_back_bias
        print(sess.run(err,feed_dict={inp:xtr,weights:prv_wts,forw_bias:prv_for_bias,back_bias:prv_back_bias}))
    print(data[73420,:])
