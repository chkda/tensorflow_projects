import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
data = pd.read_csv('F:/my files/python files/neural networks/tensorflow_projects/activity-recog/train.csv',sep=',')
out = pd.get_dummies(data['activity'])
data = data.drop(['rn','activity'],axis=1)
out = out.values
data = data.values
xtr,xte,ytr,yte = train_test_split(data,out,test_size=0.3,random_state=42)
print(data.shape)
wts = {'1st':tf.Variable(tf.random_normal([561,300],dtype='float32')),
       '2nd':tf.Variable(tf.random_normal([300,150],dtype='float32')),
       '3rd':tf.Variable(tf.random_normal([150,50],dtype='float32')),
       '4th':tf.Variable(tf.random_normal([50,6],dtype='float32'))
       }

bias = {'1st':tf.Variable(tf.random_normal([300],dtype='float32')),
        '2nd':tf.Variable(tf.random_normal([150],dtype='float32')),
        '3rd':tf.Variable(tf.random_normal([50],dtype='float32')),
        '4th':tf.Variable(tf.random_normal([6],dtype='float32'))
        }
def nn_model(x_t,weight,bia):
    den_1 = tf.add(tf.matmul(x_t,weight['1st']),bia['1st'])
    den_1 = tf.nn.relu(den_1)
    #den_1 = tf.nn.dropout(den_1,keep_prob=0.2)
    den_2 = tf.add(tf.matmul(den_1,weight['2nd']),bia['2nd'])
    den_2 = tf.nn.relu(den_2)
    #den_2 = tf.nn.dropout(den_2,keep_prob=0.2)
    den_3 = tf.add(tf.matmul(den_2,weight['3rd']),bia['3rd'])
    den_3 = tf.nn.relu(den_3)
    #den_3 = tf.nn.dropout(den_3,keep_prob=0.2)
    den_4 = tf.add(tf.matmul(den_3,weight['4th']),bia['4th'])
    return den_4
inp = tf.placeholder(dtype='float32',shape=[None,561],name='inputs')
oup = tf.placeholder(dtype='float32',shape=[None,6],name='label')
pred = nn_model(inp,wts,bias)
cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=oup)
cost = tf.reduce_mean(cost)
opt = tf.train.AdamOptimizer(learning_rate=0.001)
train = opt.minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(oup,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,dtype='float32'))
bat = 50
epochs = 100
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for start,end in zip(range(0,len(data),bat),range(bat,len(data),bat)):
            batch_x = xtr[start:end]
            batch_y = ytr[start:end]
            sess.run(train,feed_dict={inp:batch_x,oup:batch_y})
        loss,acc = sess.run([cost,accuracy],feed_dict={inp:xte,oup:yte})
        print('epoch:',epoch,'loss=',loss,'accuracy=',acc)



