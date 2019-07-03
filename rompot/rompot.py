#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:



# this function seeks to ensure reproducibility
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# In[4]:


rs = np.loadtxt('myrALL.csv',delimiter=',')
rs1 = np.reshape(rs, [10,20001,2])
rs2 = np.transpose(rs1, (1,0,2))

qtrain = np.expand_dims(rs2[:,[0],[0]],1)
ptrain = np.expand_dims(rs2[:,[0],[1]],1)
print(qtrain.shape)
print(ptrain.shape)

n_steps = 20001
n_instances = 1
d = 1
dt = 0.001


def tf_diff_axis_0(a):
    return a[1:]-a[:-1]


# In[8]:


# # assume data is in the form: n_steps x n_instances x d
# qdat = np.array([[[0.3, -0.5, 1.0, 7.0],[3.7,-0.4,-0.15, -7.0]], 
#                  [[-7.0,3.14,2.5, 8.0],[3.7,-0.4,-0.15, -7.0]], 
#                  [[2.0,6.14,9.5,9.0],[3.7,-0.4,-0.15, -7.0]]])
# pdat = np.random.normal(size=qdat.shape)
# print(qdat.shape)


# In[9]:


# set up a neural network model for a potential function V : R^d --> R
reset_graph()

# here we take both q and p to be n_steps x n_instances x d
qts = tf.placeholder(tf.float32, shape=(None, n_instances, d), name="qts")
pts = tf.placeholder(tf.float32, shape=(None, n_instances, d), name="pts")

# flatten in such a way that we get a two-dimensional matrix consisting of blocks
# each block consists of one instance of dimension n_steps x d
# the number of such blocks is n_instances
q = tf.transpose(tf.layers.flatten(tf.transpose(qts, perm=[2, 1, 0])),perm=[1, 0])

# we know a priori that the potential depends on the difference

# keep track of outputs of each layer
output = []

# put the inputs in output[0]
output.append(q)

# here is a little Python magic that enables us to define a new function
# called "my_dense_layer" which is the same as the TF function tf.layers.dense
# except that we have preset activation=selu
# ...in short, we have partially evaluated tf.layers.dense...
from functools import partial
my_dense_layer = partial(tf.layers.dense) #,activation=tf.nn.tanh)


# In[10]:


depth = 2
numneurons = [16, 16]

# here is where we define our deep neural network
with tf.name_scope("dnn"):
    
    # we iteratively create hidden layers
    for j in range(depth):
        thisname = "hidden" + str(j)
        
        # the input to hidden layer j is outputs[j]
        # the output of hidden layer j is stored in outputs[j+1]
        # this function handles the creation of all weight and bias variables,
        # for each hidden layer!
        if j == (depth-1):
            myact = tf.nn.softplus
        else:
            myact = tf.nn.softplus

        output.append(my_dense_layer(output[j], 
                                     units=numneurons[j],
                                     name=thisname,
                                     activation=myact))

    # to get from the high-dimensional output of the final hidden layer
    # to a scalar output, we use this function, which basically uses 
    # a linear transformation of the form "w^T h + b"
    # --> h is the vector of outputs from the last hidden layer
    # --> w is a weight vector of the same dimension as h
    # --> b is a scalar
    Vpredraw0 = tf.layers.dense(output[depth], units=1, name='output')
    Vpredraw = tf.exp(Vpredraw0)
    Vpred = tf.reshape(Vpredraw, shape=[n_instances, n_steps])


# In[11]:


# automatically differentiate potential and generate gradV : R^d --> R^d
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
gradVpredraw = batch_jacobian(Vpredraw, q)
gradVpred = tf.reshape(gradVpredraw, shape=[n_instances, n_steps, d])


# In[12]:


# compute loss and set up optimizer
pdot = tf_diff_axis_0(pts)/dt
loss = tf.reduce_mean( tf.square( pdot + tf.transpose(gradVpred[:,:-1], perm=[1,0,2]) ) )
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)


# In[13]:


# typical TF initialization
init = tf.global_variables_initializer()

# we set up TF to save its progress to disk
saver = tf.train.Saver()


# In[ ]:


with tf.Session() as sess:
    init.run()
    saver.restore(sess, "./rompotworking/rompot_model_final.ckpt")
    maxsteps = 500000
    for i in range(maxsteps):
        # we run the "training_op" defined above, corresponding to one optimization step
        # note that we must feed in xx and yy for the placeholders X and y
        sess.run(training_op, feed_dict={qts : qtrain, pts : ptrain})
        # periodically tell us what is happening to the loss function
        if (i % 1000) == 0:
            # note this is another way to grab the value of the loss function
            # we still have to feed in xx and yy for the placeholders X and y
            print(i, "Loss:", loss.eval(feed_dict={qts : qtrain, pts : ptrain}))
    # save final trained TF model to disk
    save_path = saver.save(sess, "./rompot_model_final.ckpt")

