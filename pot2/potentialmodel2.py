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


f = np.load('coultrajnew.npz')
qdat = np.transpose(f['r'],(0,2,1))[:,:800,:]
pdat = np.transpose(f['v'],(0,2,1))[:,:800,:]

# remember momentum = mass * velocity
# so we msut set mass of second particle
pdat[:,:,3:] *= 0.5

n_steps, n_instances, d = qdat.shape
n_instances = 10
dt = 0.001
print(qdat.shape)
print(pdat.shape)


# In[5]:



# In[7]:


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
qdiff = tf.norm(q[:, 3:] - q[:, :3], axis=1, keepdims=True)
print(qdiff)

# keep track of outputs of each layer
output = []

# put the inputs in output[0]
output.append(qdiff)

# here is a little Python magic that enables us to define a new function
# called "my_dense_layer" which is the same as the TF function tf.layers.dense
# except that we have preset activation=selu
# ...in short, we have partially evaluated tf.layers.dense...
from functools import partial
my_dense_layer = partial(tf.layers.dense) #,activation=tf.nn.tanh)


# In[10]:


depth = 2
numneurons = [256, 256]

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
            myact = tf.nn.tanh
        else:
            myact = tf.nn.tanh

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
    Vpredraw = tf.layers.dense(output[depth], units=1, name='output')
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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
training_op = optimizer.minimize(loss)


# In[13]:


# typical TF initialization
init = tf.global_variables_initializer()

# we set up TF to save its progress to disk
saver = tf.train.Saver()


# In[ ]:


with tf.Session() as sess:
    init.run()
    maxsteps = 50000
    batches = 80
    for i in range(maxsteps):
        # we run the "training_op" defined above, corresponding to one optimization step
        # note that we must feed in xx and yy for the placeholders X and y
        si = (i % 80) * 10
        ei = si + 10
        print("Batch: ",si, ei)
        qbatch = qdat[:, si:ei, :]
        pbatch = pdat[:, si:ei, :]
        sess.run(training_op, feed_dict={qts : qbatch, pts : pbatch})
        # periodically tell us what is happening to the loss function
        if (i % 1000) == 0:
            # note this is another way to grab the value of the loss function
            # we still have to feed in xx and yy for the placeholders X and y
            print(i, "Loss:", loss.eval(feed_dict={qts : qbatch, pts : pbatch}))
    # save final trained TF model to disk
    save_path = saver.save(sess, "./pot2_model_final.ckpt")

