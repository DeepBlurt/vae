#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name:  vae
   File Name：     main
   Description :
   date：          18-4-14
   Change Activity: 恢复模型数据不管用啊
                   
-------------------------------------------------
"""
from vae import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 100
epoch = 800
input_placeholder = tf.placeholder(tf.float32, [batch_size, 784])

vae = Vae(input_placeholder, "./model/model.ckpt")
cost, minimize = vae.train_op(0.001)
reconstruct = vae.predict()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(epoch):
        start = time.time()
        batch, _ = mnist.train.next_batch(batch_size)
        err, _ = sess.run([cost, minimize], feed_dict={input_placeholder: batch})
        duration = time.time() - start
        print(duration, err)
    # vae.save('./model/model.ckpt')
    batch, _ = mnist.train.next_batch(batch_size)
    recon = sess.run(reconstruct, feed_dict={input_placeholder: batch})

n = 10
recon = np.reshape(recon, (100, 28, 28))
batch = np.reshape(batch, (100, 28, 28))
print(recon.shape)
gallery = np.zeros((28*n, 28*2*n))
print(batch.shape)

for i in range(n):
    for j in range(n):
        gallery[i*28:(i+1)*28, j*2*28:j*2*28+28] = recon[i*n+j]
        gallery[i*28:(i+1)*28, j*2*28+28:(j+1)*2*28] = batch[i*n+j]


plt.figure()
plt.imshow(gallery, cmap="gray")
plt.show()

