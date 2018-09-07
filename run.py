from time import time
import numpy as np
import tensorflow as tf

from skimage import data
from skimage.transform import resize
from scipy.misc import imresize

import matplotlib.pyplot as plt
plt.style.use("ggplot")

import gif
from helpers import flatten, linear, im_split


# Loading the data - A single image representing an astronaut (change if you need)
img = data.astronaut()
plt.imshow(img);

xs, ys = im_split(img)
plt.imshow(ys.reshape(img.shape));


### Normalizing
# So the NN won't focus on the position of the pixel (color), but rather on the disribution
xs_norm = (xs - np.mean(xs)) / np.std(xs)
ys_norm = ys / 255.0

plt.imshow(ys_norm.reshape(img.shape));


### Build the graph:
# Reseting TensorFlow default graph
tf.reset_default_graph()

X = tf.placeholder(name='X', shape=[None, 2], dtype=tf.float32)
Y = tf.placeholder(name='Y', shape=[None, 3], dtype=tf.float32)

n_neurons = 64
layers = [2, n_neurons, n_neurons, n_neurons, n_neurons, n_neurons, n_neurons, 3]
activ = tf.nn.relu

# Specify "reuse=True" if errors or try resest the default graph

actual_layer = X
for layer_i in range(1, len(layers)):
    actual_layer = linear(
        X=actual_layer,
        activation=activ if layer_i < (len(layers) - 1) else None,
        n_outputs=layers[layer_i],
        scope=("layer_" + str(layer_i))
    )
    
Y_pred = actual_layer

assert(X.get_shape().as_list() == [None, 2])
assert(Y.get_shape().as_list() == [None, 3])
assert(Y_pred.get_shape().as_list() == [None, 3])

#l2 loss
cost = tf.squared_difference(x=Y, y=Y_pred)
cost = tf.reduce_sum(cost, axis=1)
cost = tf.reduce_mean(cost)
assert(cost.get_shape().as_list() == [])

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

n_iterations = 200
batch_size = 50

sess = tf.Session()


### Training day:

sess.run(tf.global_variables_initializer())

imgs = []
costs = []

gif_step = n_iterations // 10
step_i = 0

tic = time()

for iter_i in range(n_iterations):
    idxs = np.random.permutation(range(len(xs)))
    n_batches = len(xs) // batch_size
    
    for batch_i in range(n_batches):
        idxs_i = idxs[batch_i * batch_size : (batch_i + 1) * batch_size]
        
        train_cost = sess.run(
            fetches=[cost, optimizer],
            feed_dict={
                X: xs_norm[idxs_i],
                Y: ys_norm[idxs_i]
            }
        )[0]
        
    # Every "gif_step" iters, draw a prediction, and try to recreate the image
    if (iter_i + 1) % gif_step == 0:     # The +1 is just to include the last iter (y)
        costs.append(train_cost / n_batches)
        ys_pred = Y_pred.eval(feed_dict={X: xs_norm}, session=sess)
        
        img_pred = np.clip(a=(ys_pred * 255).reshape(img.shape), a_min=0, a_max=255).astype(np.uint8)
        imgs.append(img_pred)
        
        # Plot the cost
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(costs)
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Cost")
        ax[1].imshow(img_pred)
        fig.suptitle("Iteration: {}".format(iter_i))
        plt.show()

toc = time()
print("\n\nDuration: {:.2f} sec".format(toc - tic))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(img)
axs[1].imshow(imgs[-1]);

gif.build_gif(imgs, saveto="anim.gif", interval=0.1)
ipyd.Image(url='anim.gif?{}'.format(np.random.rand()), height=500, width=500)




#### Wrapping the whole stuff in a func:
def train(x_train, y_train, optimizer, loss, scope=None, n_iterations=100, batch_size=50):
    with tf.Session() as sess, tf.variable_scope(scope or "new_train"):
        
        sess.run(tf.global_variables_initializer())

        imgs = []
        costs = []

        gif_step = n_iterations // 10 if (n_iterations >= 10) else 1
        step_i = 0


        tic = time()

        for iter_i in range(n_iterations):
            idxs = np.random.permutation(range(len(x_train)))
            n_batches = len(x_train) // batch_size

            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i * batch_size : (batch_i + 1) * batch_size]

                train_cost = sess.run(
                    fetches=[loss, optimizer],
                    feed_dict={
                        X: x_train[idxs_i],
                        Y: y_train[idxs_i]
                    }
                )[0]

            # Every "gif_step" iters, draw a prediction, and try to recreate the image
            if (iter_i + 1) % gif_step == 0:     # The +1 is just to include the last iter (y)
                costs.append(train_cost / n_batches)
                ys_pred = Y_pred.eval(feed_dict={X: x_train}, session=sess)

                if x_train.std() == 1.0:
                    ys_pred = ys_pred * 255
                    
                img_pred = np.clip(a=ys_pred.reshape(img.shape), a_min=0, a_max=255).astype(np.uint8)
                imgs.append(img_pred)

                # Plot the cost
                fig, ax = plt.subplots(1, 2)
                ax[0].plot(costs)
                ax[0].set_xlabel("Iteration")
                ax[0].set_ylabel("Cost")
                ax[1].imshow(img_pred)
                fig.suptitle("Iteration: {}".format(iter_i + 1))
                plt.show()

                
        toc = time()
        print("\n\nDuration: {:.2f} sec".format(toc - tic))
        
        return imgs, costs

train(xs_norm, ys_norm, optimizer=optimizer, loss=cost, n_iterations=1, scope="train");
