import numpy as np
import tensorflow as tf
from functools import reduce

train_x = [ [ 0, 0 ]
          , [ 0, 1 ]
          , [ 1, 0 ]
          , [ 1, 1 ]
          ]
train_y = [ [ 0 ]
          , [ 1 ]
          , [ 1 ]
          , [ 0 ] 
          ]

train_inputs = np.array( train_x ).astype(np.float32)
train_labels = np.array( train_y ).astype(np.float32)

def makePlaceholder( x ):
    return tf.placeholder( tf.float32, shape=[ None, len( x[0] ) ] )


def addBias( x ):
    x_rows = tf.shape( x )[0]
    bias_expanded = tf.tile( tf.ones( [1,1], dtype=tf.float32 )
                           , tf.stack( [x_rows, 1] )
                           )
    return tf.concat( [bias_expanded, x], 1 )


graph_nn = tf.Graph()
with graph_nn.as_default():
    # make placeholder for input and output
    x = makePlaceholder( train_x )
    y = makePlaceholder( train_y )

    # NN model
    xWithBias = addBias( x )
    w1 = tf.Variable( tf.random_uniform([ 3, 2 ], -1.0, 1.0) )
    h1 = tf.nn.sigmoid( tf.matmul( xWithBias, w1 ) )

    h1WithBias = addBias(h1)
    w2 = tf.Variable( tf.random_uniform([ 3, 1 ], -1.0, 1.0) )
    h2 = tf.nn.sigmoid( tf.matmul( h1WithBias, w2 ) )


    # culc diff
    # loss = -tf.reduce_sum( y*tf.log(h2) + (1-y)*tf.log(1-h2) )
    loss = tf.reduce_mean( tf.square(h2 - y) )

    # optimize with GradientDescent
    train_step = tf.train.GradientDescentOptimizer( 0.1 ).minimize( loss )

    # correct check
    # correct_check = reduce( lambda f, s: f and s, tf.equal(tf.to_float(tf.greater(h2, 0.5)), y) )
    correct_check = tf.equal( tf.to_float(tf.greater(h2, 0.5)), y )


with tf.Session(graph=graph_nn) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    fd = { x: train_inputs
         , y: train_labels
         }


    for step in range(50000):
        sess.run( train_step, feed_dict=fd )
        if (step+1)% 5000 == 0:
            print(step+1)

    prob = h2.eval( session=sess, feed_dict=fd )
    checked = correct_check.eval( session=sess, feed_dict=fd )
    isSucceed = reduce( lambda f, s: f and s, checked )

 
    print('output probability:')
    print(prob)
    print('classification is {0}'.format("Succeeded!" if isSucceed else "failed."))

