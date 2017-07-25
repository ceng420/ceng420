
import tensorflow as tf

# linear_model = W * x + b

W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W * x + b

squared_deltas = tf.square(linear_model - y)

loss = tf.reduce_sum(squared_deltas)

''' To initialize all the variables in a TensorFlow program,
    you must explicitly call a special operation as follows: '''

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

# apply the mode with the inputs x with target y and print the sum square error
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
