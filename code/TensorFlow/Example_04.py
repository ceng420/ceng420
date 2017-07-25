import tensorflow as tf

# linear_model = W * x + b

W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

''' To initialize all the variables in a TensorFlow program,
    you must explicitly call a special operation as follows: '''

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

# apply the mode with the inputs x
print(sess.run(linear_model, {x:[1,2,3,4]}))
