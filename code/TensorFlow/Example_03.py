import tensorflow as tf

# Create two placeholders to hold floating point Tensors

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = tf.constant(3.0, dtype=tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

times_node = adder_node * c

# Create a session object to execute the  computational graph
sess = tf.Session()

print(sess.run(times_node, {a: [1,3], b: [2, 4]}))