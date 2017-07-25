import tensorflow as tf

# Create two floating point Tensors node1 and node2
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

# Create a session object to execute the  computational graph
sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)

print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))
