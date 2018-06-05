import tensorflow as tf

# lets introduce coefficients of quadratic equation; y = a*x^2 + b*x

# Placeholder is a container for data. Its value (data) can be filled and overwritten when needed
coeffs = tf.placeholder(dtype=tf.float32, shape=(2,))

x = tf.Variable(2, dtype=tf.float32)

# replaces a * b with coeffs
y = coeffs[0] * x ** 2 + coeffs[1] * x

optimizer = tf.train.GradientDescentOptimizer(.001)
train_step = optimizer.minimize(y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        sess.run(train_step, feed_dict={coeffs: [3, 2]})
        print(sess.run(x))
