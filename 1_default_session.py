import tensorflow as tf

# find polynomial minimum
x = tf.Variable(5, dtype=tf.float32)
y = x**2 - 25

# explore other optimizers in tf.train
optimizer = tf.train.GradientDescentOptimizer(.1)
train_step = optimizer.minimize(y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        sess.run(train_step)
        print(sess.run(x))
