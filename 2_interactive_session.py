import tensorflow as tf

# The only difference with a regular Session is that an InteractiveSession installs itself as
# the default session on construction. The methods tf.Tensor.eval and tf.Operation.run will use that session to run ops.

# important: Execution: tf.Operation.run(), tf.Tensor.eval() vs sess.run(tf.Operation), sess.run(tf.Tensor)

sess = tf.InteractiveSession()

x = tf.Variable(5, dtype=tf.float32)
y = x ** 2 - 25

optimizer = tf.train.GradientDescentOptimizer(.1)
train_step = optimizer.minimize(y)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(10):
    # sess.run(train_step)
    train_step.run()
    print(x.eval())
