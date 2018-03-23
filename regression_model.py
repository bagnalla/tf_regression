import gzip, pickle, os.path
import tensorflow as tf

INPUT_SIZE = 1
NUM_HIDDEN_LAYERS = 0
HIDDEN_SIZES = [] # length should equal NUM_HIDDEN_LAYERS
INCLUDE_BIASES = False
WEIGHT_DECAY = 0.00002


# This builds the feedforward network op and returns it. A weight
# decay term is added to a collection so it can be referred to by the
# loss function.
def inference(images, name='m0', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        sizes = HIDDEN_SIZES + [1]
        w0 = tf.get_variable('w0',
                             (INPUT_SIZE,
                              (sizes[0]
                               if NUM_HIDDEN_LAYERS > 0 else 1)),
                             initializer=tf.contrib.layers.xavier_initializer())
        ws = [w0] + [tf.get_variable(
            'w' + str(i+1), [sizes[i], sizes[i+1]],
            initializer=tf.contrib.layers.xavier_initializer())
                     for i in range(NUM_HIDDEN_LAYERS)]
        weight_decay = tf.multiply(sum([tf.nn.l2_loss(w) for w in ws]),
                                   WEIGHT_DECAY, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        l = images
        if INCLUDE_BIASES:
            b0 = tf.get_variable(
                'b0', [sizes[0]],
                initializer=tf.contrib.layers.xavier_initializer())
            bs = [b0] + [tf.get_variable(
                'b' + str(i+1), [sizes[i+1]],
                initializer=tf.contrib.layers.xavier_initializer())
                         for i in range(NUM_HIDDEN_LAYERS)]
            for i in range(NUM_HIDDEN_LAYERS):
                l = tf.nn.relu(tf.matmul(l, ws[i] + bs[i]))
        else:
            for i in range(NUM_HIDDEN_LAYERS):
                l = tf.nn.relu(tf.matmul(l, ws[i]))
        out = tf.matmul(l, ws[-1])
        return out


def loss(hypothesis, values):
    mse = tf.losses.mean_squared_error(values, hypothesis)
    weight_reg = tf.add_n(tf.get_collection('losses'))
    loss = mse + weight_reg
    # loss = tf.Print(loss, [values, mse, weight_reg])
    return loss


# The training op.
def training(loss, x, learning_rate, decay_step, decay_factor):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(learning_rate, global_step, decay_step,
                                    decay_factor, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    grads = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads)
    return train_op


def save_weights(sess, dir='models'):
    os.makedirs(dir, exist_ok=True)
    all_vars = tf.trainable_variables()
    with gzip.open(dir + "/params.pkl.gz", "w") as f:
        pickle.dump(tuple(map(lambda x: x.eval(sess), all_vars)), f)


def load_weights(sess, dir, model_name='m0'):
    filename = dir + '/params.pkl.gz' if dir else 'params.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')
        with tf.variable_scope(model_name, reuse=True):
            sizes = [INPUT_SIZE] + HIDDEN_SIZES + [1]
            if INCLUDE_BIASES:
                for i in range(NUM_HIDDEN_LAYERS+1):
                    w_var = tf.get_variable('w' + str(i),
                                            [sizes[i], sizes[i+1]])
                    b_var = tf.get_variable('b' + str(i),
                                            [sizes[i+1]])
                    sess.run([w_var.assign(weights[2*i]),
                              b_var.assign(weights[2*i+1])])
            else:
                for i in range(NUM_HIDDEN_LAYERS+1):
                    w_var = tf.get_variable('w' + str(i),
                                            [sizes[i], sizes[i+1]])
                    sess.run([w_var.assign(weights[i])])
