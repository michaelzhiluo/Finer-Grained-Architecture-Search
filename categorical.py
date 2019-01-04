import tensorflow as tf

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)."""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """Draw a sample from the Gumbel-Softmax distribution."""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)

def gumbel_softmax_sample_no_noise(logits, temperature):
    """Draw a sample from the Gumbel-Softmax distribution."""
    y = logits #+ sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)

def gumbel_softmax(logits, temperature, hard_assignment):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        A [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard is True, then the returned sample will be one-hot, otherwise it will
        be a probability distribution that sums to one across classes.
    """
    assert temperature > 0, "Temperature must be positive."
    y = gumbel_softmax_sample_no_noise(logits, temperature)
    if hard_assignment is True:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, -1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
