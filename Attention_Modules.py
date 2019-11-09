import tensorflow as tf

class Transformer(tf.keras.layers.Attention):
    '''
    Refer: https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/dense_attention.py#L182-L303
    This is only for getting the attention history(scores).
    '''
    def __init__(self, size, use_scale=False, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.use_scale = use_scale

        self.layer_Dict = {
            'Query': tf.keras.layers.Dense(size),
            'Value': tf.keras.layers.Dense(size),
            'Key': tf.keras.layers.Dense(size)
            }

    def call(self, inputs, mask=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        q = self.layer_Dict['Query'](inputs[0])
        v = self.layer_Dict['Value'](inputs[1])
        k = self.layer_Dict['Key'](inputs[2]) if len(inputs) > 2 else v
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k)
        if v_mask is not None:
            # Mask of shape [batch_size, 1, Tv].
            v_mask = tf.expand_dims(v_mask, axis=-2)
        if self.causal:
            # Creates a lower triangular mask, so position i cannot attend to
            # positions j>i. This prevents the flow of information from the future
            # into the past.
            scores_shape = tf.shape(scores)
            # causal_mask_shape = [1, Tq, Tv].
            causal_mask_shape = tf.concat(
                [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]],
                axis=0)
            causal_mask = _lower_triangular_mask(causal_mask_shape)
        else:
            causal_mask = None
        scores_mask = _merge_masks(v_mask, causal_mask)
        result, attention_distribution = _apply_scores(scores=scores, value=v, scores_mask=scores_mask)
        if q_mask is not None:
            # Mask of shape [batch_size, Tq, 1].
            q_mask = tf.expand_dims(q_mask, axis=-1)
            result *= tf.cast(q_mask, dtype=result.dtype)
        return result, attention_distribution

class BahdanauAttention(tf.keras.layers.AdditiveAttention):
    '''
    Refer: https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/dense_attention.py#L307-L440
    This is for getting the attention history(scores).
    '''
    def __init__(self, size, use_scale=False, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)        
        self.use_scale = use_scale

        self.layer_Dict = {
            'Query': tf.keras.layers.Dense(size),
            'Value': tf.keras.layers.Dense(size),
            'Key': tf.keras.layers.Dense(size)
            }

    def call(self, inputs, mask=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        q = self.layer_Dict['Query'](inputs[0])
        v = self.layer_Dict['Value'](inputs[1])
        k = self.layer_Dict['Key'](inputs[2]) if len(inputs) > 2 else v
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k)
        if v_mask is not None:
            # Mask of shape [batch_size, 1, Tv].
            v_mask = tf.expand_dims(v_mask, axis=-2)
        if self.causal:
            # Creates a lower triangular mask, so position i cannot attend to
            # positions j>i. This prevents the flow of information from the future
            # into the past.
            scores_shape = tf.shape(scores)
            # causal_mask_shape = [1, Tq, Tv].
            causal_mask_shape = tf.concat(
                [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]],
                axis=0)
            causal_mask = _lower_triangular_mask(causal_mask_shape)
        else:
            causal_mask = None
        scores_mask = _merge_masks(v_mask, causal_mask)
        result, attention_distribution = _apply_scores(scores=scores, value=v, scores_mask=scores_mask)
        if q_mask is not None:
            # Mask of shape [batch_size, Tq, 1].
            q_mask = tf.expand_dims(q_mask, axis=-1)
            result *= tf.cast(q_mask, dtype=result.dtype)
        return result, attention_distribution

def _apply_scores(scores, value, scores_mask=None):
    if scores_mask is not None:
        padding_mask = tf.logical_not(scores_mask)
        # Bias so padding positions do not contribute to attention distribution.
        scores -= 1.e9 * tf.cast(padding_mask, dtype=tf.keras.backend.floatx())
    attention_distribution = tf.nn.softmax(scores)
    return tf.matmul(attention_distribution, value), attention_distribution

def _lower_triangular_mask(shape):
    """Creates a lower-triangular boolean mask over the last 2 dimensions."""
    row_index = tf.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_index = tf.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-1)
    return tf.greater_equal(row_index, col_index)

def _merge_masks(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return tf.logical_and(x, y)


# https://www.tensorflow.org/tutorials/text/nmt_with_attention
class BahdanauAttention2(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention2, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

