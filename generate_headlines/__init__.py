import tensorflow as tf
from tensorflow.contrib.seq2seq import *


t1 = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], shape=[2, 2, 3], dtype=tf.int32)
t2 = tf.constant([[[5, 5, 5], [6, 6, 6]], [[7, 7, 7], [8, 8, 8]]], shape=[2, 2, 3], dtype=tf.int32)

t1_t2_0 = tf.concat([t1, t2], 0)

t1_t2_1 = tf.concat([t1, t2], 1)

t1_t2_2 = tf.concat([t1, t2], 2)

t3 = tf.constant([[[1, 1, 0], [2, 0, 2]], [[0, 3, 3], [4, 0, 0]]], shape=[2, 2, 3], dtype=tf.int32)
t4 = tf.ones([2, 2, 3])
t5 = tf.ones_like(t4) * (-2 ** 32 + 1)
t6 = tf.where(tf.equal(t3, 0), t5, t4)
t7 = tf.ones_like(t3[0, :, :])
t8 = t4 * t5  # element-wise product
t9 = tf.tile(tf.expand_dims(tf.range(10), 0), [2, 1])

a1 = sequence_loss
a2 = Decoder
a3 = dynamic_decode
a4 = BasicDecoder
a5 = BasicDecoderOutput
a6 = BeamSearchDecoder
a7 = BeamSearchDecoderOutput
a8 = BeamSearchDecoderState
a9 = Helper
a10 = CustomHelper
a11 = FinalBeamSearchDecoderOutput
a12 = gather_tree

a13 = GreedyEmbeddingHelper
a14 = InferenceHelper
a15 = SampleEmbeddingHelper
a16 = ScheduledEmbeddingTrainingHelper
a17 = ScheduledOutputTrainingHelper
a18 = TrainingHelper

a19 = BahdanauAttention
a20 = LuongAttention
a21 = hardmax
a22 = AttentionWrapperState
a23 = AttentionWrapper
a24 = AttentionMechanism
a25 = tile_batch
a26 = safe_cumprod

a27 = monotonic_attention
a28 = BahdanauMonotonicAttention
a29 = LuongMonotonicAttention

h1 = tf.train.slice_input_producer
h2 = tf.train.shuffle_batch
h3 = tf.nn.moments
h4 = tf.layers.dense
h6 = tf.reduce_sum

h5 = tf.split  # 阶数减小
h7 = tf.tile   # 阶数不变
h8 = tf.where
h9 = tf.expand_dims  # 阶数增大
h10 = tf.equal
h12 = tf.layers.conv1d


with tf.Session() as sess:

    # print('-----------------------------')
    # print(sess.run(t1_t2_0), '\n', t1_t2_0.shape)
    #
    # print('-----------------------------')
    # print(sess.run(t1_t2_1), '\n', t1_t2_1.shape)
    #
    # print('-----------------------------')
    # print(sess.run(t1_t2_2), '\n', t1_t2_2.shape)
    #
    # print('-----------------------------')
    # print(sess.run(t6), '\n', t6.shape)

    print('-----------------------------')
    print(sess.run(t9), '\n', t9.shape)



