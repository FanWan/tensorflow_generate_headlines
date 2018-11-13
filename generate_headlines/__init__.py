import tensorflow as tf
from tensorflow.contrib.seq2seq import *


t1 = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], shape=[2, 2, 3], dtype=tf.int32)
t2 = tf.constant([[[5, 5, 5], [6, 6, 6]], [[7, 7, 7], [8, 8, 8]]], shape=[2, 2, 3], dtype=tf.int32)


t1_t2_0 = tf.concat([t1, t2], 0)

t1_t2_1 = tf.concat([t1, t2], 1)

t1_t2_2 = tf.concat([t1, t2], 2)

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


def test_data_valid():

    pass


with tf.Session() as sess:

    print('-----------------------------')
    print(sess.run(t1_t2_0), '\n', t1_t2_0.shape)

    print('-----------------------------')
    print(sess.run(t1_t2_1), '\n', t1_t2_1.shape)

    print('-----------------------------')
    print(sess.run(t1_t2_2), '\n', t1_t2_2.shape)

