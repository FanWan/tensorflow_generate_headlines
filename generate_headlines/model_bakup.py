import tensorflow as tf
from tensorflow.contrib import rnn
from data_utils import *


class Model(object):
    def __init__(self, word2index, article_max_len, summary_max_len, args,
                 train=True,
                 fixed_rate=0.001,
                 learning_rate_decay=0.9,
                 learning_rate_decay_steps=100,
                 max_lr=0.01):
        """
        :param word2index: word-index pair
        :param article_max_len: maximum length of input sequences
        :param summary_max_len: maximum length of headline sequences
        :param args: hparameters of the model
        :param train: true--training, false--validating
        """
        self.vocabulary_size = len(word2index)
        self.embedding_size = args.embedding_size
        self.embedding_type = args.embedding_type
        self.word2index = word2index
        self.summary_max_len = summary_max_len

        self.num_hidden = args.num_hidden
        self.num_encoding_layers = args.num_encoding_ayers
        self.num_decoding_layers = args.num_decoding_layers
        self.beam_width = args.beam_width
        self.clip = args.clip

        self.learning_rate_type = args.learning_rate_type
        self.fixed_rate = fixed_rate
        self.learning_rate_decay = learning_rate_decay
        self.max_lr = max_lr
        self.learning_rate_decay_steps = learning_rate_decay_steps

        # setting dropout
        if train:
            self.keep_prob = args.keep_prob
        else:
            self.keep_prob = 1.0

        # adding a full-connected layer to decoding process
        with tf.variable_scope("decoder/projection"):
            self.projection_layer = tf.layers.Dense(self.vocabulary_size, use_bias=False)

        # adding placeholders
        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        self.X = tf.placeholder(tf.int32, [None, article_max_len])
        self.X_len = tf.placeholder(tf.int32, [None])
        self.decoder_input = tf.placeholder(tf.int32, [None, summary_max_len])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_target = tf.placeholder(tf.int32, [None, summary_max_len])
        self.global_step = tf.Variable(0, trainable=False)

        self.encoder_emb_inp, self.decoder_emb_inp, self.embeddings = self.add_emddings()

        self.encoder_output, self.encoder_state = self.build_encoder()

        self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell(train)

        self.logits, self.sample_id, self.final_state = self.decode_process(train)

        if train:
            with tf.name_scope("loss"):
                logits_projection = tf.transpose(self.logits, perm=[1, 0, 2])
                logits_reshape = tf.concat([logits_projection,
                                            tf.zeros([self.batch_size,
                                                      self.summary_max_len - tf.shape(self.logits_reshape)[1],
                                                      self.vocabulary_size])], axis=1)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_reshape,
                                                                               labels=self.decoder_target)
                weights = tf.sequence_mask(self.decoder_len, self.summary_max_len, dtype=tf.float32)
                self.loss = tf.reduce_sum(cross_entropy * weights / tf.to_float(self.batch_size))

                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.clip)

                if self.learning_rate_type == 'cyclic':
                    self.learning_rate = self.triangular_lr(self.global_step)
                elif self.learning_rate_type == 'exponential':
                    self.learning_rate = tf.train.exponential_decay(
                        self.fixed_rate,
                        self.global_step,
                        decay_steps=self.learning_rate_decay_steps,
                        decay_rate=self.learning_rate_decay,
                        staircase=True)
                else:
                    self.learning_rate = fixed_rate

                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        else:
            self.prediction = tf.transpose(self.sample_id, perm=[1, 0, 2])

    def add_emddings(self):
        """
            creating the embedding matrix: pre-trained glove, word2vec or training word2vec
        """
        with tf.name_scope("embedding"):
            if self.embedding_type == 'glove':
                init_embeddings = tf.constant(get_glove_embedding(self.word2index,
                                                                  self.embedding_size), dtype=tf.float32)
            elif self.embedding_type == 'word2vec':
                init_embeddings = tf.constant(get_word2vec_embedding(self.word2index,
                                                                     self.embedding_size), dtype=tf.float32)
            else:
                init_embeddings = tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            print('loading embedding matrix')
            # transpose input matrix to the dimension of [max_time, batch_size, embedding_size]
            encoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(embeddings, self.X), perm=[1, 0, 2])
            decoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(embeddings, self.decoder_input), perm=[1, 0, 2])
            return encoder_emb_inp, decoder_emb_inp, embeddings

    def build_encoder(self):
        """
            creating Bidirectional-LSTM encoding cells
        """
        with tf.name_scope("encoder"):
            fw_cells = [self.make_rnn_cell(self.num_hidden) for _ in range(self.num_encoding_layers)]
            bw_cells = [self.make_rnn_cell(self.num_hidden) for _ in range(self.num_encoding_layers)]

            encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells,
                bw_cells,
                self.encoder_emb_inp,
                sequence_length=self.X_len,
                time_major=True,
                dtype=tf.float32)

            # concat output and state of bidirectional cell
            encoder_output = tf.concat(encoder_outputs, 2)
            encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
            encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
            encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            return encoder_output, encoder_state

    def build_decoder_cell(self, train):
        """
           building the attention decoder cell. If train is false, performs tiling
           Passes last encoder state.
        """
        with tf.variable_scope("decoder_cell"):
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.make_rnn_cell(self.num_hidden * 2)
                                                        for _ in range(self.num_decoding_layers)])
            if not train:
                memory = tf.contrib.seq2seq.tile_batch(
                    tf.transpose(self.encoder_output, perm=[1, 0, 2]), multiplier=self.beam_width)
                batch_size = self.batch_size * self.beam_width
                initial_state = tf.contrib.seq2seq.tile_batch(self.encoder_state,
                                                              multiplier=self.beam_width)
                memory_sequence_length = tf.contrib.seq2seq.tile_batch(self.X_len,
                                                                       multiplier=self.beam_width)
            else:
                memory = tf.transpose(self.encoder_output, perm=[1, 0, 2])
                batch_size = self.batch_size
                memory_sequence_length = self.X_len
                initial_state = self.encoder_state
            decoder_cell = self.wrap_attention(decoder_cell, self.num_hidden * 2, memory, memory_sequence_length, True)
            decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=initial_state)
            return decoder_cell, decoder_initial_state

    def decode_process(self, train):
        """
            if train is true, using TrainingHelper and BasicDecoder,
            otherwise, using BeamSearchDecoder.
        """
        with tf.name_scope("decode_process") as decoder_scope:
            if train:
                helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_inp,
                                                           self.decoder_len,
                                                           time_major=True,
                                                           name="trainingHelper")
                decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell,
                                                          helper,
                                                          self.decoder_initial_state,
                                                          output_layer=self.projection_layer)
                outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                            output_time_major=True,
                                                                            scope=decoder_scope)
                logits = outputs.rnn_output
                sample_id = outputs.sample_id
            else:
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=self.decoder_cell,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                    end_token=tf.constant(3),
                    initial_state=self.decoder_initial_state,
                    beam_width=self.beam_width,
                    output_layer=self.projection_layer
                )
                outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    output_time_major=True,
                    maximum_iterations=self.summary_max_len,
                    scope=decoder_scope)
                logits = tf.no_op()
                sample_id = outputs.predicted_ids
            return logits, sample_id, final_state

    def make_rnn_cell(self, num_units):
        """
            creating LSTM cell wrapped with dropout.
        """
        cell = tf.nn.rnn_cell.LSTMCell(num_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
        return cell

    def wrap_attention(self, decode_cell, num_units, memory, memory_sequence_length, normalize):
        """
            wraping the given cell with Bahdanau Attention.
        """
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units,
                                                                   memory,
                                                                   memory_sequence_length=memory_sequence_length,
                                                                   name='BahdanauAttention',
                                                                   normalize=normalize)
        return tf.contrib.seq2seq.AttentionWrapper(decode_cell,
                                                   attention_mechanism,
                                                   attention_layer_size=num_units)

    def triangular_lr(self, current_step):
        """
            defining cyclic learning rate.
        """
        step_size = self.learning_rate_decay_steps
        base_lr = self.fixed_rate
        max_lr = self.max_lr

        cycle = tf.floor(1 + current_step / (2 * step_size))
        x = tf.abs(current_step / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * tf.maximum(0.0, tf.cast((1.0 - x), dtype=tf.float32)) * (0.99999 ** tf.cast(
            current_step, dtype=tf.float32))
        return lr
