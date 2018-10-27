import time
import tensorflow as tf
import argparse
import pickle
import os
from model import Model
from data_utils import build_dict, build_dataset, batch_iter

# Uncomment next 2 lines to suppress error and Tensorflow info verbosity. Or change logging levels
# tf.logging.set_verbosity(tf.logging.FATAL)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
start = time.perf_counter()


def add_arguments(parser):
    parser.add_argument("--num_hidden", type=int, default=64, help="Network size.")
    parser.add_argument("--num_encoding_layers", type=int, default=2, help="encode network depth.")
    parser.add_argument("--num_decoding_layers", type=int, default=2, help="decode network depth.")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width for beam search decoder.")
    parser.add_argument("--clip", type=int, default=5, help="clipping gradients.")

    parser.add_argument("--embedding_type", type=str, default='glove', help="Use glove as initial word embedding.")
    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")

    parser.add_argument("--learning_rate_type", type=str, default='exponential', help="learning rate type.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Dropout keep prob.")
    parser.add_argument("--with_model", action="store_true", default=False, help="Continue from pre-saved model.")

    parser.add_argument("--article_max_len", type=int, default=150, help="maximum length of content.")
    parser.add_argument("--headline_max_len", type=int, default=20, help="maximum length of headline.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    # dumping train parameters
    hyper_params_path = './runtime_params/args.pickle'
    word2index_path = './runtime_params/word_dict.pickle'
    if not os.path.exists(os.path.dirname(hyper_params_path)):
        os.makedirs(os.path.dirname(hyper_params_path))
        with open(hyper_params_path, "wb") as f:
            pickle.dump(args, f)

    if not os.path.exists("seq2seq_model"):
        os.mkdir("seq2seq_model")
    else:
        if args.with_model:
            old_model_checkpoint_path = open('seq2seq_model/checkpoint', 'r')
            old_model_checkpoint_path = "".join(["seq2seq_model/",
                                                 old_model_checkpoint_path.read().splitlines()[0].split('"')[1]])

    print("Building dictionary...")
    word_dict, reversed_dict, article_list, headline_list = build_dict(train=True, word2index_path=word2index_path)
    print("Loading training dataset...")
    train_x, train_y = build_dataset(word_dict, article_list, args.article_max_len,
                                     headline_list=headline_list, headline_max_len=args.headline_max_len, train=True)

    with tf.Session() as sess:
        model = Model(word_dict, args)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if 'old_model_checkpoint_path' in globals():
            print("Continuing training from pre-trained model:", old_model_checkpoint_path, "......")
            saver.restore(sess, old_model_checkpoint_path)

        batches = batch_iter(train_x, train_y, args.batch_size, args.num_epochs)
        num_batches_per_epoch = (len(train_x) - 1) // args.batch_size + 1

        print("\nIteration starts.")
        print("Number of batches per epoch :", num_batches_per_epoch)
        for batch_x, batch_y in batches:

            # calculating actual length of input sequences
            batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))

            # adding the start tag(<s>) of decoding process in decoding sequences
            batch_decoder_input = list(map(lambda x: [word_dict["<s>"]] + list(x), batch_y))

            # calculating actual length of decoding sequences
            batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))

            # adding the end tag(</s>) of decoding process in decoding sequences
            batch_decoder_output = list(map(lambda x: list(x) + [word_dict["</s>"]], batch_y))

            # padding sequences
            batch_decoder_input = list(
                map(lambda d: d + (args.headline_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_input))
            batch_decoder_output = list(
                map(lambda d: d + (args.headline_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_output))

            train_feed_dict = {
                model.batch_size: len(batch_x),
                model.X: batch_x,
                model.X_len: batch_x_len,
                model.decoder_input: batch_decoder_input,
                model.decoder_len: batch_decoder_len,
                model.decoder_target: batch_decoder_output
            }

            _, step, loss = sess.run([model.train_op, model.global_step, model.loss], feed_dict=train_feed_dict)

            if step % 20 == 0:
                print("step {0}: loss = {1}".format(step, loss))

            if step % num_batches_per_epoch == 0:
                hours, rem = divmod(time.perf_counter() - start, 3600)
                minutes, seconds = divmod(rem, 60)
                saver.save(sess, "./seq2seq_model/model.ckpt", global_step=step)
                print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch),
                      "Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), "\n")
