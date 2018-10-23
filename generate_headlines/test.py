import tensorflow as tf
import pickle
from model import Model
from data_utils import build_dict, build_dataset, batch_iter


with open("args.pickle", "rb") as f:
    args = pickle.load(f)

print("Loading dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict(train=False)
print("Loading validation dataset...")
valid_x = build_dataset(word_dict, article_max_len, summary_max_len, train=False)
valid_x_len = [len([y for y in x if y != 0]) for x in valid_x]

with tf.Session() as sess:
    print("Loading saved model...")
    model = Model(reversed_dict, article_max_len, summary_max_len, args, train=False)
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state("./seq2seq_model/")
    saver.restore(sess, ckpt.model_checkpoint_path)

    # fi
    batches = batch_iter(valid_x, [0] * len(valid_x), args.batch_size, 1)

    print("Writing summaries to 'result.txt'...")
    for batch_x, _ in batches:
        batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]

        valid_feed_dict = {
            model.batch_size: len(batch_x),
            model.X: batch_x,
            model.X_len: batch_x_len,
        }

        prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
        prediction_output = [[reversed_dict[y] for y in x] for x in prediction[:, 0, :]]

        with open("result.txt", "a") as f:
            for line in prediction_output:
                summary = list()
                for word in line:
                    if word == "</s>":
                        break
                    if word not in summary:
                        summary.append(word)
                print(" ".join(summary), file=f)

    print('Summaries are saved to "result.txt"...')
