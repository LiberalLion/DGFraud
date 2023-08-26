import tensorflow as tf

class Algorithm(object):
    def __init__(self, **kwargs):
        self.nodes = None

    def forward_propagation(self):
        pass

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver()
        save_path = saver.save(sess, 'tmp/temp.ckpt')
        print(f"Model saved in file: {save_path}")

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver()
        save_path = 'tmp/temp.ckpt'
        saver.restore(sess, save_path)
        print(f"Model restored from file: {save_path}")