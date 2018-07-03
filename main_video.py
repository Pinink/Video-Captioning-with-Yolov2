#!/usr/bin/python
import tensorflow as tf
import cv2
import os
import time
from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_test_data
import numpy  as np
FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'test',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', True,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', './models/112999.npy',
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

def main(argv):
    config = Config()
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size

    with tf.Session() as sess:
        if FLAGS.phase == 'train':
            # training phase
            data = prepare_train_data(config)
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            tf.get_default_graph().finalize()
            model.train(sess, data)
        else:
            # testing phase
            cap = cv2.VideoCapture(0)
            #cap = cv2.VideoCapture('./video3.mp4')
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            i = 1
            vocabulary = prepare_test_data(config)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if i == 1 or i % 4 == 0:
                    caption = model.test(sess,frame,vocabulary)
                    
                i += 1
                word_and_img = np.concatenate((np.zeros((50,np.shape(frame)[1], 3), np.uint8),frame),axis = 0)
                cv2.putText(word_and_img,caption,(15,30),cv2.FONT_HERSHEY_TRIPLEX,0.5,(18,87,220),1)
                cv2.imshow('VideoShow', word_and_img)
                cv2.waitKey(5)

if __name__ == '__main__':
    tf.app.run()
