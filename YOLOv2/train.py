import os
import cv2
import numpy as np
import argparse
import datetime
import tensorflow as tf
import yolo_config as cfg
from yolo_net import YOLONet
from pascal_coco import pascal_coco
from decode import decode
from utils import preprocess_image, postprocess, draw_detection
from config import anchors, class_names

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        self.ckpt_file = self.output_dir + '/model.ckpt'
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        #gpu_options = tf.GPUOptions()
        #config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        #if self.weights_file is not None:
         #   print('Restoring weights from: ' + self.weights_file)
          #  self.saver.restore(self.sess, self.weights_file)
        self.saver.restore(self.sess, "./model.ckpt")
        self.writer.add_graph(self.sess.graph)

    def train(self):

       
        last_score = 0.0
        for step in range(1, self.max_iter + 1):
            print("cur step:" + str(step))
            
            images, labels = self.data.get()
       
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            #if step % self.summary_iter == 0:
            #    summary_str, loss, _ = self.sess.run([self.summary_op, self.net.total_loss, self.train_op],feed_dict=feed_dict)
            #    print('Cur_loss : '+ str(loss))
            if step % self.summary_iter == 0:
                summary_str, loss = self.sess.run([self.summary_op, self.net.total_loss],feed_dict=feed_dict)
                print('Cur_loss : '+ str(loss))
                coord_xy_loss,coord_wh_loss,class_loss,object_loss,noobject_loss,coord_loss,L2loss = self.sess.run([self.net.coord_xy_loss,self.net.coord_wh_loss,self.net.class_loss,self.net.object_loss,self.net.noobject_loss,self.net.coord_loss,self.net.L2loss],feed_dict = feed_dict)
                img1 = self.sess.run([self.net.img1],feed_dict = feed_dict)
                #print('mask : ' + str(img1))
                print('Coord_xy_loss :' + str(coord_xy_loss))
                print('Coord_wh_loss :' + str(coord_wh_loss))
                print('Class_loss : ' + str(class_loss))
                print('Object_loss : ' + str(object_loss))
                print('Noobject_loss : ' + str(noobject_loss))
                print('Coord_loss : ' + str(coord_loss))
                print('L2_loss : ' + str(L2loss))
            else:
                self.sess.run(self.train_op, feed_dict=feed_dict)
                #img1,img2 = self.sess.run([self.net.img1,self.net.img2],feed_dict = feed_dict)
                #print(img1)
                #print(img2)

            if step % 10 == 0:
                input_size = (416,416)
                image_file = './yolo2_data/train.jpg'
                image = cv2.imread(image_file)
                image_shape = image.shape[:2] #只取wh，channel=3不取
                tf_image = tf.placeholder(tf.float32,[1,input_size[0],input_size[1],3])
                output_sizes = input_size[0]//32, input_size[1]//32 # 特征图尺寸是图片下采样32倍
                output_decoded = decode(model_output=self.net.logits,output_sizes=output_sizes,num_class=len(class_names),anchors=anchors)  # 解码
                image_cp = preprocess_image(image,input_size)
                bboxes,obj_probs,class_probs = self.sess.run(output_decoded,feed_dict={self.net.images: image_cp})
                #print(obj_probs)
                bboxes,scores,class_max_index = postprocess(bboxes,obj_probs,class_probs,image_shape=image_shape)

                #if len(scores) > 0 and last_score[0] < scores[0]:
                #    if scores[0] > 0.6:
                #        self.saver.save(self.sess, './data/pascal_voc/tempweight/model.ckpt')
                #        last_score = scores
                print(scores)
                print(class_max_index)

                #print('img2 :' + str(img2))
            if step %  self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

def main():

    

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

   # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet(True)
    pascal = pascal_coco('train')

    solver = Solver(yolo, pascal)

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
