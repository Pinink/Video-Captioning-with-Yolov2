
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from yolo_net import YOLONet
from decode import decode
from utils import preprocess_image, postprocess, draw_detection
import yolo_config as cfg

def main():
    anchors = cfg.ANCHORS
    class_names = cfg.COCONAME
    input_size = (416,416)
    image_file = './yolo2_data/car.jpg'
    image = cv2.imread(image_file)
    image_shape = image.shape[:2] 
    tf_image = tf.placeholder(tf.float32,[1,input_size[0],input_size[1],3])
    net = YOLONet(False)
    output_sizes = input_size[0]//32, input_size[1]//32 
    output_decoded = decode(model_output=net.logits,output_sizes=output_sizes,
                               num_class=len(class_names),anchors=anchors)  
    image_cp = preprocess_image(image,input_size)
    model_path = "./model.ckpt"
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,model_path)
        #bboxes,obj_probs,class_probs = sess.run(output_decoded,feed_dict={tf_image:image_cp})
        bboxes,obj_probs,class_probs = sess.run(output_decoded,feed_dict={net.images: image_cp})
        #img1= sess.run([net.img1],feed_dict={net.images: image_cp})
        #print(img1)
        #print(img2)
        #print(obj_probs)
        #print(sess.run(tf.reduce_max(obj_probs),feed_dict={net.images: image_cp}))
        bboxes,scores,class_max_index = postprocess(bboxes,obj_probs,class_probs,image_shape=image_shape)
        print(scores)
        print(class_max_index)
        img_detection = draw_detection(image, bboxes, scores, class_max_index, class_names)
    #cv2.imwrite("./yolo2_data/detection.jpg", img_detection)
    #print('YOLO_v2 detection has done!')
        cv2.imshow("detection_results", img_detection)
        cv2.waitKey(-1)

if __name__ == '__main__':
    main()
