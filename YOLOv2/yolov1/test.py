import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc
import decode
import 
class Detector(object):

    def __init__(self, net,data, weight_file):
        self.net = net
        self.weights_file = weight_file
        self.data = data
        self.coconame = cfg.COCONAME
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        #print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        #self.saver.restore(self.sess, './model.ckpt')
        #ckpt = tf.train.get_checkpoint_state('/home/ubuntu/Documents/zx/DLProject/yolo_tensorflow/data/pascal_voc/output/IT4000')
        #self.saver = tf.train.import_meta_graph("/home/ubuntu/Documents/zx/DLProject/yolo_tensorflow/data/pascal_voc/output/IT1000/model.ckpt" +'.meta')
        print()
        #self.saver.restore(self.sess, "/home/ubuntu/Documents/zx/YOLO_small.ckpt")
        #self.saver.restore(self.sess,ckpt)
    def print_gt(self):
        print('______________________')
        print(str(len(self.data.gt_labels)))
        print(str(len(self.data.mAP_labels)))
        for img_i in range(len(self.data.gt_labels)):
            imname = self.data.gt_labels[img_i]['imname']
            #imname = 'test/cat.jpg'
            img = cv2.imread(imname)
            #result = self.detect(img)
            #img = self.data.image_read(imname, False)
            img_h, img_w, _ = img.shape
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img = (img / 255.0) * 2.0 - 1.0
            img = np.reshape(img, (1, self.image_size, self.image_size, 3))
            
            label = self.data.mAP_labels[img_i]['label']
            img_index = self.data.mAP_labels[img_i]['index']
            with open('./gt_info/' + str(img_index) + '.txt', 'w') as f:
                o = 0
                while label[o][0] != 0 and label[o][1] != 0 and label[o][2] != 0 and label[o][3] != 0: 
                    f.write(self.classes[int(label[o][4])])
                    f.write(' '+ str(int(label[o][0])) + ' ' + str(int(label[o][1])) + ' ' + str(int(label[o][2])) + ' ' + str(int(label[o][3])) + '\n')
                    o = o + 1
            feed_dict = {self.net.images: img}
            net_output = self.sess.run(self.net.logits,feed_dict=feed_dict)
            #print(net_output)
            result = self.interpret_output(net_output[0])
            #print(result)
            #print(len(result))
            for i in range(len(result)):
                result[i][1] *= (1.0 * img_w / self.image_size)
                result[i][2] *= (1.0 * img_h / self.image_size)
                result[i][3] *= (1.0 * img_w / self.image_size)
                result[i][4] *= (1.0 * img_h / self.image_size)
                name = result[i][0]
                porb = result[i][5]
                x = int(result[i][1])
                y = int(result[i][2])
                w = int(result[i][3] / 2)
                h = int(result[i][4] / 2)
                result[i][1] = x - w
                result[i][2] = y - h
                result[i][3] = x + w
                result[i][4] = y + h
            img_index = self.data.gt_labels[img_i]['index']
            with open('./pre_info/' + str(img_index) + '.txt', 'w') as f:
                for i in range(len(result)):
                    f.write(str(result[i][0]))#self.classes[int(result[i][0])])
                    f.write(' '+str(result[i][5]) + ' '+str(int(result[i][1])) + ' '+str(int(result[i][2])) + ' '+str(int(result[i][3])) + ' '+str(int(result[i][4])) + '\n')
    def draw_result(self, img, result):
        for i in range(len(result)):
            print(result)
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        #self.saver.save(self.sess, './model/IT5000+/model.ckpt')
        print(net_output)
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))
        # 7*7*2*20
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))
        #7*7*20
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
        #7*7*2
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        #7*7*2*4
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)
        offset = np.transpose(
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),
            (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):


        output_decoded = decode(model_output=model_output,output_sizes=output_sizes,
                               num_class=len(class_names),anchors=anchors)  # 解码

        bboxes,obj_probs,class_probs = sess.run(output_decoded,feed_dict={tf_image:image_cp})
    # 【2】筛选解码后的回归边界框——NMS(post process后期处理)
        image_cp = preprocess_image(image,input_size)
        bboxes,scores,class_max_index = postprocess(bboxes,obj_probs,class_probs,image_shape=image_shape)

    # 【3】绘制筛选后的边界框
        img_detection = draw_detection(image, bboxes, scores, class_max_index, class_names)
        cv2.imwrite("./yolo2_data/detection.jpg", img_detection)
        print('YOLO_v2 detection has done!')
        cv2.imshow("detection_results", img_detection)
        cv2.waitKey(0)

# 【1】图像预处理(pre process前期处理)
def preprocess_image(image,image_size=(416,416)):
    # 复制原图像
    image_cp = np.copy(image).astype(np.float32)

    # resize image
    image_rgb = cv2.cvtColor(image_cp,cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb,image_size)

    # normalize归一化
    image_normalized = image_resized.astype(np.float32) / 225.0

    # 增加一个维度在第0维——batch_size
    image_expanded = np.expand_dims(image_normalized,axis=0)

    return image_expanded

# 【2】筛选解码后的回归边界框——NMS(post process后期处理)
def postprocess(bboxes,obj_probs,class_probs,image_shape=(416,416),threshold=0.5):
    # bboxes表示为：图片中有多少box就多少行；4列分别是box(xmin,ymin,xmax,ymax)
    bboxes = np.reshape(bboxes,[-1,4])
    # 将所有box还原成图片中真实的位置
    bboxes[:,0:1] *= float(image_shape[1]) # xmin*width
    bboxes[:,1:2] *= float(image_shape[0]) # ymin*height
    bboxes[:,2:3] *= float(image_shape[1]) # xmax*width
    bboxes[:,3:4] *= float(image_shape[0]) # ymax*height
    bboxes = bboxes.astype(np.int32)

    # (1)cut the box:将边界框超出整张图片(0,0)—(415,415)的部分cut掉
    bbox_min_max = [0,0,image_shape[1]-1,image_shape[0]-1]
    bboxes = bboxes_cut(bbox_min_max,bboxes)

    # ※※※置信度*max类别概率=类别置信度scores※※※
    obj_probs = np.reshape(obj_probs,[-1])
    class_probs = np.reshape(class_probs,[len(obj_probs),-1])
    class_max_index = np.argmax(class_probs,axis=1) # 得到max类别概率对应的维度
    class_probs = class_probs[np.arange(len(obj_probs)),class_max_index]
    scores = obj_probs * class_probs

    # ※※※类别置信度scores>threshold的边界框bboxes留下※※※
    keep_index = scores > threshold
    class_max_index = class_max_index[keep_index]
    scores = scores[keep_index]
    bboxes = bboxes[keep_index]

    # (2)排序top_k(默认为400)
    class_max_index,scores,bboxes = bboxes_sort(class_max_index,scores,bboxes)
    # ※※※(3)NMS※※※
    class_max_index,scores,bboxes = bboxes_nms(class_max_index,scores,bboxes)

    return bboxes,scores,class_max_index

# 【3】绘制筛选后的边界框
def draw_detection(im, bboxes, scores, cls_inds, labels, thr=0.3):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x/float(len(labels)), 1., 1.)  for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    # draw image
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,(box[0], box[1]), (box[2], box[3]),colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)
        # cv2.rectangle(imgcv, (box[0], box[1]-20), ((box[0]+box[2])//3+120, box[1]-8), (125, 125, 125), -1)  # puttext函数的背景
        cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, (255,255,255), thick//3)
    return imgcv

######################## 对应【2】:筛选解码后的回归边界框#########################################
# (1)cut the box:将边界框超出整张图片(0,0)—(415,415)的部分cut掉
def bboxes_cut(bbox_min_max,bboxes):
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_min_max = np.transpose(bbox_min_max)
    # cut the box
    bboxes[0] = np.maximum(bboxes[0],bbox_min_max[0]) # xmin
    bboxes[1] = np.maximum(bboxes[1],bbox_min_max[1]) # ymin
    bboxes[2] = np.minimum(bboxes[2],bbox_min_max[2]) # xmax
    bboxes[3] = np.minimum(bboxes[3],bbox_min_max[3]) # ymax
    bboxes = np.transpose(bboxes)
    return bboxes

# (2)按类别置信度scores降序，对边界框进行排序并仅保留top_k
def bboxes_sort(classes,scores,bboxes,top_k=400):
    index = np.argsort(-scores)
    classes = classes[index][:top_k]
    scores = scores[index][:top_k]
    bboxes = bboxes[index][:top_k]
    return classes,scores,bboxes

# (3)计算IOU+NMS
# 计算两个box的IOU
def bboxes_iou(bboxes1,bboxes2):
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)

    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax-int_ymin,0.)
    int_w = np.maximum(int_xmax-int_xmin,0.)

    # 计算IOU
    int_vol = int_h * int_w # 交集面积
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1]) # bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1]) # bboxes2面积
    IOU = int_vol / (vol1 + vol2 - int_vol) # IOU=交集/并集
    return IOU
# NMS，或者用tf.image.non_max_suppression(boxes, scores,self.max_output_size, self.iou_threshold)
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]
###################################################################################################

def main():
    yolo = YOLONet(False)
    #weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    voc2007 = pascal_voc('val')
    # detect from camera
    detector = Detector(yolo,voc2007, ' ')
    cap = cv2.VideoCapture(-1)
    detector.camera_detector(cap)
    #detector.print_gt()
    # detect from image file
    #imname = 'test/cat4.jpg'
    #image = cv2.imread(imname)
    #inputs = cv2.resize(image, (448,448))
    #cv2.imwrite("./cat4.jpg",inputs)
    #detector.image_detector(imname)


if __name__ == '__main__':
    main()
