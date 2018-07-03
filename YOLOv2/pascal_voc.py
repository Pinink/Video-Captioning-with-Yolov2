import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import yolo.config as cfg


class pascal_voc(object):
    def __init__(self, phase):
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        self.data_name = 'VOC2012'
        self.data_path = os.path.join(self.devkil_path, self.data_name)
        self.object_number = 60
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = int(self.image_size / 32)
        self.num_anchor = cfg.NUM_ANCHOR
        self.anchors = cfg.ANCHORS
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.flipped = cfg.FLIPPED
        self.phase = phase
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.mAP_labels = None
        self.prepare()

    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        # batch 13,13,5,5[x ,y zai 13*13 d cell [0,1]]
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, self.num_anchor,5))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):
        #print(imname)
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        return image

    def prepare(self):
        gt_labels,mAP_labels= self.load_labels()
        #np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        self.mAP_labels = mAP_labels
        return gt_labels ,mAP_labels

    def load_labels(self):

        cache_file = os.path.join(self.cache_path, self.data_name + self.phase +'label.pkl')
        mAP_info_file = os.path.join(self.cache_path,self.data_name + self.phase  + 'mAP.pkl')

        if os.path.isfile(cache_file) and os.path.isfile(mAP_info_file):
            print('Loading labels from ' + cache_file)
            print('Warning! if dataset has been changed!')
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            with open(mAP_info_file,'rb') as mf:
                mAP_labels = pickle.load(mf)
            return gt_labels,mAP_labels

        print('Generating the labels...')

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'train.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'val.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        mAP_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            mAPlabel,numm = self.load_gt_for_mAP(index)
            if num == 0 or numm == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname,
                              'label': label,
                              'index':index,
                              'flipped': False})
            mAP_labels.append({'imname':imname,
                                'index':index,
                                'label':mAPlabel})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        with open(mAP_info_file,'wb') as mf:
            pickle.dump(mAP_labels,mf)
        return gt_labels ,mAP_labels

    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)

        #h_ratio = 1.0 * self.image_size / im.shape[0]
        #w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])
        # 13*13*5*5[x y w h label]
        #print(self.cell_size)
        label = np.zeros((self.cell_size, self.cell_size, 5,5))

        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            #!!! x/shape[1] y/shape[0]
            x1 = float(bbox.find('xmin').text)/im.shape[1] 
            y1 = float(bbox.find('ymin').text)/im.shape[0] 
            x2 = float(bbox.find('xmax').text)/im.shape[1] 
            y2 = float(bbox.find('ymax').text)/im.shape[0]

            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            
            x = (x1 + x2)/2.0
            y = (y1 + y2)/2.0
            w = x2 - x1
            h = y2 - y1
            coor = np.array([x,y,w,h]) * self.cell_size 
            '''print(x)
            print('--------')
            print(y)
            print('--------------')
            print(w)
            print('-------------------')
            print(h)
            print('---------------------------')'''
            #boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            index_x = int(coor[0])
            coor[0] -= index_x
            coor[1] -= index_y
            #!!!!!
            index_y = int(coor[1])
            max_iou = 0
            index_anchor = 0
            anchor_wh = [0,0]
            for j, anchor in enumerate(self.anchors):
                iou = self.iou(coor, anchor)
                if iou > max_iou :
                    max_iou = iou
                    index_anchor = j
                    anchor_wh = anchor
        # label[w][h][anchor_index][5]
            #print(coor)
            #print(coor.shape)
            label[index_x,index_y,index_anchor,0:4] = coor
            label[index_x,index_y,index_anchor,4] = cls_ind

        return label, len(objs)
    def iou(self,coor, anchor):
        ax1 = 0
        ay1 = 0
        ax2 = coor[2]
        ay2 = coor[3]
        bx1 = 0
        by1 = 0
        bx2 = anchor[0]
        by2 = anchor[1]
        lu = [0,0]
        rd = np.minimum([ax2,ay2],[bx2,by2])

        inter = np.maximum(0.0, rd - lu)
        inter_square = inter[..., 0] * inter[..., 1]

            # calculate the boxs1 square and boxs2 square
        square1 = coor[2] * coor[3]
        square2 = anchor[0] * anchor[1]

        union_square = np.maximum(square1 + square2 - inter_square, 1e-10)

        return np.clip(inter_square / union_square, 0.0, 1.0)

    def load_gt_for_mAP(self,index):

        label = np.zeros((self.object_number,5))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        object_t = tree.findall('object')
        for i, object_i in enumerate(object_t):
            bbox = object_i.find('bndbox')
            #print(i)
            #print(filename)

            label[i][0] = int(bbox.find('xmin').text) 
            label[i][1] = int(bbox.find('ymin').text)
            label[i][2] = int(bbox.find('xmax').text)
            label[i][3] = int(bbox.find('ymax').text)
            label[i][4] = self.class_to_ind[object_i.find('name').text.lower().strip()]
        return label, len(object_t)
