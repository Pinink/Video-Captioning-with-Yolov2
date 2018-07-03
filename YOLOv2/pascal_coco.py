import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import yolo_config as cfg
from pycocotools.coco import COCO
import skimage.io as io
import pylab

class pascal_coco(object):
    def __init__(self, phase):
        self.dataDir='..'
        self.ImageDir = '/home/ubuntu/Documents/ZX/coco/images/'
        self.dataType='train2014'
        self.annFile='{}/annotations/instances_{}.json'.format(self.dataDir,self.dataType)
        self.coco=COCO(self.annFile)
        self.cache_path = cfg.CACHE_PATH
        self.data_name = 'COCO'
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
        gt_labels= self.load_labels()
        #np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
       
        return gt_labels

    def load_labels(self):

        cache_file = os.path.join(self.cache_path, self.data_name + self.phase +'label.pkl')
        #mAP_info_file = os.path.join(self.cache_path,self.data_name + self.phase  + 'mAP.pkl')

        if os.path.isfile(cache_file):
            print('Loading labels from ' + cache_file)
            print('Warning! if dataset has been changed!')
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Generating the labels...')

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

     

        gt_labels = []
        mAP_labels = []
        #cats = self.coco.loadCats(self.coco.getCatIds())
        #nms=[cat['name'] for cat in cats]
        #print('COCO categories: \n{}\n'.format(' '.join(nms)))
        #nms = set([cat['supercategory'] for cat in cats])
        #print('COCO supercategories: \n{}'.format(' '.join(nms)))
        #catIds = coco.getCatIds(nms);
        imgIds = self.coco.getImgIds();
        print('Load '+ str(len(imgIds)) + ' images')
        #print(imgIds)
        for i in range(len(imgIds)):
            if i <= 5:
                continue
            label, num = self.load_coco_annotation(imgIds[i])
            img = self.coco.loadImgs(imgIds[i])[0]
            #cv2.rectangle(img1, (int(x),int(y)), (int(x+width),int(y+height)),(255, 0, 0), 2)
            #label, num = self.load_pascal_annotation(index)
            #mAPlabel,numm = self.load_gt_for_mAP(index)
            if num == 0 :
                print(str(i)  + 'no object!~~~~~~~~~')
                continue
            #imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            
            imname = self.ImageDir +  img['file_name']
            gt_labels.append({'imname': imname,
                              'label': label,
                              'index':imgIds[i],
                              'flipped': False})
            
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_coco_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        img = self.coco.loadImgs(index)
        if len(img) != 1:
            print('error~~~~~~~~~' + str(len(img)))
            #print(img)
        img = img[0]
            #print(img)
            #print('/home/ubuntu/Documents/ZX/coco/images/' + img['file_name'])
            #img1 = cv2.imread('/home/ubuntu/Documents/ZX/coco/images/' + img['file_name'])
        shape = [img['height'],img['width']]
        annIds = self.coco.getAnnIds(imgIds=img['id'])
        anns = self.coco.loadAnns(annIds)
        #print(anns)
        label = np.zeros((self.cell_size, self.cell_size, 5,5))
        num_object = 0
        #img1 = cv2.imread('/home/ubuntu/Documents/ZX/coco/images/' + img['file_name'])
        for ann in anns:
            if 'bbox' in ann:
                num_object = num_object + 1
                x, y, width, height = ann['bbox']
                x1 = float(x)/shape[1]
                y1 = float(y)/shape[0]
                x2 = float(x + width)/shape[1]
                y2 = float(y + height)/shape[0]
                cls_ind =  ann["category_id"]
                x = (x1 + x2)/2.0
                y = (y1 + y2)/2.0
                w = x2 - x1
                h = y2 - y1
                coor = np.array([x,y,w,h]) * self.cell_size 
                #print(coor)
                index_x = int(coor[0])
                index_y = int(coor[1])
                coor[0] = coor[0] - index_x
                coor[1] = coor[1] - index_y
                max_iou = 0
                index_anchor = 0
                anchor_wh = [0,0]
                for j, anchor in enumerate(self.anchors):
                    iou = self.iou(coor, anchor)
                    if iou > max_iou :
                        max_iou = iou
                        index_anchor = j
                        anchor_wh = anchor
                #print(index_x,index_y,index_anchor)
                label[index_x,index_y,index_anchor,0:4] = coor
                label[index_x,index_y,index_anchor,4] = cls_ind - 1
                #x = (coor[0] + index_x) / self.cell_size * shape[1]
                #y = (coor[1] + index_y) / self.cell_size * shape[0]
                #width = coor[2] / self.cell_size * shape[1]
                #height = coor[3] / self.cell_size * shape[0]
                #xw = x - width/2
                #xd = x + width/2
                #yw = y - height/2
                #yd = y + height/2
                #img1 = cv2.rectangle(img1, (int(xw),int(yw)), (int(xd),int(yd)),(255, 0, 0), 2)
                #img1 = cv2.circle(img1,(int(coor[0] + index_x), int(coor[1] + index_y)), 63, (0,0,255), -1)
        #print(anns)
        #cv2.imshow('result.jpg',img1)
        #cv2.waitKey(-1)
        return label, num_object
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
