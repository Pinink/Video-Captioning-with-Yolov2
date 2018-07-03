from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import cv2
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
dataDir='..'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(nms);
imgIds = coco.getImgIds();

img = coco.loadImgs(imgIds[5])
#print(img)
img = img[0]
#print(img)
print('/home/ubuntu/Documents/ZX/coco/images/' + img['file_name'])
img1 = cv2.imread('/home/ubuntu/Documents/ZX/coco/images/' + img['file_name'])
#print(img1)
#cv2.imshow('result.jpg',img1)
annIds = coco.getAnnIds(imgIds=img['id'])
anns = coco.loadAnns(annIds)
for ann in anns:
    if 'bbox' in ann:
        if ann['iscrowd']==0:
            x, y, width, height = ann['bbox']
            print(x)
            print(y)
            print(width)
            print(height)
            cv2.rectangle(img1, (int(x),int(y)), (int(x+width),int(y+height)),(255, 0, 0), 2)
            
cv2.imshow('s.jpg',img1)
cv2.waitKey(0)