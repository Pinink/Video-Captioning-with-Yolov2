import numpy as np
import tensorflow as tf
import yolo_config as cfg
import os
from tensorflow.python import pywrap_tensorflow
slim = tf.contrib.slim


class YOLONet(object):

    def __init__(self, is_training=False):
        self.classes = cfg.CLASSES
        self.is_training = is_training
        self.num_class = 80#len(self.classes)
        self.num_anchor = cfg.NUM_ANCHOR
        self.anchors = cfg.ANCHORS
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        #self.output_size = (self.cell_size * self.cell_size) *\
            #(self.num_class + self.boxes_per_cell * 5)
            #7*7*(20+10)
        self.output_size = (self.num_class + 5) * self.num_anchor
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        #7*7*20
        self.use_pretrain = True
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell
        #7*7*20 + 7*7*2
        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE
        self.ckpt_file = './yolo2_data/yolo2_coco.ckpt'#cfg.CKPT_FILE
        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA
        self.lamda = cfg.LAMDA




        '''self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))'''
        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        self.logits = self.build_network(
            self.images, num_outputs = self.output_size, alpha=self.alpha,
            is_training=is_training)
        #self.img1
        '''if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, self.num_anchor,5])
            self.loss_layer(self.logits, self.labels)
            tf.add_to_collection("losses",tf.losses.get_total_loss())
            self.total_loss = tf.add_n(tf.get_collection("losses"))
            tf.summary.scalar('total_loss', self.total_loss)'''
        if is_training:
            self.class_loss = None
            self.object_loss = None
            self.noobject_loss = None
            self.coord_xy_loss =None
            self.coord_wh_loss = None
            self.coord_loss = None
            self.L2loss = tf.constant([0])
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, self.num_anchor,5])
            self.loss_layer(self.logits, self.labels)
            #self.L2loss = tf.add_n(tf.get_collection("losses"))
            #tf.add_to_collection("losses",tf.losses.get_total_loss())
            self.total_loss = tf.add_n(tf.get_collection("losses"))
            tf.summary.scalar('total_loss', self.total_loss)



    def build_network(self,images,num_outputs,alpha,keep_prob=0.5,is_training=True,scope='yolo'):
        with tf.variable_scope(scope):
                index = 0
                img = images
                img = self.Conv(img,32,3,1,1)
                #self.img1 = img
                img = self.Poolling(img,2)
                img = self.Conv(img,64,3,1,3)
                img = self.Poolling(img,4)
                img = self.Conv(img,128,3,1,5)
                img = self.Conv(img,64,1,0,6)
                img = self.Conv(img,128,3,1,7)
                img = self.Poolling(img,8)
                img = self.Conv(img,256,3,1,9)
                img = self.Conv(img,128,1,0,10)
                img = self.Conv(img,256,3,1,11)
                img = self.Poolling(img,12)
                img = self.Conv(img,512,3,1,13)
                img = self.Conv(img,256,1,0,14)
                img = self.Conv(img,512,3,1,15)
                img = self.Conv(img,256,1,0,16)
                img = self.Conv(img,512,3,1,17)
                res_img = img
                #res_img = tf.space_to_depth(img,block_size = 2,name = 'res_img')
                
                img = self.Poolling(img,18)
                img = self.Conv(img,1024,3,1,19)
                img = self.Conv(img,512,1,0,20)
                img = self.Conv(img,1024,3,1,21)
                img = self.Conv(img,512,1,0,22)
                img = self.Conv(img,1024,3,1,23)

                img = self.Conv(img,1024,3,1,24)
                img = self.Conv(img,1024,3,1,25)

                res_img = self.Conv(res_img,64,1,0,26)
                #res_img = self.reorg(res_img)
                res_img = tf.space_to_depth(res_img,block_size = 2,name = 'res_img')
                img = tf.concat([res_img,img],axis = -1,name = 'concat')
                
                img = self.Conv(img,1024,3,1,27)
                img = self.Conv(img,num_outputs,1,0,28,use_batch = False)
        return img

    def not_used_get_init(self,name,channels,out_size,size):
        if self.use_pretrain :
            reader = pywrap_tensorflow.NewCheckpointReader(self.ckpt_file)
            var_to_shape_map = reader.get_variable_to_shape_map()
            if 'conv' in name:
                num = ''.join([x for x in name if x.isdigit()])
                number = int(num)
                w_or_b = '  '
                if 'weight' in name:
                    w_or_b = 'weight'
                elif 'bias' in name:
                    w_or_b = 'bias'
                if number < 26 :
                    number = number + 1
                else :
                    number = number + 2
                for other_name in var_to_shape_map:
                    if 'conv' in other_name and w_or_b in other_name:
                        other_num = ''.join([x for x in other_name if x.isdigit()])
                        other_number = int(other_num)
                        if other_number == number:
                            result = tf.constant(reader.get_tensor(other_name))
                            print(result.get_shape())
                            if w_or_b == 'weight':
                                assert result.get_shape()[0] == size
                                assert result.get_shape()[1] == size
                                assert result.get_shape()[2] == channels
                                assert result.get_shape()[3] == out_size
                            else :
                                assert result.get_shape()[0] == out_size
                            return result
            if 'full' in name:
                num = ''.join([x for x in name if x.isdigit()])
                number = int(num)
                if number == 31:
                    number = 36
                else:
                    number = number + 4
                w_or_b = '  '
                if 'weight' in name:
                    w_or_b = 'weight'
                elif 'bias' in name:
                    w_or_b = 'bias'
                for other_name in var_to_shape_map:
                    if 'fc' in other_name and w_or_b in other_name:
                        other_num = ''.join([x for x in other_name if x.isdigit()])
                        other_number = int(other_num)
                        if other_number == number:
                            result = tf.constant(reader.get_tensor(other_name))
                            if w_or_b == 'weight' :
                                assert result.get_shape()[0] == channels
                                assert result.get_shape()[1] == out_size
                            else :
                                assert result.get_shape()[0] == out_size
                            return result
    def get_init(self,name,channels,out_size,size,iname = '0'):
        if self.use_pretrain :
            reader = pywrap_tensorflow.NewCheckpointReader(self.ckpt_file)
            var_to_shape_map = reader.get_variable_to_shape_map()
            index_name = "0_0"
            if name == 1:
                index_name ='1'
            elif name == 3:
                index_name = '2'
            elif name >= 5 and name <=7:
                index_name = '3_' + str(name - 4)
            elif name >= 9 and name <=11:
                index_name = '4_' + str(name - 8)
            elif name >= 13 and name <=17:
                index_name = '5_' + str(name -12)
            elif name >= 19 and name <=23:
                index_name = '6_' + str(name - 18)
            elif name  >=24 and name <= 25:
                index_name = '7_' + str(name - 23)
            elif name == 26:
                index_name = '_shortcut'
            elif name == 27:
                index_name ='8'
            elif name == 28:
                index_name = '_dec'
            total_name = 'conv' + index_name 

            for other_name in var_to_shape_map:
                #print(other_name)
                #print(total_name)
                if other_name == 'conv7_1_bn/beta':
                    self.img1 = tf.constant(reader.get_tensor(other_name))
                if iname == '0':
                    if other_name == total_name+ '/kernel':
                        print(total_name)
                        result = tf.constant(reader.get_tensor(other_name))
                        print(result.get_shape())
                        print(size)
                        assert result.get_shape()[0] == size
                        assert result.get_shape()[1] == size
                        assert result.get_shape()[2] == channels
                        assert result.get_shape()[3] == out_size
                        return result
                elif iname == 'b':
                    if other_name == total_name+ '/bias':
                        print(total_name)
                        result = tf.constant(reader.get_tensor(other_name))
                        print(result.get_shape())
                        print(size)
                        return result
                else:
                    if other_name == total_name+ iname:
                        print(total_name)
                        result = tf.constant(reader.get_tensor(other_name))
                        print(result.get_shape())
                        print(size)
                        return result

    def Conv(self,input,out_size,size,pad_size,name,stride = 1,use_batch = True):
        channels = input.get_shape()[-1]
        # [batch_size,h,w,channels]
        if self.use_pretrain:
            weight = tf.Variable(self.get_init(name,channels,out_size,size))
            #bias = tf.Variable(self.get_init('conv'+ str(name) + '/biases',channels,out_size,size))
        else:
            weight = tf.get_variable('conv' + str(name)+"/weight", shape=[size,size,channels,out_size],initializer=tf.contrib.layers.xavier_initializer())
            #bias = tf.get_variable('conv' + str(name)+"/bias", shape=[out_size],initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.lamda)(weight))
        #pad_size = size/2
        #if padding_size > 0 :
        #    input = tf.pad(input,np.array([[0,0],[padding_size,padding_size],[padding_size,padding_size],[0,0]]))
        #if pad_size > 0:
        #    input = tf.pad(input,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        output = tf.nn.conv2d(input,weight,strides = [1,stride,stride,1],padding = 'SAME',name = ('conv'+ str(name)))
        #output = tf.add(output,bias,name = ('conv'+str(name) + '_bias'))
        #print(channels)
        #print(output)
        if use_batch and self.use_pretrain:
            scale = tf.Variable(self.get_init(name,channels,out_size,size,iname = '_bn/gamma'))
            shift = tf.Variable(self.get_init(name,channels,out_size,size,iname = '_bn/beta'))
            mean = tf.Variable(self.get_init(name,channels,out_size,size,iname = '_bn/moving_mean'))
            variance = tf.Variable(self.get_init(name,channels,out_size,size,iname = '_bn/moving_variance'))
            output = tf.nn.batch_normalization(output, mean, variance, shift, scale, 1e-05)
            output = tf.maximum(self.alpha*output,output)
        elif use_batch:
            output = tf.layers.batch_normalization(output,axis=-1,momentum=0.9,training=self.is_training)
            output = tf.maximum(self.alpha*output,output)
        elif self.use_pretrain:
            bias =  tf.Variable(self.get_init(name,channels,out_size,size,iname = 'b'))
            #print('bias~~~~~~~~~~~~~~')
            output = tf.add(output,bias,name = ('conv'+str(name) + '_bias'))
        else :
            bias = tf.get_variable('conv' + str(name)+"/bias", shape=[out_size],initializer=tf.contrib.layers.xavier_initializer())
            output = tf.add(output,bias,name = ('conv'+str(name) + '_bias'))
        return output
    def Poolling(self,input,name,size=2,stride=2):
        output = tf.nn.max_pool(input,ksize = [1,size,size,1],strides = [1,stride,stride,1],padding = 'SAME',name = ('pool'+ str(name)))
        return output
    def Full(self,input,size,act,name):
        channels = input.get_shape()[-1]
        if self.use_pretrain:
            weight = tf.Variable(self.get_init('full' + str(name) + '/weights',channels,size,0))
            bias = tf.Variable(self.get_init('full' + str(name) + '/biases',channels,size,0))
        else :
            weight = tf.get_variable('fullc' + str(name)+"/weight", shape=[channels,size],initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('fullc' + str(name)+"/bias", shape=[size],initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.lamda)(weight))
        output = tf.matmul(input,weight)
        output = tf.add(output,bias)
        if act == 1:
            output = tf.maximum(self.alpha*output,output)
        return output
    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, pred, labels):

        pred_size = tf.shape(pred)
        pred_lenth =  pred_size[0] * pred_size[1] * pred_size[2]
        pred = tf.reshape(pred,(pred_size[0],pred_size[1],pred_size[2],self.num_anchor,self.num_class + 5))
        # batch * h * w * 5 * 25
        pred_box = pred[:,:,:,:,0:4]
        pred_confidence = pred[:,:,:,:,4]
        pred_confidence = tf.sigmoid(pred_confidence)
        #self.img2 = pred_confidence
        # one-hot
        pred_class = pred[:,:,:,:,5:]
        pred_class = tf.nn.softmax(pred_class)
        #self.img1 = tf.reduce_max(pred_class)
        #self.img2 = pred_class
        #pred_class = tf.reshape()
        truth_box = labels[:,:,:,:,0:4]
        truth_class_int = labels[:,:,:,:,4]
        object_mask = tf.reduce_sum(labels, 4)
        object_mask = tf.cast(object_mask > 0, tf.float32)
        ###########################################
        pred_xy = tf.sigmoid(pred_box[:,:,:,:,0:2])
        anchor_w = tf.constant(self.anchors[:,0],dtype = 'float32')
        anchor_h = tf.constant(self.anchors[:,1],dtype = 'float32')

        anchor_w = tf.tile(anchor_w,[pred_lenth])
        anchor_h = tf.tile(anchor_h,[pred_lenth])

        anchor_w = tf.reshape(anchor_w,(pred_size[0],pred_size[1],pred_size[2],self.num_anchor))
        anchor_h = tf.reshape(anchor_h,(pred_size[0],pred_size[1],pred_size[2],self.num_anchor),)
        pred_w = tf.clip_by_value(pred_box[:,:,:,:,2], -10.0, 10.0)
        pred_h = tf.clip_by_value(pred_box[:,:,:,:,3], -10.0, 10.0)
        #pred_w = pred_box[:,:,:,:,2]
        #pred_h = pred_box[:,:,:,:,3]
        self.img1 = pred_w
        self.img2 = tf.clip_by_value(pred_box[:,:,:,:,2], -10.0, 10.0)
        pred_w = tf.expand_dims(tf.exp(pred_w) * anchor_w, 4)
        pred_h = tf.expand_dims(tf.exp(pred_h) * anchor_h, 4)
        pred_box = tf.concat([pred_xy,pred_w,pred_h],4)
        #box_coordinate = tf.concat([box_coordinate_xy, box_coordinate_w, box_coordinate_h], 4)
        ######################################################################
        #mask = tf.cast(truth_class_int > 0, tf.float32)
        mask = object_mask
        #self.img1 = tf.reduce_sum(mask)
        n_mask = 1 - mask
        
        #self.img2 = tf.reduce_sum(n_mask)
        batch_size_float32 = tf.cast(self.batch_size,tf.float32)
        ##########Coord_loss###########################################################
        ### May be error because of type !!
        dwh = tf.square(tf.sqrt(pred_box[:,:,:,:,2:4]) - tf.sqrt(truth_box[:,:,:,:, 2:4]))
        #dwh = tf.square(pred_box[:,:,:,:,2:4] - truth_box[:,:,:,:, 2:4])
        dwh = mask * tf.reduce_sum(dwh,4)
        dxy = mask * tf.reduce_sum(tf.square(pred_xy - truth_box[:,:,:,:,0:2]),4)
        coord_xy_loss = self.coord_scale * tf.reduce_sum(dxy,name = 'coord_xy_loss') / batch_size_float32
        coord_wh_loss = self.coord_scale * tf.reduce_sum(dwh,name = 'coord_wh_loss') / batch_size_float32
        coord_loss = coord_xy_loss + coord_wh_loss
        ################################################################################
        ##########Class_loss############################################################
        truth_class_int = tf.cast(truth_class_int,tf.int32)
        truth_calss = tf.one_hot(truth_class_int,self.num_class)

        
        dclass = mask * tf.reduce_sum(tf.square(pred_class - truth_calss),4)
        class_loss = self.class_scale * tf.reduce_sum(dclass) / batch_size_float32
        ################################################################################
        ##########Confidence_loss############################################################
        pred_iou = self.calc_iou(pred_box,truth_box)
        noobject_loss = self.noobject_scale * tf.reduce_sum(n_mask * tf.square(0- pred_confidence)) / batch_size_float32
        object_loss =  self.object_scale *tf.reduce_sum(mask * tf.square(pred_iou - pred_confidence)) / batch_size_float32
        
        ################################################################################
        self.coord_xy_loss = coord_xy_loss
        self.coord_wh_loss = coord_wh_loss
        self.class_loss = class_loss
        self.object_loss = object_loss
        self.noobject_loss = noobject_loss
        self.coord_loss = coord_loss
        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)

        tf.summary.scalar('class_loss', class_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.scalar('noobject_loss', noobject_loss)
        tf.summary.scalar('coord_loss', coord_loss)

        tf.summary.histogram('boxes_delta_x', pred_box[..., 0])
        tf.summary.histogram('boxes_delta_y', pred_box[..., 1])
        tf.summary.histogram('boxes_delta_w', pred_box[..., 2])
        tf.summary.histogram('boxes_delta_h', pred_box[..., 3])
        tf.summary.histogram('iou', pred_iou)