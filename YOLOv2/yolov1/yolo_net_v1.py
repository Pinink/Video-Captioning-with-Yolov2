import numpy as np
import tensorflow as tf
import yolo.config as cfg
import os
from tensorflow.python import pywrap_tensorflow
slim = tf.contrib.slim


class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.is_training = is_training
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.output_size = (self.cell_size * self.cell_size) *\
            (self.num_class + self.boxes_per_cell * 5)
            #7*7*(20+10)
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
        self.ckpt_file = cfg.CKPT_FILE
        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA
        self.lamda = cfg.LAMDA
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            tf.add_to_collection("losses",tf.losses.get_total_loss())
            self.total_loss = tf.add_n(tf.get_collection("losses"))
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,images,num_outputs,alpha,keep_prob=0.5,is_training=True,scope='yolo'):
        with tf.variable_scope(scope):
                index = 0
                img = images
                img = self.Conv(img,64,7,2,1)
                img = self.Poolling(img,2,2,2)
                img = self.Conv(img,192,3,1,3)
                img = self.Poolling(img,2,2,4)
                img = self.Conv(img,128,1,1,5)
                img = self.Conv(img,256,3,1,6)
                img = self.Conv(img,256,1,1,7)
                img = self.Conv(img,512,3,1,8)
                img = self.Poolling(img,2,2,9)
                img = self.Conv(img,256,1,1,10)
                img = self.Conv(img,512,3,1,11)
                img = self.Conv(img,256,1,1,12)
                img = self.Conv(img,512,3,1,13)
                img = self.Conv(img,256,1,1,14)
                img = self.Conv(img,512,3,1,15)
                img = self.Conv(img,256,1,1,16)
                img = self.Conv(img,512,3,1,17)
                img = self.Conv(img,512,1,1,18)
                img = self.Conv(img,1024,3,1,19)
                img = self.Poolling(img,2,2,20)
                img = self.Conv(img,512,1,1,21)
                img = self.Conv(img,1024,3,1,22)
                img = self.Conv(img,512,1,1,23)
                img = self.Conv(img,1024,3,1,24)
                img = self.Conv(img,1024,3,1,25)
                img = self.Conv(img,1024,3,2,26)
                img = self.Conv(img,1024,3,1,27)
                img = self.Conv(img,1024,3,1,28)
                img = tf.transpose(img,[0,3,1,2])
                img = tf.contrib.layers.flatten(img)
                img = self.Full(img,512,1,29)
                img = self.Full(img,4096,1,30)
                #img = tf.nn.dropout(img,keep_prob)
                img = self.Full(img,num_outputs,0,31)
        return img
    def get_init(self,name,channels,out_size,size):
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

    def Conv(self,input,out_size,size,stride,name):
        channels = input.get_shape()[-1]
        # [batch_size,h,w,channels]
        if self.use_pretrain:
            weight = tf.Variable(self.get_init('conv'+ str(name) + '/weights',channels,out_size,size))
            bias = tf.Variable(self.get_init('conv'+ str(name) + '/biases',channels,out_size,size))
        else:
            weight = tf.get_variable('conv' + str(name)+"/weight", shape=[size,size,channels,out_size],initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('conv' + str(name)+"/bias", shape=[out_size],initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.lamda)(weight))
        pad_size = size/2
        input = tf.pad(input,np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]]))
        output = tf.nn.conv2d(input,weight,strides = [1,stride,stride,1],padding = 'VALID',name = ('conv'+ str(name)))
        output = tf.add(output,bias,name = ('conv'+str(name) + '_bias'))
        #print(channels)
        print(output)
        output = tf.layers.batch_normalization(output,training=self.is_training)
        #mean, variance = tf.nn.moments(output,[0])
        #output = tf.nn.batch_normalization(output,mean,variance,0.0,1.0,0.1)
        #print(output)
        output = tf.maximum(self.alpha*output,output)
        return output
    def Poolling(self,input,size,stride,name):
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

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            #predicts = [batchsize][7*7*30]
            #predicts[:,:b] = [batchsize][7*7*20]
            #qian mian de 7*7*20 yong yu yuce
            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            #sheng xia de 7*7*2  confidence

            predict_boxes = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
            #7*7*8  yong yu hui gui
            
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])

            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])

            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[..., 5:]

            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])


            offset_tran = tf.transpose(offset, (0, 2, 1, 3))
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op
