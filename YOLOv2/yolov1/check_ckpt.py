import os
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
#checkpoint_path = os.path.join(model_dir, "model.ckpt")
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader('./YOLOV2COCO/yolo2_coco.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    a = tf.constant(reader.get_tensor(key))
    print(a)
    #print(reader.get_tensor(key))