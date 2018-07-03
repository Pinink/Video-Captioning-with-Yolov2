import tensorflow as tf
import numpy as np
import yolo_config as cfg
from tensorflow.python import pywrap_tensorflow

from base_model import BaseModel

class CaptionGenerator(BaseModel):

    def build(self):
        """ Build the model. """
        self.build_cnn()
        self.build_rnn()
        if self.is_train:
            self.build_optimizer()
            self.build_summary()

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN...")
        self.build_network(self)
        print("CNN built.")

    def build_network(self,keep_prob = 0.5,scope='yolo'):
        with tf.variable_scope(scope):
            config = self.config
            images = tf.placeholder(
                    dtype = tf.float32,
                    shape = [config.batch_size] + self.image_shape)
            index = 0
            img = images
            img = self.Conv(img,32,3,1,1)
            #self.img1 = img# #
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
            self.conv_feats = tf.reshape(img,[config.batch_size,196,256])
            self.num_ctx = 196
            self.dim_ctx = 256
            self.images = images
        return img
    def reorg(self, inputs):
        outputs_1 = inputs[:, ::2, ::2, :]
        outputs_2 = inputs[:, ::2, 1::2, :]
        outputs_3 = inputs[:, 1::2, ::2, :]
        outputs_4 = inputs[:, 1::2, 1::2, :]
        output = tf.concat([outputs_1, outputs_2, outputs_3, outputs_4], axis = 3)
        return output
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
                if other_name == 'conv7_1_bn/beta':
                    self.img1 = tf.constant(reader.get_tensor(other_name))
                if iname == '0':
                    if other_name == total_name+ '/kernel':
                        result = tf.constant(reader.get_tensor(other_name))
                        assert result.get_shape()[0] == size
                        assert result.get_shape()[1] == size
                        assert result.get_shape()[2] == channels
                        assert result.get_shape()[3] == out_size
                        return result
                else:
                    if other_name == total_name+ iname:
                        result = tf.constant(reader.get_tensor(other_name))
                        return result

    def Conv(self,input,out_size,size,pad_size,name,stride = 1,use_batch = True):
        channels = input.get_shape()[-1]
        # [batch_size,h,w,channels]
        if self.use_pretrain:
            weight = tf.Variable(self.get_init(name,channels,out_size,size))
        else:
            weight = tf.get_variable('conv' + str(name)+"/weight", shape=[size,size,channels,out_size],initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.lamda)(weight))
        if pad_size > 0:
            input = tf.pad(input,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        output = tf.nn.conv2d(input,weight,strides = [1,stride,stride,1],padding = 'VALID',name = ('conv'+ str(name)))
        if use_batch:
            scale = tf.Variable(self.get_init(name,channels,out_size,size,iname = '_bn/gamma'))
            shift = tf.Variable(self.get_init(name,channels,out_size,size,iname = '_bn/beta'))
            mean = tf.Variable(self.get_init(name,channels,out_size,size,iname = '_bn/moving_mean'))
            variance = tf.Variable(self.get_init(name,channels,out_size,size,iname = '_bn/moving_variance'))
            output = tf.nn.batch_normalization(output, mean, variance, shift, scale, 1e-05)
            output = tf.maximum(self.alpha*output,output)
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

    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN...")
        config = self.config

        # Setup the placeholders
        if self.is_train:
            contexts = self.conv_feats
            sentences = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size, config.max_caption_length])
            masks = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.max_caption_length])
        else:
            contexts = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, self.num_ctx, self.dim_ctx])
            last_memory = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_output = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_word = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size])

        # Setup the word embedding
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        # Setup the LSTM
        lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob = 1.0-config.lstm_drop_rate)

        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(self.conv_feats, axis = 1)
            initial_memory, initial_output = self.initialize(context_mean)
            initial_state = initial_memory, initial_output

        # Prepare to run
        predictions = []
        if self.is_train:
            alphas = []
            cross_entropies = []
            predictions_correct = []
            num_steps = config.max_caption_length
            last_output = initial_output
            last_memory = initial_memory
            last_word = tf.zeros([config.batch_size], tf.int32)
        else:
            num_steps = 1
        last_state = last_memory, last_output

        # Generate the words one by one
        for idx in range(num_steps):
            # Attention mechanism
            with tf.variable_scope("attend"):
                alpha = self.attend(contexts, last_output)
                context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
                                        axis = 1)
                if self.is_train:
                    tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
                                         [1, self.num_ctx])
                    masked_alpha = alpha * tiled_masks
                    alphas.append(tf.reshape(masked_alpha, [-1]))

            # Embed the last word
            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    last_word)
           # Apply the LSTM
            with tf.variable_scope("lstm"):
                current_input = tf.concat([context, word_embed], 1)
                output, state = lstm(current_input, last_state)
                memory, _ = state

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             context,
                                             word_embed],
                                             axis = 1)
                logits = self.decode(expanded_output)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)

            # Compute the loss for this step, if necessary
            if self.is_train:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = sentences[:, idx],
                    logits = logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                cross_entropies.append(masked_cross_entropy)

                ground_truth = tf.cast(sentences[:, idx], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(masks[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

                last_output = output
                last_memory = memory
                last_state = state
                last_word = sentences[:, idx]

            tf.get_variable_scope().reuse_variables()

        # Compute the final loss, if necessary
        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks)

            alphas = tf.stack(alphas, axis = 1)
            alphas = tf.reshape(alphas, [config.batch_size, self.num_ctx, -1])
            attentions = tf.reduce_sum(alphas, axis = 2)
            diffs = tf.ones_like(attentions) - attentions
            attention_loss = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs) \
                             / (config.batch_size * self.num_ctx)

            reg_loss = tf.losses.get_regularization_loss()

            total_loss = cross_entropy_loss + attention_loss + reg_loss

            predictions_correct = tf.stack(predictions_correct, axis = 1)
            accuracy = tf.reduce_sum(predictions_correct) \
                       / tf.reduce_sum(masks)

        self.contexts = contexts
        if self.is_train:
            self.sentences = sentences
            self.masks = masks
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.attention_loss = attention_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy
            self.attentions = attentions
        else:
            self.initial_memory = initial_memory
            self.initial_output = initial_output
            self.last_memory = last_memory
            self.last_output = last_output
            self.last_word = last_word
            self.memory = memory
            self.output = output
            self.probs = probs

        print("RNN built.")

    def initialize(self, context_mean):
        """ Initialize the LSTM using the mean context. """
        config = self.config
        context_mean = self.nn.dropout(context_mean)
        if config.num_initalize_layers == 1:
            # use 1 fc layer to initialize
            memory = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a')
            output = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b')
        else:
            # use 2 fc layers to initialize
            temp1 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_a1')
            temp1 = self.nn.dropout(temp1)
            memory = self.nn.dense(temp1,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a2')

            temp2 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_b1')
            temp2 = self.nn.dropout(temp2)
            output = self.nn.dense(temp2,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b2')
        return memory, output

    def attend(self, contexts, output):
        """ Attention Mechanism. """
        config = self.config
        reshaped_contexts = tf.reshape(contexts, [-1, self.dim_ctx])
        reshaped_contexts = self.nn.dropout(reshaped_contexts)
        output = self.nn.dropout(output)
        if config.num_attend_layers == 1:
            # use 1 fc layer to attend
            logits1 = self.nn.dense(reshaped_contexts,
                                    units = 1,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_a')
            logits1 = tf.reshape(logits1, [-1, self.num_ctx])
            logits2 = self.nn.dense(output,
                                    units = self.num_ctx,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_b')
            logits = logits1 + logits2
        else:
            # use 2 fc layers to attend
            temp1 = self.nn.dense(reshaped_contexts,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1a')
            temp2 = self.nn.dense(output,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1b')
            temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
            temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer])
            temp = temp1 + temp2
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = 1,
                                   activation = None,
                                   use_bias = False,
                                   name = 'fc_2')
            logits = tf.reshape(logits, [-1, self.num_ctx])
        alpha = tf.nn.softmax(logits)
        return alpha

    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(expanded_output,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc')
        else:
            # use 2 fc layers to decode
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits

    def build_optimizer(self):
        """ Setup the optimizer and training operation. """
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
        if config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            elif config.optimizer == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate = config.initial_learning_rate,
                    decay = config.decay,
                    momentum = config.momentum,
                    centered = config.centered,
                    epsilon = config.epsilon
                )
            elif config.optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate = config.initial_learning_rate,
                    momentum = config.momentum,
                    use_nesterov = config.use_nesterov
                )
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = config.initial_learning_rate
                )

            opt_op = tf.contrib.layers.optimize_loss(
                loss = self.total_loss,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("attention_loss", self.attention_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("attentions"):
            self.variable_summary(self.attentions)

        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
