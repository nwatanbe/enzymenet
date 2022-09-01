# -*- coding: utf-8 -*-
import tensorflow as tf


## ResNet base model ##
class ResNet(tf.keras.models.Model):
    def __init__(self, stacks, version, use_bias, classes, bn_ep=1.001e-5):

        super(ResNet, self).__init__()

        self.version = version
        self.stacks = stacks

        # initial layers
        self.initial_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same", use_bias=use_bias, name="initial_conv1")
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="pool1")

        ## v1 initial , v2 final layers
        self.bn = None
        self.act = None
        if version == 1:
            self.bn = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name="conv1_bn")
            self.act = tf.keras.layers.Activation("relu", name="conv1_relu")
        
        else:
            self.bn = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name="post_bn")
            self.act = tf.keras.layers.Activation("relu", name="post_relu")
        
        # GAP & dense
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")
        self.dense = tf.keras.layers.Dense(units=classes, activation=None, name="prediction")
    
    
    def call(self, inp, training):
        net = self.initial_conv(inp)

        if self.version == 1:
            net = self.bn(net, training=training)
            net = self.act(net)
        
        net = self.max_pool(net)

        for stack_ in self.stacks:
            net = stack_(net, training=training)
        
        if self.version == 2:
            net = self.bn(net, training=training)
            net = self.act(net)
        
        net = self.gap(net)
        output = self.dense(net)

        return output, net


# for Resnet18 or ResNet34
class residual_block1(tf.keras.layers.Layer):
    def __init__(self, filters, stride1, use_bias, bn_ep, shortcut_flag, name):
        super(residual_block1, self).__init__()

        self.stride1 = stride1
        self.shortcut_flag = shortcut_flag

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=stride1, padding="same", use_bias=use_bias, name= name + "_conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn1")
        self.act1 = tf.keras.layers.Activation("relu", name=name + "_act1")

        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=use_bias, name= name + "_conv2")
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn2")
        
        self.add = tf.keras.layers.Add(name=name + "_add")
        self.act_out = self.act2 = tf.keras.layers.Activation("relu", name=name + "_out")

        self.conv0 = None
        self.bn0 = None
        #if stride1 != 1:
        if shortcut_flag:
            self.conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=stride1, use_bias=use_bias, name= name + "_conv0")
            self.bn0 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn0")
    
    def call(self, inp, training):

        #if self.stride1 != 1:
        if self.shortcut_flag:
            short_cut = self.conv0(inp)
            short_cut = self.bn0(short_cut, training=training)
        
        else:
            short_cut = inp
        
        x = self.conv1(inp)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([x, short_cut])
        out = self.act_out(x)

        return out


class stack1(tf.keras.layers.Layer):
    def __init__(self, block_num, filters, stride1, use_bias, bn_ep, name):
        super(stack1, self).__init__()
        self.blocks = []
        name_ = name + "_block{}"
        for idx in range(block_num):
            if idx == 0:
                self.blocks.append(residual_block1(filters, stride1, use_bias, bn_ep, shortcut_flag=True, name=name_.format(idx)))
            
            else:
                self.blocks.append(residual_block1(filters, 1, use_bias, bn_ep, shortcut_flag=False, name=name_.format(idx)))
        
    def call(self, inp, training):
        x = inp
        for block_ in self.blocks:
            x = block_(x, training=training)
        
        return x


# for ResNet50, ResNet100, ResNet152
class residual_block2(tf.keras.layers.Layer):
    def __init__(self, filters, stride1, use_bias, bn_ep, shortcut_flag, name):
        super(residual_block2, self).__init__()

        self.stride1 = stride1
        self.shortcut_flag = shortcut_flag

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same", use_bias=use_bias, name= name + "_conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn1")
        self.act1 = tf.keras.layers.Activation("relu", name=name + "_act1")

        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=stride1, padding="same", use_bias=use_bias, name= name + "_conv2")
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn2")
        self.act2 = tf.keras.layers.Activation("relu", name=name + "_act2")
        
        self.conv3 = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=1, strides=1, padding="same", use_bias=use_bias, name= name + "_conv3")
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn3")


        self.add = tf.keras.layers.Add(name=name + "_add")
        self.act_out = self.act2 = tf.keras.layers.Activation("relu", name=name + "_out")

        self.conv0 = None
        self.bn0 = None
        #if stride1 != 1:
        if shortcut_flag:
            self.conv0 = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=1, strides=stride1, use_bias=use_bias, name= name + "_conv0")
            self.bn0 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn0")
    
    def call(self, inp, training):

        #if self.stride1 != 1:
        if self.shortcut_flag:
            short_cut = self.conv0(inp)
            short_cut = self.bn0(short_cut, training=training)
        
        else:
            short_cut = inp
        
        x = self.conv1(inp)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([x, short_cut])
        out = self.act_out(x)

        return out


class stack2(tf.keras.layers.Layer):
    def __init__(self, block_num, filters, stride1, use_bias, bn_ep, name):
        super(stack2, self).__init__()
        self.blocks = []
        name_ = name + "_block{}"
        for idx in range(block_num):
            if idx == 0:
                self.blocks.append(residual_block2(filters, stride1, use_bias, bn_ep, shortcut_flag=True, name=name_.format(idx)))
            
            else:
                self.blocks.append(residual_block2(filters, 1, use_bias, bn_ep, shortcut_flag=False, name=name_.format(idx)))
        
    def call(self, inp, training):
        x = inp
        for block_ in self.blocks:
            x = block_(x, training=training)
        
        return x


# for Resnet18v2 or ResNet34v2
class residual_block3(tf.keras.layers.Layer):
    def __init__(self, filters, stride1, use_bias, bn_ep, shortcut_flag, name):
        super(residual_block3, self).__init__()

        self.stride1 = stride1
        self.shortcut_flag = shortcut_flag

        self.bn0 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn0")
        self.act0 = tf.keras.layers.Activation("relu", name=name + "_act0")

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=stride1, padding="same", use_bias=use_bias, name= name + "_conv1")

        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn2")
        self.act2 = tf.keras.layers.Activation("relu", name=name + "_act2")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=use_bias, name= name + "_conv2")
        
        
        self.add = tf.keras.layers.Add(name=name + "_add")

        self.conv0 = None
        #if stride1 != 1:
        if shortcut_flag:
            self.conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=stride1, use_bias=use_bias, name= name + "_conv0")
    
    def call(self, inp, training):
        pre_x = self.bn0(inp, training=training)
        pre_x = self.act0(pre_x)

        #if self.stride1 != 1:
        if self.shortcut_flag:
            short_cut = self.conv0(pre_x)
        
        else:
            short_cut = inp
        
        x = self.conv1(pre_x)

        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.conv2(x)

        out = self.add([x, short_cut])

        return out


class stack3(tf.keras.layers.Layer):
    def __init__(self, block_num, filters, stride1, use_bias, bn_ep, name):
        super(stack3, self).__init__()
        self.blocks = []
        name_ = name + "_block{}"
        for idx in range(block_num):
            if idx == 0:
                self.blocks.append(residual_block3(filters, stride1, use_bias, bn_ep, shortcut_flag=True, name=name_.format(idx)))
            
            else:
                self.blocks.append(residual_block3(filters, 1, use_bias, bn_ep, shortcut_flag=False, name=name_.format(idx)))
        
    def call(self, inp, training):
        x = inp
        for block_ in self.blocks:
            x = block_(x, training=training)
        
        return x




# for ResNet50v2, ResNet100v2, ResNet152v2
class residual_block4(tf.keras.layers.Layer):
    def __init__(self, filters, stride1, use_bias, bn_ep, shortcut_flag, name):
        super(residual_block4, self).__init__()

        self.stride1 = stride1
        self.shortcut_flag =shortcut_flag

        self.bn0 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn0")
        self.act0 = tf.keras.layers.Activation("relu", name=name + "_act0")

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same", use_bias=use_bias, name= name + "_conv1")

        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn2")
        self.act2 = tf.keras.layers.Activation("relu", name=name + "_act2")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=stride1, padding="same", use_bias=use_bias, name= name + "_conv2")

        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=bn_ep, name=name + "_bn3")
        self.act3 = tf.keras.layers.Activation("relu", name=name + "_act3")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=1, strides=1, padding="same", use_bias=use_bias, name= name + "_conv3")

        self.add = tf.keras.layers.Add(name=name + "_add")

        self.conv0 = None
        #if stride1 != 1:
        if shortcut_flag:
            self.conv0 = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=1, strides=stride1, use_bias=use_bias, name= name + "_conv0")
    
    def call(self, inp, training):

        pre_x = self.bn0(inp, training=training)
        pre_x = self.act0(pre_x)

        #if self.stride1 != 1:
        if self.shortcut_flag:
            short_cut = self.conv0(pre_x)
        
        else:
            short_cut = inp
        
        x = self.conv1(pre_x)


        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.conv2(x)

        x = self.bn3(x, training=training)
        x = self.act3(x)
        x = self.conv3(x)

        out = self.add([x, short_cut])

        return out


class stack4(tf.keras.layers.Layer):
    def __init__(self, block_num, filters, stride1, use_bias, bn_ep, name):
        super(stack4, self).__init__()
        self.blocks = []
        name_ = name + "_block{}"
        for idx in range(block_num):
            if idx == 0:
                self.blocks.append(residual_block4(filters, stride1, use_bias, bn_ep, shortcut_flag=True, name=name_.format(idx)))
            
            else:
                self.blocks.append(residual_block4(filters, 1, use_bias, bn_ep, shortcut_flag=False, name=name_.format(idx)))
        
    def call(self, inp, training):
        x = inp
        for block_ in self.blocks:
            x = block_(x, training=training)
        
        return x






def ResNet18(classes, use_bias=True, bn_ep=1.001e-5):
    stack_1 = stack1(2, 64, 1, use_bias, bn_ep, name="Conv1")
    stack_2 = stack1(2, 128, 2, use_bias, bn_ep, name="Conv2")
    stack_3 = stack1(2, 256, 2, use_bias, bn_ep, name="Conv3")
    stack_4 = stack1(2, 512, 2, use_bias, bn_ep, name="Conv4")
    stack_fn = [stack_1, stack_2, stack_3, stack_4]

    return ResNet(stack_fn, 1, use_bias, classes, bn_ep)



def ResNet34(classes, use_bias=True, bn_ep=1.001e-5):
    stack_1 = stack1(3, 64, 1, use_bias, bn_ep, name="Conv1")
    stack_2 = stack1(4, 128, 2, use_bias, bn_ep, name="Conv2")
    stack_3 = stack1(6, 256, 2, use_bias, bn_ep, name="Conv3")
    stack_4 = stack1(3, 512, 2, use_bias, bn_ep, name="Conv4")
    stack_fn = [stack_1, stack_2, stack_3, stack_4]

    return ResNet(stack_fn, 1, use_bias, classes, bn_ep)


def ResNet50(classes, use_bias=True, bn_ep=1.001e-5):
    stack_1 = stack2(3, 64, 1, use_bias, bn_ep, name="Conv1")
    stack_2 = stack2(4, 128, 2, use_bias, bn_ep, name="Conv2")
    stack_3 = stack2(6, 256, 2, use_bias, bn_ep, name="Conv3")
    stack_4 = stack2(3, 512, 2, use_bias, bn_ep, name="Conv4")
    stack_fn = [stack_1, stack_2, stack_3, stack_4]

    return ResNet(stack_fn, 1, use_bias, classes, bn_ep)

def ResNet100(classes, use_bias=True, bn_ep=1.001e-5):
    stack_1 = stack2(3, 64, 1, use_bias, bn_ep, name="Conv1")
    stack_2 = stack2(4, 128, 2, use_bias, bn_ep, name="Conv2")
    stack_3 = stack2(23, 256, 2, use_bias, bn_ep, name="Conv3")
    stack_4 = stack2(3, 512, 2, use_bias, bn_ep, name="Conv4")
    stack_fn = [stack_1, stack_2, stack_3, stack_4]

    return ResNet(stack_fn, 1, use_bias, classes, bn_ep)

def ResNet152(classes, use_bias=True, bn_ep=1.001e-5):
    stack_1 = stack2(3, 64, 1, use_bias, bn_ep, name="Conv1")
    stack_2 = stack2(8, 128, 2, use_bias, bn_ep, name="Conv2")
    stack_3 = stack2(36, 256, 2, use_bias, bn_ep, name="Conv3")
    stack_4 = stack2(3, 512, 2, use_bias, bn_ep, name="Conv4")
    stack_fn = [stack_1, stack_2, stack_3, stack_4]

    return ResNet(stack_fn, 1, use_bias, classes, bn_ep)





def ResNet18v2(classes, use_bias=True, bn_ep=1.001e-5):
    stack_1 = stack3(2, 64, 1, use_bias, bn_ep, name="Conv1")
    stack_2 = stack3(2, 128, 2, use_bias, bn_ep, name="Conv2")
    stack_3 = stack3(2, 256, 2, use_bias, bn_ep, name="Conv3")
    stack_4 = stack3(2, 512, 2, use_bias, bn_ep, name="Conv4")
    stack_fn = [stack_1, stack_2, stack_3, stack_4]

    return ResNet(stack_fn, 2, use_bias, classes, bn_ep)

def ResNet34v2(classes, use_bias=True, bn_ep=1.001e-5):
    stack_1 = stack3(3, 64, 1, use_bias, bn_ep, name="Conv1")
    stack_2 = stack3(4, 128, 2, use_bias, bn_ep, name="Conv2")
    stack_3 = stack3(6, 256, 2, use_bias, bn_ep, name="Conv3")
    stack_4 = stack3(3, 512, 2, use_bias, bn_ep, name="Conv4")
    stack_fn = [stack_1, stack_2, stack_3, stack_4]

    return ResNet(stack_fn, 2, use_bias, classes, bn_ep)

def ResNet50v2(classes, use_bias=True, bn_ep=1.001e-5):
    stack_1 = stack4(3, 64, 1, use_bias, bn_ep, name="Conv1")
    stack_2 = stack4(4, 128, 2, use_bias, bn_ep, name="Conv2")
    stack_3 = stack4(6, 256, 2, use_bias, bn_ep, name="Conv3")
    stack_4 = stack4(3, 512, 2, use_bias, bn_ep, name="Conv4")
    stack_fn = [stack_1, stack_2, stack_3, stack_4]

    return ResNet(stack_fn, 2, use_bias, classes, bn_ep)

def ResNet100v2(classes, use_bias=True, bn_ep=1.001e-5):
    stack_1 = stack4(3, 64, 1, use_bias, bn_ep, name="Conv1")
    stack_2 = stack4(4, 128, 2, use_bias, bn_ep, name="Conv2")
    stack_3 = stack4(23, 256, 2, use_bias, bn_ep, name="Conv3")
    stack_4 = stack4(3, 512, 2, use_bias, bn_ep, name="Conv4")
    stack_fn = [stack_1, stack_2, stack_3, stack_4]

    return ResNet(stack_fn, 2, use_bias, classes, bn_ep)

def ResNet152v2(classes, use_bias=True, bn_ep=1.001e-5):
    stack_1 = stack4(3, 64, 1, use_bias, bn_ep, name="Conv1")
    stack_2 = stack4(8, 128, 2, use_bias, bn_ep, name="Conv2")
    stack_3 = stack4(36, 256, 2, use_bias, bn_ep, name="Conv3")
    stack_4 = stack4(3, 512, 2, use_bias, bn_ep, name="Conv4")
    stack_fn = [stack_1, stack_2, stack_3, stack_4]

    return ResNet(stack_fn, 2, use_bias, classes, bn_ep)




if __name__ == "__main__":
    """
    import numpy as np

    dic = {
        "embedding_dim":64,
        "d_model":1024,
        "input_vocab_size":24,
        "maximun_position_encoding":1024,
        "channel_dim":3,
        "cv_size":4,
        "seq_split_len":128,
        "d_split_len":128,
        "split_final_shape":["none", 128, 128, 8],
        "resnet_input_shape":[128, 128, 3],
        "out_class":7,
        "rate":0.1
    }

    dic["split_final_shape"][0] = None 

    ec_pred = EC_Predictor(**dic)
    inp = tf.random.uniform((5, 1024), minval=0, maxval=23, dtype=tf.int32)
    mask = np.ones((5, 1024))
    mask[0:2, 996:1024] = 0
    mask = tf.constant(mask, dtype=tf.float32)
    mask = mask[:, :, tf.newaxis]

    res = ec_pred(inp, True, mask)
    print(res.shape)
    ec_pred.summary()
    """
    #stack_l = stack1(3, 128, 2, True, 1.001e-5, "stack1")
    #block1 = residual_block1(128, 2, True, 1.001e-5, "block1")

    #res18 = ResNet18(7)
    #res34 = ResNet34(7)
    #res50 = ResNet50(7)
    #res100 = ResNet100(7)
    #res152 = ResNet152(7)

    res18v2 = ResNet18v2(7)
    res34v2 = ResNet34v2(7)
    res50v2 = ResNet50v2(7)
    res100v2 = ResNet100v2(7)
    res152v2 = ResNet152v2(7)




    """
    inp = tf.keras.layers.Input((224, 224, 3))
    x = res50v2(inp, True)
    model = tf.keras.models.Model(inp, x)
    model.summary()
    """

    """
    y = tf.random.uniform((128, 128, 128, 3))

    import time
    x1 = time.time()
    res = model(y, True)
    print("{:.2f}s".format(time.time() - x1))
    print(res.shape)
    """












