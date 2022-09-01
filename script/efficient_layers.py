# -*- coding: utf-8 -*-
"""
Transformerに必要な基本的なlayer群を記載
tf2.1.0で動作させることを想定して実装
"""

import tensorflow as tf
import numpy as np

from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers



## padding_mask ##
def create_padding_mask(inp, padding_id=0):
    inp_ = tf.cast(tf.math.equal(inp, padding_id), dtype=tf.float32)
    mask = 1 - inp_
    return mask[:, :, tf.newaxis]




## Positional embedding ##
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_seqlen, embedding_dim):
        super(PositionalEmbedding, self).__init__()

        self.max_seqlen = max_seqlen
        self.embedding_dim = embedding_dim
        self.emb_initializer = initializers.get(tf.keras.initializers.TruncatedNormal(stddev=0.02)) #通常は uniformだが ALBERTはtruncated_normalなので
        self.emb_consraint = constraints.get(None)
        self.emb_regularizer = regularizers.get(None)
    
    def build(self, input_shape):
        
        # 重みの初期化と設定 W = [max_seqlen, embedding_dim]
        self.embedding_table = self.add_weight(
            name="position_embedding",
            dtype=tf.keras.backend.floatx(),
            shape=[self.max_seqlen, self.embedding_dim],
            initializer = self.emb_initializer,
            regularizer = self.emb_regularizer,
            constraint = self.emb_consraint
        )

        self.built = True
    
    def call(self, seqlen):
        output = tf.slice(
            self.embedding_table,
            [0, 0],
            [seqlen, -1]
        )

        return output # [..., seqlen, embedding_dim] 通常は[seqlen, embedding]になるため後でbroadcastが必要


## projector_layer (= tf.keras.layers.Dense ?)
class projector_layer_2d(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        w_initializer="truncated_normal",
        b_initializer = "zeros",
        use_einsum = True
    ):
        super(projector_layer_2d, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.w_initializer = None
        if w_initializer == "truncated_normal":
            self.w_initializer = initializers.get(tf.keras.initializers.TruncatedNormal(stddev=0.02))
        else:
            self.w_initializer = initializers.get(w_initializer)

        self.b_initializer = initializers.get(b_initializer)

        self.use_einsum = use_einsum
    
    def build(self, input_shape):
        self.projector_table = self.add_weight(
            name="projector_table",
            dtype=tf.keras.backend.floatx(),
            shape=[self.embedding_dim, self.hidden_dim],
            initializer = self.w_initializer
        ) # [embedding_dim, hidden_dim]

        self.projector_bias = self.add_weight(
            name="projector_bias",
            dtype=tf.keras.backend.floatx(),
            shape=[self.hidden_dim],
            initializer = self.b_initializer
        )

        self.built = True
    
    def call(self, input_tensor):
        if self.use_einsum:
            output = tf.einsum("BLE,EH->BLH", input_tensor, self.projector_table) # [batch, seqlen, embedding_dim] -> [batch, seqlen, hidden_dim]
        else:
            output = tf.matmul(input_tensor, self.projector_table)
        
        output += self.projector_bias
        
        return output #[batch, seqlen, hidden_dim]


class EmbeddingPostprocessor(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_seqlen, embedding_dim, hidden_dim, rate):
        super(EmbeddingPostprocessor, self).__init__()

        self.embedding_dim = embedding_dim


        self.token_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.position_emb = PositionalEmbedding(max_seqlen, embedding_dim)
        self.projector = projector_layer_2d(embedding_dim, hidden_dim)

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, inputs, training):
        seqlen = tf.shape(inputs)[-1]
        # token embedding
        output = self.token_emb(inputs)

        # position embedding
        position_output = self.position_emb(seqlen) #[...., seqlen, embedding_dim]
        num_dims = output.shape.ndims
        brodcast_shape = [1] * (num_dims-2) + [seqlen, self.embedding_dim]
        output += tf.reshape(position_output, brodcast_shape)

        # layer_norm & dropout
        output = self.layernorm(output)
        output = self.dropout(output, training=training)

        # projecter [Batch, Len, embedding_dim] -> [Batch, Len, hidden_dim]
        output = self.projector(output)

        return output



class ConvertImg_v1(tf.keras.layers.Layer): # seqlengthは足し合わせ 特徴量次元はconv2dで圧縮
    def __init__(self, channel_dim, size, seq_split_len, d_split_len, split_final_shape=[None, 128, 128, 8]):

        super(ConvertImg_v1, self).__init__()

        self.seq_split_len = seq_split_len
        self.d_split_len = d_split_len
        self.split_final = split_final_shape
        
        self.conv_col1 = tf.keras.layers.Conv2D(filters=channel_dim, kernel_size=size, strides=(1, 1), padding="same", activation=None)
        self.bn_col1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.act_col1 = tf.keras.layers.Activation("relu")

        self.conv_final = tf.keras.layers.Conv2D(filters=channel_dim, kernel_size=1, strides=(1, 1), padding="same", activation=None)
    
    def reshape_add(self, x, reshape_size):
        x = tf.transpose(x, perm=[0, 2, 1]) # [Batch, length, d_model] -> [Batch, d_model, length]
        x = tf.reshape(x, reshape_size) #[Batch, d_model, split_num, split_len]
        x = tf.transpose(x, perm=[0, 3, 1, 2]) # [Batch, split_len, d_model, split_num]
        x = tf.reduce_sum(x, axis=-1) # [Batch, split_len, d_model]
        return x
    
    def reshape_only(self, x, reshape_size):
        x = tf.reshape(x, reshape_size) #[Batch, split_len, d_model] -> [Batch, split_len, d_split_num, d_split_len]
        x = tf.transpose(x, perm=[0, 1, 3, 2])
        return x
    
    def call(self, inp, training):

        ## create_reshape_size
        shape_ = tf.shape(inp)
        batch = shape_[0]
        seqlen = shape_[1]
        d_model = shape_[2]

        seq_split_num = seqlen // self.seq_split_len
        reshape_size1 = [batch, d_model, seq_split_num, self.seq_split_len]


        d_split_num = d_model // self.d_split_len
        reshape_size2 = [batch, self.seq_split_len, d_split_num, self.d_split_len]

        ## convert [Batch, selen, d_model] -> [Batch, seq_split_len, d_model]
        x = self.reshape_add(inp, reshape_size1)

        ## convert [Batch, seq_split_len, d_model] -> [Batch, seq_split_len, d_splitlen, d_split_num]
        x = self.reshape_only(x, reshape_size2)
        x.set_shape(self.split_final)

        ## convert img [Batch, seq_split_len, d_model] -> [Batch, seq_split_len, d_splitlen, channel_dim]
        x = self.conv_col1(x)
        x = self.bn_col1(x, training=training)
        x = self.act_col1(x)
        output = self.conv_final(x)

        return output




class ConvertImg_v2(tf.keras.layers.Layer): # seqlengthをconv2d圧縮, 特徴量次元は圧縮なし
    def __init__(self, channel_dim, size, seq_split_len, split_shape=[None, 128, 128, 8]):

        super(ConvertImg_v2, self).__init__()

        self.seq_split_len = seq_split_len
        self.split_shape = split_shape
        
        self.conv = tf.keras.layers.Conv2D(filters=channel_dim, kernel_size=size, strides=(1, 1), padding="same", activation=None)
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.act = tf.keras.layers.Activation("relu")

        self.conv_final = tf.keras.layers.Conv2D(filters=channel_dim, kernel_size=1, strides=(1, 1), padding="same", activation=None)
    
    def reshape_fn(self, x, reshape_size):
        x = tf.transpose(x, perm=[0, 2, 1]) # [Batch, length, d_model] -> [Batch, d_model, length]
        x = tf.reshape(x, reshape_size) #[Batch, d_model, split_num, split_len]
        x = tf.transpose(x, perm=[0, 3, 1, 2]) # [Batch, split_len, d_model, split_num]
        return x
    
    def call(self, inp, training):

        ## create_reshape_size
        shape_ = tf.shape(inp)
        batch = shape_[0]
        seqlen = shape_[1]
        d_model = shape_[2]

        seq_split_num = seqlen // self.seq_split_len
        reshape_size = [batch, d_model, seq_split_num, self.seq_split_len]


        ## convert [Batch, selen, d_model] -> [Batch, seq_split_len, d_model, channel_dim]
        x = self.reshape_fn(inp, reshape_size)
        x.set_shape(self.split_shape)

        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)

        output = self.conv_final(x)

        return output


class ConvertImg_v3(tf.keras.layers.Layer): # seqlengthはconv2d圧縮 特徴量次元はdenseで圧縮
    def __init__(self, channel_dim, size, seqlen, d_model, seq_split_len, d_split_len):

        super(ConvertImg_v3, self).__init__()

        self.channel_dim = channel_dim
        self.seq_split_len = seq_split_len
        self.d_split_len = d_split_len
        self.seqlen = seqlen
        self.d_model = d_model

        self.proj_ds = tf.keras.layers.Dense(units=d_split_len, use_bias=True)
        self.proj_bn = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.proj_act = tf.keras.layers.Activation("relu")
        
        self.conv = tf.keras.layers.Conv2D(filters=channel_dim, kernel_size=size, strides=(1, 1), padding="same", activation=None)
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.act = tf.keras.layers.Activation("relu")

        self.conv_final = tf.keras.layers.Conv2D(filters=channel_dim, kernel_size=1, strides=(1, 1), padding="same", activation=None)
    
    def reshape_fn(self, x, reshape_size):
        x = tf.transpose(x, perm=[0, 2, 1]) # [Batch, length, d_model] -> [Batch, d_model, length]
        x = tf.reshape(x, reshape_size) #[Batch, d_model, split_num, split_len]
        x = tf.transpose(x, perm=[0, 3, 1, 2]) # [Batch, split_len, d_model, split_num]
        return x
    
    
    def call(self, inp, training):

        ## create_reshape_size
        shape_ = tf.shape(inp)
        batch = shape_[0]
        seqlen = self.seqlen
        d_model = self.d_model

        seq_split_num = seqlen // self.seq_split_len
        reshape_size = [batch, d_model, seq_split_num, self.seq_split_len]
        shape_set1 = [None, self.seq_split_len, d_model, seq_split_num]


        shape_trans = [0, 1, 3, 2]
        shape_set2 = [None, self.seq_split_len, self.channel_dim, d_model]
        shape_set3 = [None, self.seq_split_len, self.d_split_len, self.channel_dim]


        ## convert [Batch, seq_len, d_model] -> [Batch, seq_split_len, d_model, channel_dim]
        x = self.reshape_fn(inp, reshape_size)
        x.set_shape(shape_set1)

        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)


        ## convert [Batch, seq_split_len, d_model, channel_dim] -> [Batch, seq_split_len, d_split_len, channel_dim]
        x = tf.transpose(x, perm=shape_trans)
        x.set_shape(shape_set2)

        x = self.proj_ds(x)
        x = self.proj_bn(x, training=training)
        x = self.proj_act(x)

        x = tf.transpose(x, perm=shape_trans)
        x.set_shape(shape_set3)



        # output
        output = self.conv_final(x)

        return output
