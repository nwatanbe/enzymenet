# -*- coding: utf-8 -*-
import tensorflow as tf
from efficient_layers import EmbeddingPostprocessor, ConvertImg_v3
from resnet import ResNet50v2 #データ量等で適切なものを選択
from iv_resnet import ResNet50v2 as iv_ResNet50v2


# 中間ベクトル抽出用 ResNet
class IV_model(tf.keras.models.Model):
    def __init__(
        self,
        embedding_dim,
        d_model,
        input_vocab_size,
        maximum_position_encoding,
        channel_dim,
        cv_size,
        seq_split_len,
        d_split_len,
        out_class,
        out_act,
        rate = 0.1
    ):

        super(IV_model, self).__init__()

        self.emb = EmbeddingPostprocessor(input_vocab_size, maximum_position_encoding, embedding_dim, d_model, rate)
        self.cvimg = ConvertImg_v3(channel_dim, cv_size, maximum_position_encoding, d_model, seq_split_len, d_split_len)
        self.resnet = iv_ResNet50v2(out_class, use_bias=True)
        self.final_act = tf.keras.layers.Activation(out_act)
    

    def call(self, inp, training, padding_mask):
        x = self.emb(inp, training)
        x *= padding_mask
        x = self.cvimg(x, training)
        x, iv = self.resnet(x, training)
        output = self.final_act(x)

        return output, iv

# 予測用ResNet
## ResNet ##
class EC_Predictor(tf.keras.models.Model):
    def __init__(
        self,
        embedding_dim,
        d_model,
        input_vocab_size,
        maximum_position_encoding,
        channel_dim,
        cv_size,
        seq_split_len,
        d_split_len,
        out_class,
        out_act,
        rate = 0.1
    ):

        super(EC_Predictor, self).__init__()

        self.emb = EmbeddingPostprocessor(input_vocab_size, maximum_position_encoding, embedding_dim, d_model, rate)
        self.cvimg = ConvertImg_v3(channel_dim, cv_size, maximum_position_encoding, d_model, seq_split_len, d_split_len)
        self.resnet = ResNet50v2(out_class, use_bias=True)
        self.final_act = tf.keras.layers.Activation(out_act)
    

    def call(self, inp, training, padding_mask):
        x = self.emb(inp, training)
        x *= padding_mask
        x = self.cvimg(x, training)
        x = self.resnet(x, training)
        output = self.final_act(x)

        return output


# 中間ベクトル抽出用 ResNet
class iv_model(tf.keras.models.Model):
    def __init__(
        self,
        embedding_dim,
        d_model,
        input_vocab_size,
        maximum_position_encoding,
        channel_dim,
        cv_size,
        seq_split_len,
        d_split_len,
        out_class,
        out_act,
        rate = 0.1
    ):

        super(IV_model, self).__init__()

        self.emb = EmbeddingPostprocessor(input_vocab_size, maximum_position_encoding, embedding_dim, d_model, rate)
        self.cvimg = ConvertImg_v3(channel_dim, cv_size, maximum_position_encoding, d_model, seq_split_len, d_split_len)
        self.resnet = iv_ResNet50v2(out_class, use_bias=True)
        self.final_act = tf.keras.layers.Activation(out_act)
    

    def call(self, inp, training, padding_mask):
        x = self.emb(inp, training)
        x *= padding_mask
        x = self.cvimg(x, training)
        x, iv = self.resnet(x, training)
        output = self.final_act(x)

        return output, iv


# EC4桁目まで予測用モデル(転移学習用)
# 転移学習用モデル
class transfer_model(tf.keras.models.Model):
    def __init__(
        self,
        emb_layer,
        cvimg_layer,
        resnet_layers,
        out_class,
        out_act
    ):

        super(transfer_model, self).__init__()

        self.emb = emb_layer
        self.cvimg = cvimg_layer
        self.resnets = resnet_layers
        self.dense = tf.keras.layers.Dense(units=out_class, activation=None, name="prediction")
        self.act = tf.keras.layers.Activation(out_act)

    def call(self, inp, training, padding_mask):
        x = self.emb(inp, training)
        x *= padding_mask
        x = self.cvimg(x, training)
        x = self.resnets[4](x)
        x = self.resnets[5](x)
        x = self.resnets[0](x, training=training)
        x = self.resnets[1](x, training=training)
        x = self.resnets[2](x, training=training)
        x = self.resnets[3](x, training=training)
        x = self.resnets[6](x, training=training)
        x = self.resnets[7](x)
        x = self.resnets[8](x)

        x = self.dense(x)
        output = self.act(x)

        return output


# 転移学習モデル 中間ベクトル抽出モデル
class transfer_iv_model(tf.keras.models.Model):
    def __init__(
        self,
        emb_layer,
        cvimg_layer,
        resnet_layers,
        out_class,
        out_act
    ):

        super(transfer_iv_model, self).__init__()

        self.emb = emb_layer
        self.cvimg = cvimg_layer
        self.resnets = resnet_layers
        self.dense = tf.keras.layers.Dense(units=out_class, activation=None, name="prediction")
        self.act = tf.keras.layers.Activation(out_act)

    def call(self, inp, training, padding_mask):
        x = self.emb(inp, training)
        x *= padding_mask
        x = self.cvimg(x, training)
        x = self.resnets[4](x)
        x = self.resnets[5](x)
        x = self.resnets[0](x, training=training)
        x = self.resnets[1](x, training=training)
        x = self.resnets[2](x, training=training)
        x = self.resnets[3](x, training=training)
        x = self.resnets[6](x, training=training)
        x = self.resnets[7](x)
        interlayer_vec = self.resnets[8](x)

        x = self.dense(interlayer_vec)
        output = self.act(x)

        return output, interlayer_vec