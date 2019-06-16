
import  tensorflow as tf

import numpy as np
import keras
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.engine import Layer
from keras.layers import ReLU
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from data_keras import label_acc_score,keras_data
import math
import  matplotlib.pyplot as plt
import keras.backend as backend
import h5py
pth='mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
pth_vgg='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.image_data_format()
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                                 name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def hed(input_shape=(256, 256, 3),alpha=1):
    img_input = Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input) # no batch ,need to try
    x = layers.BatchNormalization(
    epsilon=1e-3, momentum=0.999, name='vgg_bn_Conv1')(x)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x) # no batch ,need to try
    x = layers.BatchNormalization(
    epsilon=1e-3, momentum=0.999, name='vgg_bn_Conv2')(x)
    x_side_0=x
    x = layers.Conv2D(3, (1, 1),
                      activation='relu',
                      padding='same',
                      name='vgg_mobile')(x)

    x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                         name='Conv1_pad')(x)
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.Conv2D(first_block_filters,
                  kernel_size=3,
                  strides=(2, 2),
                  padding='valid',
                  use_bias=False,
                  name='Conv1')(x)
    x = layers.BatchNormalization(
    epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)
    x_side_1 = x
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)
    x_side_2 = x
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)
    x_side_3 = x
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)
    x_side_4 = x

    x_side_1=Conv2D(1,(1,1),padding='same',)(x_side_1)
    x_side_2=Conv2D(1,(1,1),padding='same',)(x_side_2)
    x_side_3=Conv2D(1,(1,1),padding='same',)(x_side_3)
    x_side_4=Conv2D(1,(1,1),padding='same',)(x_side_4)
    x_side_0=Conv2D(1,(1,1),padding='same',)(x_side_0)
    # x_side_1 =keras.layers.UpSampling2D((2,2),interpolation='bilinear',)(x_side_1)
    # x_side_2 =keras.layers.UpSampling2D((4,4),interpolation='bilinear',)(x_side_2)
    # x_side_3 =keras.layers.UpSampling2D((8,8),interpolation='bilinear',)(x_side_3)
    # x_side_4 =keras.layers.UpSampling2D((16,16),interpolation='bilinear',)(x_side_4)
    x_side_1=BilinearUpsampling(output_size=256)(x_side_1)
    x_side_2 = BilinearUpsampling(output_size=256)(x_side_2)
    x_side_3 = BilinearUpsampling(output_size=256)(x_side_3)
    x_side_4 = BilinearUpsampling(output_size=256)(x_side_4)

    x_fuse=Concatenate(axis=3)([x_side_1,x_side_2,x_side_3,x_side_4,x_side_0])
    x_fuse=Conv2D(1,(1,1))(x_fuse)

    x_side_0=keras.layers.Activation('sigmoid')(x_side_0)
    x_side_1=keras.layers.Activation('sigmoid')(x_side_1)
    x_side_2=keras.layers.Activation('sigmoid')(x_side_2)
    x_side_3=keras.layers.Activation('sigmoid')(x_side_3)
    x_side_4=keras.layers.Activation('sigmoid')(x_side_4)
    x_fuse  =keras.layers.Activation('sigmoid')(x_fuse)




    model=Model(inputs=img_input,outputs=[x_side_0,x_side_1,x_side_2,x_side_3,x_side_4,x_fuse])
    return model

def model_initialize(model):
    m=h5py.File('hehe.h5')['model_weights']
    v=h5py.File(pth_vgg)
    # # def model_initialize(model,vgg_pth,mobile_pth):
    # vgg=h5py.File(pth_vgg)
    # model_layers=model.layers

    model.layers[1].set_weights(v['block1_conv1'].values())
    model.layers[3].set_weights(v['block1_conv2'].values())
    for i in model.layers[5:]:
        try:

            i.set_weights(m[i.name].values()[0].values())
        except:
            print (i.name+'    initialize failed')
    return model
model=hed()
model=model_initialize(model)
batch_size=24
train_data=keras_data(batch_size=batch_size)
val_data=keras_data(image_set='test',batch_size=batch_size)
steps = math.ceil(30000 / batch_size)
optim=keras.optimizers.Adam()
def cross_entropy_balanced(y_true, y_pred):
    _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))
    y_true = tf.cast(y_true, tf.float32)
    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)
    beta = count_neg / (count_neg + count_pos)
    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta))
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)
def my_loss(y_true,y_pred):
    y_reshape=tf.reshape(y_true,[-1])
    a=tf.reduce_sum(1.-y_reshape)
    b=tf.reduce_sum(y_reshape)
    a=tf.cast(a,tf.float32)
    beta=a/(a+b)
    return beta/(1-beta)*tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred,beta/(1-beta))
model.compile(optimizer=optim,loss=[my_loss,my_loss,my_loss,my_loss,my_loss,my_loss]
              ,loss_weights=[0.2,0.2,0.2,0.2,0.2,1])
history=model.fit_generator(train_data,steps_per_epoch=steps,epochs=10,use_multiprocessing=True, verbose=1,workers=4,)
keras.models.save_model(model,'hed.h5',include_optimizer=False)

def picture(pre_numpy,img_numpy,mask_numpy):
    #pre_numpy has shape num,height,width,channel
    voc_colormap=np.array([[0, 0, 0], [245,222,179]])
    num=len(img_numpy)
    target=(pre_numpy>0.5).squeeze().astype(int)
    mean,std=np.array((0.485, 0.456, 0.406)),np.array((0.229, 0.224, 0.225))
    img=img_numpy*std+mean
    img=img.squeeze()
    mask=voc_colormap[mask_numpy.squeeze().astype(int)]
    tar=voc_colormap[target]/255.
    tmp=np.concatenate((img,tar,mask),axis=0)
    for i,j in enumerate(tmp,1):
        plt.subplot(3,num,i)
        plt.imshow(j)
    # plt.savefig('true_weight_alpha=05')
    plt.show()
# def test(model,thread=0.5):
#     img=[]
#     pred=[]
#     mask=[]
#     l5_list=[]
#     for image,mask_img in val_data:
#         l5=model.predict(image)
#         label=(l5>thread).squeeze().astype(int)
#         l5_list.append(l5)
#         pred.append(label)
#         img.append(image)
#         mask.append(mask_img)
#     return np.concatenate(img,axis=0),np.concatenate(pred,axis=0),np.concatenate(mask,axis=0),l5_list
def test(model):
    img=[]
    pred=[]
    mask=[]
    l0_list=[]
    l1_list=[]
    l2_list=[]
    l3_list=[]
    l4_list=[]
    l5_list=[]

    for image,mask_img in val_data:
        l0,l1,l2,l3,l4,\
        l5=model.predict(image)
        label=(l5>0.5).squeeze().astype(int)
        l0_list.append(l0)
        l1_list.append(l1)
        l2_list.append(l2)
        l3_list.append(l3)
        l4_list.append(l4)
        l5_list.append(l5)
        pred.append(label)
        img.append(image)
        mask.append(mask_img[0])
    return np.concatenate(img,axis=0),np.concatenate(pred,axis=0),np.concatenate(mask,axis=0),[np.concatenate(np.array(l0_list),axis=0),np.concatenate(np.array(l1_list),axis=0)
                    ,np.concatenate(np.array(l2_list),axis=0),np.concatenate(np.array(l3_list),axis=0),np.concatenate(np.array(l4_list),axis=0),np.concatenate(np.array(l5_list),axis=0)]




plt.rcParams['figure.dpi'] = 300
model=keras.models.load_model('hed.h5',custom_objects={'BilinearUpsampling':BilinearUpsampling })
img,pred,mask,l=test(model)
picture(pred[10:14],img[10:14],mask[10:14])
c=np.sum(l[0:5],axis=0)
picture((c[20:24]/5>0.3).squeeze().astype(int),img[20:24],mask[20:24])
# image.fromarray(b[0].astype(np.uint8))