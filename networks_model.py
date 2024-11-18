import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Concatenate, Layer, Lambda, Add
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras import backend as K
import numpy as np
import math
from layer_custom import *

"""base unet used"""    

def base_unet(opt, out_size, f_size=8, ker_size=(3, 3, 3)):
    conv_1c0 = Conv3D(f_size, kernel_size=(1, 1, 1), padding='same', activation='relu')
    conv_1c1 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1c12 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation='relu')
    pool1 = Conv3D(2 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), activation='relu')

    conv_1c2 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1c22 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    pool2 = Conv3D(4 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), activation='relu')

    conv_1c3 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1c32 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    pool3 = Conv3D(8 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), activation='relu')
    
    conv_1c4 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1c42 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    pool4 = Conv3D(16 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), activation='relu')
    upS4 = Conv3DTranspose(8 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), activation='relu')
    
    conv_1e3 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1e32 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1e33 = Conv3D(8 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    upS3 = Conv3DTranspose(4 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), activation='relu')
    
    conv_1e2 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1e22 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1e23 = Conv3D(4 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    upS2 = Conv3DTranspose(2 * f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), activation='relu')
    
    conv_1e1 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1e12 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1e13 = Conv3D(2 * f_size, kernel_size=ker_size, padding='same', activation='relu')

    upS1 = Conv3DTranspose(f_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), activation='relu')
    conv_1e0 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1e02 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation='relu')
    conv_1e03 = Conv3D(f_size, kernel_size=ker_size, padding='same', activation='relu')

    conv_output_tc = Conv3D(out_size, kernel_size=(1, 1, 1), padding='same', activation='tanh')
    
    input_img = Input(shape=opt['in_shape'])
    # unet start
    c0 = conv_1c0(input_img)
    c1 = conv_1c1(c0)
    c1 = Add()([conv_1c12(c1), c0])

    c1_ = pool1(c1)
    c2 = conv_1c2(c1_)
    c2 = Add()([conv_1c22(c2), c1_])

    c2_ = pool2(c2)
    c3 = conv_1c3(c2_)
    c3 = Add()([conv_1c32(c3), c2_])

    c3_ = pool3(c3)
    c4 = conv_1c4(c3_)
    c4 = Add()([conv_1c42(c4), c3_])

    c5 = pool4(c4)

    e1 = upS4(c5)
    e12 = conv_1e3(Concatenate(axis=-1)([e1, c4]))
    e13 = conv_1e32(e12)
    e13 = Add()([conv_1e33(e13), e1])

    e2 = upS3(e13)
    e22 = conv_1e2(Concatenate(axis=-1)([e2, c3]))
    e23 = conv_1e22(e22)
    e23 = Add()([conv_1e23(e23), e2])

    e3 = upS2(e23)
    e32 = conv_1e1(Concatenate(axis=-1)([e3, c2]))
    e33 = conv_1e12(e32)
    e33 = Add()([conv_1e13(e33), e3])

    e4 = upS1(e33)
    e42 = conv_1e0(Concatenate(axis=-1)([e4, c1]))
    e43 = conv_1e02(e42)
    e43 = Add()([conv_1e03(e43), e4])

    output = conv_output_tc(e43)

    model_net = Model(inputs = [input_img], outputs = [output])
    return model_net

def cg_grad_model(x,x_init,smv,opt):
    x_ref = Input(shape=opt['in_shape'])
    cus_grad = CustomLayer_opt(x_init,x,smv,opt)(x_ref)
    model_df = Model(inputs=[x_ref], outputs=[cus_grad])
    return model_df

def joint_model(smv,opt):
    X = Input(shape=opt['img_shape'])
    L = Lambda(lambda x: tf.zeros_like(X[...,1:2]))(X)
    model_vnet = base_unet(opt, 1)
    model_CGgrad = cg_grad_model(X,L,smv,opt)

    x_k = model_CGgrad(L)
    for it in range(opt['iter']):
        net_out = model_vnet(x_k)
        x_k = model_CGgrad(net_out)
        
    model_joint = Model(inputs = X, outputs = [net_out, x_k], name='joint_model')
    return model_joint