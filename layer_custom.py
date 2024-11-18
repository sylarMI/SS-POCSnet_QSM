import tensorflow as tf

from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Concatenate, Layer
from keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization, Activation, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
from QSM_func import *
from networks_model import *

class CustGradClass:

    def __init__(self,x1,x2,x3,opt):
        self.f = tf.custom_gradient(lambda x: CustGradClass._f(self, x))
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.opt = opt
        
    @staticmethod
    def _f(self, x):
        fx = conjgrad_ss(self.x1,self.x2,self.x3,x,self.opt)
        def grad(dy):
            grad = conjgrad_ss_grad(self.x1, self.x2, self.x3, dy, self.opt) # compute gradient
            return grad
        return fx, grad


class CustomLayer(Layer):
    def __init__(self,init_x,x, opt):

        self.c = CustGradClass(x[...,0:len(opt['rad'])],init_x,x[...,len(opt['rad']):len(opt['rad'])+1],opt)
        super(CustomLayer, self).__init__()

    def call(self, inputdata):
        
        return self.c.f(inputdata)

class CustGradClass_opt:

    def __init__(self,x1,x2,x3,smv,opt):
        self.f = tf.custom_gradient(lambda x: CustGradClass_opt._f(self, x))
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.smv = smv
        self.opt = opt
        
    @staticmethod
    def _f(self, x):
        fx = conjgrad_ss_opt(self.x1,self.x2,self.x3,x,self.smv,self.opt)
        def grad(dy):
            grad = conjgrad_ss_grad_opt(self.x1, self.x2, self.x3, dy, self.smv, self.opt) # compute gradient
            return grad
        return fx, grad


class CustomLayer_opt(Layer):
    def __init__(self,init_x,x,smv,opt):
        self.c = CustGradClass_opt(x[...,0:len(opt['rad'])],init_x,x[...,len(opt['rad']):len(opt['rad'])+1],smv, opt)
        super(CustomLayer_opt, self).__init__()

    def call(self, inputdata):        
        return self.c.f(inputdata)
