import numpy as np
import math
import scipy
from scipy import ndimage
from utils import *
from QSM_func import *

class Data_loaders():
    def __init__(self, opt):
        self.root_dir = opt['train_path']
        self.batch_size = opt['batch_size']
        self.img1_path, self.img2_path, self.label_path = list_simu_data(opt)      
        self.data_size = len(self.label_path)
        self.img_height = opt['patch_size'][0] #define the dims of image, image will be resize with this dim
        self.img_width = opt['patch_size'][1]
        self.img_depth = opt['patch_size'][2]
        self.shuffled_idx = np.arange(self.data_size)
        self.absolute_ind = 0
        
    #shuffle the order of all samples
    def shuffle_all(self):
        np.random.shuffle(self.shuffled_idx)
        return self

    '''load data one by one'''
    def next(self, index, opt):
        
        '''initialize image and label arrays'''
        input1 = np.empty((self.img_height, self.img_width, self.img_depth), dtype='float32')
        input_1 = np.empty((len(index), self.img_height, self.img_width, self.img_depth, len(opt['rad'])), dtype='float32')
        input_2 = np.empty((len(index), self.img_height, self.img_width, self.img_depth, 1), dtype='float32')
        input_3 = np.empty((len(index), self.img_height, self.img_width, self.img_depth,1), dtype='float32')
        label = np.empty((len(index), self.img_height, self.img_width, self.img_depth,1), dtype='float32')
        M = np.empty((self.img_height, self.img_width, self.img_depth, len(opt['rad'])), dtype='float32')
            
        for i in range(len(index)):
            idx = index[i]
            input1_name, input2_name, label_name = self.img1_path[idx], self.img2_path[idx], self.label_path[idx] 
            input1_tmp, input2_tmp = open_img_seg(input1_name, input2_name)
            label_tmp,_           = open_img_seg(label_name)

            if not opt['is_patch']: # randomize certain batches in one epoch
                randint_height = np.random.randint(input1_tmp.shape[0] - self.img_height)
                randint_width  = np.random.randint(input1_tmp.shape[1] - self.img_width)            
                randint_depth  = np.random.randint(input1_tmp.shape[2] - self.img_depth)

                input1_tmp_c   = input1_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth]
                input2_tmp_c = input2_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 
                label_tmp_c   = label_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 

                input_1[i]  = input1_tmp_c
                input_2[i]  = input2_tmp_c*100
                label[i] = np.tanh(10*label_tmp_c)         
            else:
            
                input_1[i,...] = input1_tmp
                input_2[i,...,0] = input2_tmp
                label[i,...,0] = np.clip(label_tmp,-10,10)
                                         
        
        if opt['is_aug']:

                    
            for d in range(len(index)):
                '''random scaling'''
                if np.random.randint(2):
                    rnd = max(np.amax(label[d]),-np.amin(label[d]))#*np.random.rand()
                    rnd = max(0.01, rnd)
                    s_rnd = 1/rnd * np.random.rand()

                    if s_rnd * max(np.amax(label[d]),-np.amin(label[d])) > 1:
                        s_rnd = 1

                    input_2[d] = s_rnd * input_2[d]
                    label[d] = s_rnd * label[d]

                '''random negative value'''    
                if np.random.randint(2):
                    input_2[d] = -input_2[d]
                    label[d] = -label[d]
        return input_1, input_2, label

'''data inference'''
class Data_loaders_invivo():
    def __init__(self, opt):
        self.root_dir = opt['train_path']
        self.batch_size = opt['batch_size']
        if opt['datatype'] == 'phan':
            print(opt['datatype'])
            self.p1 = 2  # padding size for each dimension
            self.p2 = 2
            self.p3 = 5
            self.img1_path, self.img2_path, self.label_path = list_invivo_data(opt)
        if opt['datatype'] == 'phan_lr':
            print(opt['datatype'])
            self.p1 = 2 #0
            self.p2 = 2 #0
            self.p3 = 4 #5
            self.img1_path, self.img2_path, self.label_path = list_invivo_data(opt)
        if opt['datatype'] == 'cos' or opt['datatype'] == 'rc2':
            print(opt['datatype']+' cosmos')
            self.p1 = 0
            self.p2 = 0
            self.p3 = 0
            self.img1_path, self.img2_path, self.img3_path, self.label_path = list_single_data(opt)
        if opt['datatype'] == 'phan_2':
            self.p1 = 0
            self.p2 = 0
            self.p3 = 7
            self.img1_path, self.img2_path, self.img3_path, self.label_path = list_single_data(opt)
        if opt['datatype'] == 'cmb' or opt['datatype'] == 'ms':
            print(opt['datatype'])
            self.p1 = 2
            self.p2 = 0
            self.p3 = 5
            self.img1_path, self.img2_path, self.label_path = list_mb_data(opt)
            
        self.data_size = len(self.label_path)
        self.img_height = opt['patch_size'][0] #define the dims of image, image will be resize with this dim
        self.img_width = opt['patch_size'][1]
        self.img_depth = opt['patch_size'][2]
        self.shuffled_idx = np.arange(self.data_size)
        self.absolute_ind = 0
        
    #shuffle the order of all samples
    def shuffle_all(self):
        np.random.shuffle(self.shuffled_idx)
        return self

    '''load data one by one'''
    def next(self, index, opt):
        
        '''initialize image and label arrays'''
        input1 = np.empty((len(index), self.img_height, self.img_width, self.img_depth,12), dtype='float32')
        input_1 = np.empty((len(index), self.img_height, self.img_width, self.img_depth, len(opt['rad'])), dtype='float32')
        input_1k = np.empty((len(index), self.img_height, self.img_width, self.img_depth, len(opt['rad'])), dtype='float32')
        input_1m = np.empty((len(index), self.img_height, self.img_width, self.img_depth, len(opt['rad'])), dtype='float32')
        input_2 = np.empty((len(index), self.img_height, self.img_width, self.img_depth, 1), dtype='float32')
        label = np.empty((len(index), self.img_height, self.img_width, self.img_depth,1), dtype='float32')
        M = np.empty((len(index),self.img_height, self.img_width, self.img_depth, len(opt['rad'])), dtype='float32')
        msk_t = np.zeros((self.img_height, self.img_width, self.img_depth), dtype='float32')
            
        for i in range(len(index)):
            idx = index[i]
            input1_name, input2_name, label_name = self.img1_path[idx], self.img2_path[idx], self.label_path[idx] 
            input1_tmp, input2_tmp = open_img_seg(input1_name, input2_name)
            label_tmp, _           = open_img_seg(label_name)

            if not opt['is_patch'] : # randomize certain batches in one epoch
                randint_height = np.random.randint(input1_tmp.shape[0] - self.img_height)
                randint_width  = np.random.randint(input1_tmp.shape[1] - self.img_width)            
                randint_depth  = np.random.randint(input1_tmp.shape[2] - self.img_depth)

                input1_tmp_c   = input1_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth]
                input2_tmp_c = input2_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 
                label_tmp_c   = label_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 

                input_1[i]  = input1_tmp_c
                input_2[i]  = -input2_tmp_c*input1_tmp_c*10
                label[i] = label_tmp_c *input1_tmp_c*10   
            else:       

                input1[i,...]  = np.pad(input1_tmp,((self.p1,self.p1),(self.p2,self.p2),(self.p3,self.p3),(0,0)),'constant')

                for n in range(len(opt['rad'])):
                    M[i,...,n] = input1[i,...,12-opt['rad'][n]]

                    if n is 0:
                        input_1m[i,...,n:n+1] = M[i,...,n:n+1]
                    else:
                        input_1m[i,...,n:n+1] = M[i,...,n:n+1]-M[i,...,n-1:n]

                if opt['datatype'] == 'cos':
                    input_2[i,...,0] = np.pad(input2_tmp,((self.p1,self.p1),(self.p2,self.p2),(self.p3,self.p3)),'constant')/(2*math.pi*127*0.025)  #in ppm unit
                elif opt['datatype'] == 'rc2':
                    input_2[i,...,0] = np.pad(input2_tmp,((self.p1,self.p1),(self.p2,self.p2),(self.p3,self.p3)),'constant')/(2*math.pi*7*42.775*0.008)
                elif opt['datatype'] == 'phan_e':
                    input_1m[i,...] = np.where((input_1m[i,...]==1) & (np.tile(label[i,...],len(opt['rad'])) < -0.99),0,input_1m[i,...])
                    input_2[i,...,0] = np.pad(input2_tmp,((self.p1,self.p1),(self.p2,self.p2),(self.p3,self.p3)),'constant')
                else:
                    input_2[i,...,0] = np.pad(input2_tmp,((self.p1,self.p1),(self.p2,self.p2),(self.p3,self.p3)),'constant')/(2*math.pi*127*0.023)

                label[i,...,0] = np.clip(np.pad(label_tmp,((self.p1,self.p1),(self.p2,self.p2),(self.p3,self.p3)),'constant')*M[...,-1],-10,10)

        return input_1m, input_2, label
    
    
'''patch-based inference and combine'''    
class Data_loaders_infer():
    def __init__(self, root_dir, input_1, input_2, label, index, patch_size, patch_num):
        self.root_dir = root_dir
        self.num_labels = 2 
        self.batch_size = patch_num[0]*patch_num[1]*patch_num[2] #get batch size
        self.nil_img_path, self.nil_liv_path,\
        self.nil_les_path = list_invivo_data(root_dir, input_1, input_2, label, index)  
        self.data_size = len(self.nil_img_path)
        self.batch_idx_max = np.floor_divide(self.data_size,self.batch_size) #max batch size
        self.img_height = patch_size[0] #define the dims of image, image will be resize with this dim
        self.img_width = patch_size[1]
        self.img_depth = patch_size[2]
        self.patch_size = patch_size
        self.absolute_ind = 0
        self.patch_num = patch_num
        
    #shuffle the order of all samples
    def shuffle_all(self):
        np.random.shuffle(self.shuffled_idx)
        return self
    
    #read one batch of images and labels from files 
    def next(self, isbatch, rand_flip = 0):
        
        #initial image and label arrays
        image1 = np.empty((self.batch_size, self.img_height, self.img_width, self.img_depth), dtype='float32')
        image2 = np.empty((self.batch_size, self.img_height, self.img_width, self.img_depth), dtype='float32')
        label = np.empty((self.batch_size, self.img_height, self.img_width, self.img_depth), dtype='float32')
        strides = np.zeros(3)
        
        # read image and label
        idx = self.absolute_ind
        img1_name, img2_name, label_name = self.nil_img_path[idx], self.nil_liv_path[idx], self.nil_les_path[idx]
        img1_tmp, img2_tmp = open_img_seg(img1_name, img2_name)
        label_tmp,_           = open_img_seg(label_name)
        img1_tmp = img1_tmp.astype('float32')
        img2_tmp = img2_tmp.astype('float32')
        self.absolute_ind = self.absolute_ind + 1

        for i in range(3):
            if self.patch_num[i] == 1:
                strides[i] = 0
            else:
                strides[i] = (img1_tmp.shape[i] - self.patch_size[i]) // (self.patch_num[i] - 1)
        strides = strides.astype('int')
        
        z = 0
        for i in range(self.patch_num[0]):
            for j in range(self.patch_num[1]):
                for k in range(self.patch_num[2]):
                    image1_tmp_c = img1_tmp[
                                   i * strides[0]:i * strides[0] + self.patch_size[0],
                                   j * strides[1]:j * strides[1] + self.patch_size[1],
                                   k * strides[2]:k * strides[2] + self.patch_size[2]]
                    image2_tmp_c = img2_tmp[
                                   i * strides[0]:i * strides[0] + self.patch_size[0],
                                   j * strides[1]:j * strides[1] + self.patch_size[1],
                                   k * strides[2]:k * strides[2] + self.patch_size[2]]
                    label_tmp_c = label_tmp[
                                   i * strides[0]:i * strides[0] + self.patch_size[0],
                                   j * strides[1]:j * strides[1] + self.patch_size[1],
                                   k * strides[2]:k * strides[2] + self.patch_size[2]]
                    
                    image1[z]  = image1_tmp_c
                    image2[z]  = image2_tmp_c*100
                    label[z] = 100*label_tmp_c/(2*math.pi*127*0.023)
                    z=z+1
                    
        image1 = np.reshape(image1,[self.batch_size, self.img_height, self.img_width, self.img_depth, 1])
        image2 = np.reshape(image2,[self.batch_size, self.img_height, self.img_width, self.img_depth, 1])
        label = np.reshape(label,[self.batch_size, self.img_height, self.img_width, self.img_depth, 1])
        return image1, image2, label    