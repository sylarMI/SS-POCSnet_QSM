import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os
from cv2 import resize
import cv2

def list_simu_data(opt):
    img1_path = []
    img2_path = []
    label_path = [] 
    for ind in range(len(opt['train_index'])):
        img1_path.append(opt['train_path'] + opt['train_img1_path'][0] + str(opt['train_index'][ind])+'.nii.gz')
        img2_path.append(opt['train_path'] + opt['train_img2_path'][0] + str(opt['train_index'][ind])+'.nii.gz')
        label_path.append(opt['train_path'] + opt['train_label_path'][0]+ str(opt['train_index'][ind])+'.nii.gz')
#         img1_path.append(opt['train_path'] + opt['train_img1_path'][1] + str(opt['train_index'][ind])+'.nii.gz')
#         img2_path.append(opt['train_path'] + opt['train_img2_path'][1] + str(opt['train_index'][ind])+'.nii.gz')
#         label_path.append(opt['train_path'] + opt['train_label_path'][1]+ str(opt['train_index'][ind])+'.nii.gz')
#         img1_path.append(opt['train_path'] + opt['train_img1_path'][2] + str(opt['train_index'][ind])+'.nii.gz')
#         img2_path.append(opt['train_path'] + opt['train_img2_path'][2] + str(opt['train_index'][ind])+'.nii.gz')
#         label_path.append(opt['train_path'] + opt['train_label_path'][2]+ str(opt['train_index'][ind])+'.nii.gz')
        
    return img1_path, img2_path, label_path

# %% qsm in vivo data
def list_invivo_data(opt):
    nil_img_path = []
    nil_liv_path = []
    nil_les_path = []
    
    for shape1 in range(len(opt['train_index'])):
        sub_fol_path = opt['train_path'] 
        nil_img_path.append(sub_fol_path + opt['train_img1_path']+str(opt['train_index'][shape1, 0])+'-'+str(opt['train_index'][shape1, 1])+'.nii.gz')
        nil_liv_path.append(sub_fol_path + opt['train_img2_path']+str(opt['train_index'][shape1, 0])+'-'+str(opt['train_index'][shape1, 1])+'.nii.gz')
        nil_les_path.append(sub_fol_path + opt['train_label_path']+str(opt['train_index'][shape1, 0])+'-'+str(opt['train_index'][shape1, 1])+'.nii.gz')

    return nil_img_path, nil_liv_path,nil_les_path

def list_mb_data(opt):
    nil_img_path = []
    nil_liv_path = []
    nil_les_path = []
    
    for shape1 in range(len(opt['train_index'])):
        sub_fol_path = opt['train_path'] 
        nil_img_path.append(sub_fol_path + opt['train_img1_path']+str(opt['train_index'][shape1])+'.nii.gz')
#         nil_img_path.append(sub_fol_path + opt['train_img1_path']+'.nii')
        nil_liv_path.append(sub_fol_path + opt['train_img2_path']+str(opt['train_index'][shape1])+'.nii.gz')
        nil_les_path.append(sub_fol_path + opt['train_label_path']+str(opt['train_index'][shape1])+'.nii.gz')
        
    return nil_img_path, nil_liv_path,nil_les_path

# %% single qsm in vivo data
def list_single_data(opt):
    nil_img_path = []
    nil_liv_path = []
    nil_les_path = []
    
    sub_fol_path = opt['train_path'] 
    nil_img_path.append(sub_fol_path + opt['train_img1_path'] + '.nii.gz')
    nil_liv_path.append(sub_fol_path + opt['train_img2_path'] + '.nii.gz')
    nil_les_path.append(sub_fol_path + opt['train_label_path'] + '.nii.gz')

    return nil_img_path, nil_liv_path,nil_les_path

def list_invivo_batch(opt):
    nil_img_path = []
    nil_liv_path = []
    nil_les_path = []
    sub_fol_path = opt['train_path']
    for shape1 in range(len(opt['index1'])):
        for shape2 in range(len(opt['index2'])):           
            nil_img_path.append(sub_fol_path + opt['train_img1_path']+str(opt['index1'][shape1])+'-'+str(opt['index2'][shape2])+'.nii.gz')
            nil_liv_path.append(sub_fol_path + opt['train_img2_path']+str(opt['index1'][shape1])+'-'+str(opt['index2'][shape2])+'.nii.gz')
            nil_les_path.append(sub_fol_path + opt['train_label_path']+str(opt['index1'][shape1])+'-'+str(opt['index2'][shape2])+'.nii.gz')

    return nil_img_path, nil_liv_path,nil_les_path


def open_img_seg(nil_img_path,nil_seg_path = None):
    ni1_img = nib.load(nil_img_path)
    img = ni1_img.get_fdata()
    if nil_seg_path is not None:
        nil_seg = nib.load(nil_seg_path)
        seg = nil_seg.get_fdata()
    else:
        seg = None
    return img, seg


def findcenter3d(img):
    h_pos = np.array(np.where(np.sum(img,axis=(1,2))>0)).squeeze()
    w_pos = np.array(np.where(np.sum(img,axis=(0,2))>0)).squeeze()
    d_pos = np.array(np.where(np.sum(img,axis=(0,1))>0)).squeeze()
    if not h_pos.size:
        h_pos = img.shape[0]/2
    if not w_pos.size:
        w_pos = img.shape[1]/2
    if not d_pos.size:
        d_pos = img.shape[2]/2
    return int(np.mean(h_pos)), int(np.mean(w_pos)), int(np.mean(d_pos))

def display_slice(display_num, Pred, Pred1, Label):
    fig = plt.figure(figsize=(12,10))
    nonorm = matplotlib.colors.NoNorm()
    col = np.size(display_num)
    raw = 4
    sub = 0
    for i in range(col):
        subplot = fig.add_subplot(raw, col, i + 1)
        subplot.set_xticks([]), subplot.set_yticks([])
#         im=subplot.imshow(np.rot90(np.clip(Pred[0,:,:,display_num[i],0], -0.1, 0.1) * 5 + 0.5, -1),cmap = plt.cm.gray, norm=nonorm)
        im=subplot.imshow(np.rot90(Pred[sub,:,:,display_num[i],0], -1))
#         plt.colorbar(im,subplot)
        if i == 0:
            subplot.set_ylabel('Prediction', fontsize=18)
        
        subplot = fig.add_subplot(raw, col, i + 1 +col)
        subplot.set_xticks([]), subplot.set_yticks([])
#         im=subplot.imshow(np.rot90(np.clip(Pred[0,:,:,display_num[i],0], -0.1, 0.1) * 5 + 0.5, -1),cmap = plt.cm.gray, norm=nonorm)
        im=subplot.imshow(np.rot90(Pred1[sub,:,:,display_num[i],0], -1))
#         plt.colorbar(im,subplot)
        if i == 0:
            subplot.set_ylabel('Prediction_xk', fontsize=18)
        
        subplot = fig.add_subplot(raw, col, i + 1 + col*2)
        subplot.set_xticks([]), subplot.set_yticks([])
#         im=subplot.imshow(np.rot90(np.clip(Label[0,:,:,display_num[i],0], -0.1, 0.1) * 5 + 0.5,-1),
#                          cmap = plt.cm.gray, norm=nonorm)
        im=subplot.imshow(np.rot90(Label[sub,:,:,display_num[i],0], -1))
#         plt.colorbar(im,subplot)
        if i == 0:
            subplot.set_ylabel('Label', fontsize=18)
             
        subplot = fig.add_subplot(raw, col, i + 1 + col*3)
        subplot.set_xticks([]), subplot.set_yticks([])
#         im=subplot.imshow(np.rot90(np.clip((Label[0,:,:,display_num[i],0]-Pred[0,:,:,display_num[i],0]),
#                                           -0.1, 0.1) * 5 + 0.5, -1))
        im=subplot.imshow(np.rot90(Label[sub,:,:,display_num[i],0]-Pred[sub,:,:,display_num[i],0], -1))
#         plt.colorbar(im,subplot)
        if i == 0:
            subplot.set_ylabel('Dif', fontsize=18)
    plt.show()
    plt.close()
    
'''display the error line'''    
def display_error(x, y):
    fig = plt.figure(figsize=(8,3))
    plt.plot(x,y,'bo-',linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    plt.close()  

def save_nii(data, voxel_size,  save_folder, name):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    nifti_affine = np.array([[voxel_size[0],0,0,voxel_size[0]], [0,voxel_size[1],0,voxel_size[1]], [0,0,voxel_size[2],voxel_size[2]], [0,0,0,1]], dtype=np.float)

    #data = np.fliplr(data) 
    nifti = nib.Nifti1Image(data, affine=nifti_affine)
    nib.save(nifti, os.path.join(save_folder, name + '.nii.gz')) 
    
def resize3d(img, shape):
    tmp = resize(img, dsize=(shape[0], shape[1]), interpolation=cv2.INTER_LINEAR)
    tmp_z = np.transpose(
        resize(np.transpose(tmp, (2, 0, 1)), dsize=(shape[1],shape[2]),
               interpolation=cv2.INTER_LINEAR), (1, 2, 0))  # INTER_LANCZOS4
    return tmp_z