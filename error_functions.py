#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> import tensorlayer as tl
    >>> outputs = tl.act.pixel_wise_softmax(outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv3d(img1, window, strides=[1,1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv3d(img2, window, strides=[1,1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv3d(img1*img1, window, strides=[1,1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv3d(img2*img2, window, strides=[1,1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv3d(img1*img2, window, strides=[1,1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def l1(x, y):

    l1 = tf.reduce_mean(tf.reduce_mean(tf.abs(x - y), [1, 2, 3, 4]))

    return l1

def dipole_kernel(pixelSize,matrixSize):
    nx,ny,nz=pixelSize
    FOVx,FOVy,FOVz=tf.multiply(pixelSize,matrixSize)
    
    kx=np.arange(-np.ceil((nx-1)/2.0),np.floor((nx-1)/2.0)+1)*1.0/FOVx
    ky=np.arange(-np.ceil((ny-1)/2.0),np.floor((ny-1)/2.0)+1)*1.0/FOVy
    kz=np.arange(-np.ceil((nz-1)/2.0),np.floor((nz-1)/2.0)+1)*1.0/FOVz
    
    KX,KY,KZ=np.meshgrid(kx,ky,kz)
    KX=KX.transpose(1,0,2)
    KY=KY.transpose(1,0,2)
    KZ=KZ.transpose(1,0,2)
    
    K2=KX**2+KY**2+KZ**2
    
    dipole_f=1.0/3-KZ**2/K2
    dipole_f=np.fft.ifftshift(dipole_f) #note ifftshift([-2,-1,0,1,2])=[0,1,2,-2,-1]
    dipole_f[0,0,0]=0
    dipole_f=dipole_f.astype('complex')
    return dipole_f


def model_loss(pred, x, m, d, input_std,input_mean,label_std,label_mean):
    # pred : output
    # x : input
    # m : mask
    # d : dipole kernel
    pred_sc = pred * label_std + label_mean
    x2 = tf.complex(pred_sc, tf.zeros_like(pred_sc))
    x2 = tf.transpose(x2, perm=[0, 4, 1, 2, 3])
    x2k = tf.signal.fft3d(x2)

    d2 = tf.complex(d, tf.zeros_like(d))
    d2 = tf.transpose(d2, perm=[0, 4, 1, 2, 3])
    fk = tf.multiply(x2k, d2)

    f2 = tf.signal.ifft3d(fk)
    f2 = tf.transpose(f2, perm=[0, 2, 3, 4, 1])
    f2 = tf.math.real(f2)

    slice_f = tf.multiply(f2, m)
    X_c = (x * input_std) + input_mean
    X_c2 = tf.multiply(X_c, m)
    return l1(X_c2, slice_f)

def model_loss_simple(pred, x, m, d):
    # pred : output
    # x : input phase
    # m : mask
    # d : dipole kernel
    x2 = tf.complex(pred, tf.zeros_like(pred))
    x2 = tf.transpose(x2, perm=[0, 4, 1, 2, 3])
    x2k = tf.signal.fft3d(x2)

    d2 = tf.complex(d, tf.zeros_like(d))
    d2 = tf.transpose(d2, perm=[0, 4, 1, 2, 3])
    fk = tf.multiply(x2k, d2)

    f2 = tf.signal.ifft3d(fk)
    f2 = tf.transpose(f2, perm=[0, 2, 3, 4, 1])
    f2 = tf.math.real(f2)

    slice_f = tf.multiply(f2, m)
    X_c2 = tf.multiply(x, m)
    return l1(X_c2, slice_f)

def grad_loss(x, y):
    x_cen = x[:, 1:-1, 1:-1, 1:-1, :]
    x_shape = tf.shape(x)
    grad_x = tf.zeros_like(x_cen)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                x_slice = tf.slice(x, [0, i+1, j+1, k+1, 0], [x_shape[0], x_shape[1]-2, x_shape[2]-2, x_shape[3]-2, x_shape[4]])
                if i*i + j*j + k*k == 0:
                    temp = tf.zeros_like(x_cen)
                else:
                    temp = tf.scalar_mul(1.0 / tf.sqrt(tf.cast(i * i + j * j + k * k, tf.float32)), tf.nn.relu(x_slice - x_cen))
                grad_x = grad_x + temp

    y_cen = y[:, 1:-1, 1:-1, 1:-1, :]
    y_shape = tf.shape(y)
    grad_y = tf.zeros_like(y_cen)
    for ii in range(-1, 2):
        for jj in range(-1, 2):
            for kk in range(-1, 2):
                y_slice = tf.slice(y, [0, ii + 1, jj + 1, kk + 1, 0],
                                   [y_shape[0], y_shape[1] - 2, y_shape[2] - 2, y_shape[3] - 2, y_shape[4]])
                if ii*ii + jj*jj + kk*kk == 0:
                    temp = tf.zeros_like(y_cen)
                else:
                    temp = tf.scalar_mul(1.0 / tf.sqrt(tf.cast(ii * ii + jj * jj + kk * kk, tf.float32)), tf.nn.relu(y_slice - y_cen))
                grad_y = grad_y + temp

    gd = tf.abs(grad_x - grad_y)
    gdl = tf.reduce_mean(gd, [1, 2, 3, 4])
    gdl = tf.reduce_mean(gdl)
    return gdl

def tv_loss(pred):
    x_cen = pred[:, 1:-1, 1:-1, 1:-1, :]
    x_shape = tf.shape(pred)
    grad_x = tf.zeros_like(x_cen)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                x_slice = tf.slice(pred, [0, i+1, j+1, k+1, 0], [x_shape[0], x_shape[1]-2, x_shape[2]-2, x_shape[3]-2, x_shape[4]])
                if i*i + j*j + k*k == 0:
                    temp = tf.zeros_like(x_cen)
                else:
                    temp = tf.scalar_mul(1.0 / tf.sqrt(tf.cast(i * i + j * j + k * k, tf.float32)), tf.nn.relu(x_slice - x_cen))
                grad_x = grad_x + temp

    gdl = l1(grad_x, 0)
    return gdl


def loss_plus(x,y):
    return mse(x,y)+0.001*grad_loss(x,y)


def total_loss(pred, x, y, m, d, w1, w2):
    l1loss = l1(pred, y)
    mdloss = model_loss_simple(pred, x, m, d)
    tvloss = tv_loss(pred)
    tloss = l1loss + mdloss * w1 + tvloss * w2
    return l1loss, mdloss, tvloss, tloss


# x for label, y for prediction
def mse(x,y):
    mse = tf.reduce_mean(tf.compat.v1.squared_difference(x, y))
    return mse

def ssim(x,y):
    ssim = tf.reduce_mean(tf.image.ssim(x, y, max_val = tf.math.reduce_max(x)-tf.math.reduce_min(x)))
    return ssim

def nrmse(x,y):
#     if tf.norm(x) is not 0:
#         nrmse = tf.norm(x-y)/tf.norm(x)
#     else:
#         nrmse = 0
    nrmse = tf.norm(x-y)/tf.norm(x)
    return nrmse

def l2(x,y):
    l2 = tf.nn.l2_loss(y-x)
    return l2

def pnsr(x,y):
    psnr = tf.reduce_mean(tf.image.psnr(x, y, max_val=tf.math.reduce_max(x)-tf.math.reduce_min(x)))


