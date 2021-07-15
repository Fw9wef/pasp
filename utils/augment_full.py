import json
import sys
import os
import glob
import random
import cv2
import math
import numpy as np
import itertools
import pickle
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
from PIL import ImageDraw, ImageFont
import numpy as np
import albumentations as alb
from albumentations import ImageCompression, Compose, JpegCompression, ISONoise, Rotate, RandomBrightness
from albumentations.imgaug.transforms import IAAPerspective

def compression(img, quality_lower=10, quality_upper=90, p = 1.0):
    '''Downgrades quality of img by using JPEG'''
    def compress():
        return Compose([ImageCompression(quality_lower=quality_lower, quality_upper=quality_upper, always_apply=True, p=1)])
    aug = compress()
    data = {"image": img}
    new_img = aug(**data)["image"]
    return new_img
def blur(img, shape=(3, 3)):
    """
    Imposes gaussian blur on image

    :param img: np.ndarray - image
    :param shape: tuple - blur shape
    :return: np.ndarray
    """
    a = np.random.choice([x for x in range(2,shape[0]) if x%2!=0])
    img = cv2.GaussianBlur(img, (a,a), 0)
    return img
def rescale(img, coeff=(1.1, 1.3)):
    """
    Degrades image quality

    :param img: np.ndarray - image
    :param coef: float - coefficient of rescaling
    :return: np.ndarray
    """
    coef = random.uniform(coeff[0], coeff[1])
    shape = img.shape[0:2][::-1]
    new_shape = (math.floor(shape[0] / coef), math.floor(shape[1] / coef))
    img = cv2.resize(img, new_shape)
    return cv2.resize(img, shape).astype(np.uint8)
def rotate(img, angle=4):
    def compress():
        return Compose([alb.Rotate((-angle, angle), interpolation=cv2.INTER_CUBIC, border_mode=1, value=0, mask_value=None,     always_apply=True, p=1)])
    aug = compress()
    data = {"image": img}
    new_img = aug(**data)["image"]
    return new_img
def rand_br(img, limit = 0.2):
    def compress():
        return Compose([RandomBrightness(limit=limit, always_apply=True, p=1)])
    aug = compress()
    data = {"image": img}
    new_img = aug(**data)["image"]
    return new_img
def Perspective(img, scale = (0.01,0.05)):
    def compress():
        return Compose([IAAPerspective(scale=scale, keep_size=True, always_apply=True, p=1)])
    aug = compress()
    data = {"image": img}
    new_img = aug(**data)["image"]
    return new_img
def sharpen(img, sharpen_amount=3, kernel_shape=(3, 3)):
    """
    Sharpens img

    :param img: np.ndarray - source image
    :param sharpen_amount: float - sharpen coefficient
    :param kernel_shape: tuple - shape of unit kernel
    :return:
    """
    img = img.astype(np.float)
    blurred = cv2.GaussianBlur(img, kernel_shape, 0)
    img = img + sharpen_amount * (img - blurred)
    return np.clip(img, 0, 255).astype(np.uint8)
def erosion(img, kern_size=(2,2)):
    kernel = np.ones(kern_size,np.uint8)
    erosion = cv2.dilate(img,kernel,iterations = 1)
    return erosion
def dilate(img, kern_size=(2,2)):
    kernel = np.ones(kern_size,np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    return erosion
def camera_noise(img, color_shift=(0.1, 0.9), intensity=(0.7, 0.9), always_apply=True, p=1.0):
    def noise():
        return Compose([ISONoise(color_shift=color_shift, intensity=intensity, always_apply=always_apply, p=p)])
    aug = noise()
    data = {"image": img}
    new_img = aug(**data)["image"]
    return new_img
def motion_blur(img, direction=(0, 15)):
    """
    Imposes motion blur on image

    :param img: np.array - image
    :param direction: (int, int) - direction and value of motion
    :return: np.array - image
    """
    shape = (abs(np.random.choice(range(direction[0]))) * 2 + 1, abs(np.random.choice(range(direction[1]))) * 2 + 1)
    kernel = np.zeros(shape)
    if direction[0] * direction[1] > 0:
        cv2.line(kernel, (0, shape[0] - 1), (shape[1] - 1, 0), color=1)
    else:
        cv2.line(kernel, (0, 0), (shape[1] - 1, shape[0] - 1), color=1)
    kernel /= kernel.sum() + 1e-9
    return cv2.filter2D(img, -1, kernel).astype(np.uint8)
def get_glare(size, max_value):
    """
    Returns mask of glare

    :param size: int - glare size
    :param max_value: int - brightness
    :return: np.ndarray - img with glare
    """
    def gauss(x, mu):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(1, 2.)))

    gauss_values = np.array(gauss(np.linspace(-3, 3, size), 0))
    row = gauss_values.reshape(-1, 1)
    column = gauss_values.reshape(1, -1)
    glare = row.dot(column)
    glare = glare * max_value
    return glare.astype(np.int)
def add_glares(img, number_of_glares=2, size_range=(5., 20.), value_range=(-40, 40)):
    """
    Adds some random circle glares/shadows to image
    Note: shadow is a glare with negative value

    :param img: np.ndarray - image
    :param number_of_glares: int - number of glares to add
    :param size_range: (float, float) - size of glares in % of image size
    :param value_range: (int, int) - brightness/darkness of glare/shadow
    :return: np.ndarray - image with glares
    """
    image_min_size = min(img.shape[:2])
    for _ in range(number_of_glares):
        mask = np.zeros(img.shape, dtype=np.int)
        size = int((size_range[0] + random.random() * (size_range[1] - size_range[0])) * image_min_size / 100)
        if size == 0:
            continue
        value = random.randint(value_range[0], value_range[1])
        glare = get_glare(size, value)
        if img.ndim > 2:
            glare = np.array(np.dstack([glare] * img.shape[2]))

        pos = (random.randint(0, mask.shape[0] - size), random.randint(0, mask.shape[1] - size))
        mask[pos[0]:pos[0] + size, pos[1]:pos[1] + size] = glare
        img = (mask + img).clip(0, 255)

    return img.astype(np.uint8)
def horisontal_resize(img, a=0.8, b = 1.2):
    scale_factor = random.uniform(a,b)
    new_img = cv2.resize(img.copy(), (int(img.shape[1]*scale_factor), img.shape[0]))
    return new_img

def get_projective_matrix(shape, corner_size, corner_shift=(0, 0)):
    """
    Generates random params for Projective Transformation

    :param shape: (height, width) - image shape (like np.array.shape[:2])
    :param corner_size: (int, int) - corners to choose corner points from
    :param corner_shift: (int, int) - shifting corners inward for cutting the image
    :return: np.ndarray - transform matrix
    """
    dst = np.float32((
        (0, 0),
        (0, shape[1]),
        (shape[0], 0),
        (shape[0], shape[1])
    ))

    corners = (
        (
            (corner_shift[1], corner_shift[1] + corner_size[1]),
            (corner_shift[0], corner_shift[0] + corner_size[0])
        ),
        (
            (shape[1] - corner_size[1] - corner_shift[1], shape[1] - corner_shift[1]),
            (corner_shift[0], corner_size[0] + corner_shift[0])
        ),
        (
            (corner_shift[1], corner_size[1] + corner_shift[1]),
            (shape[0] - corner_size[0] - corner_shift[0], shape[0] - corner_shift[0])
        ),
        (
            (shape[1] - corner_size[1] - corner_shift[1], shape[1] - corner_shift[1]),
            (shape[0] - corner_size[0] - corner_shift[0], shape[0] - corner_shift[0])
        )
    )

    src = np.float32([(random.randint(*corner[0]), random.randint(*corner[1])) for corner in corners])

    return cv2.getPerspectiveTransform(src, dst)


def projective_transform(img, transform_matrix):
    """
    Applies projective transform on an image.

    :param img: np.array - image
    :param src: np.array - array of source corner points
    :param dst: np.array - array of destination corner points
    :return: np.array, np.array - Transformation matrix, new image
    """
    shape = img.shape[:2]

    warped = cv2.warpPerspective(img, transform_matrix, shape, flags=cv2.INTER_LINEAR)
    if warped.ndim > 2:
        warped = np.array(warped).transpose([1, 0, 2])
    else:
        warped = np.array(warped).T
    return warped

def get_aug(img, list_of_aug, dict_param, n = 3, prob = 0.3):
    aug_imgs=[]
    for i in range(n):
        aug_img = img.copy()
        for x in list_of_aug:
            if np.random.random()<0.7:
                if x.__name__=='projective_transform':
                    m = get_projective_matrix((aug_img.shape[0],aug_img.shape[1]), *dict_param['get_projective_matrix'])
                    aug_img = x(aug_img, m)
                else:
                    aug_img = x(aug_img, *dict_param[x.__name__])
        aug_imgs.append(aug_img)
    return aug_imgs
def get_random_shadow(img):
    back = np.array(np.tile(0, img.shape), dtype='float32')
    if np.random.random()>0.5:
        a = np.random.uniform(0.05,0.6)
    else:
        a = np.random.uniform(-0.6,-0.05)
    b = np.random.randint(40,200)
    c = np.random.uniform(-3,3)
    d = np.random.uniform(0.3, 0.8)
    for i in range(back.shape[0]):
        for j in range(back.shape[1]):
            back[i][j] =1-d*1/(1+np.exp(a*(j-(c*i-b)/(-1))))
    blend_img = (back*img).astype(int)
    return np.array(blend_img, dtype = 'uint8')

def get_random_glare(img):
    back = np.array(np.tile(0, img.shape), dtype='float32')
    if np.random.random()>0.5:
        a = np.random.uniform(0.05,0.6)
    else:
        a = np.random.uniform(-0.6,-0.05)
    b = np.random.randint(40,200)
    c = np.random.uniform(-3,3)
    d = np.random.uniform(0.3, 0.8)
    for i in range(back.shape[0]):
        for j in range(back.shape[1]):
            back[i][j] =1-d*1/(1+np.exp(a*(j-(c*i-b)/(-1))))
    white = np.array(np.tile(255, img.shape), dtype='uint8')
    new_img = white - (white - img) * back
    return np.array(new_img, dtype = 'uint8')
def get_augmentation(img, prob = 0.7):
    aug_img = img.copy()
    if np.random.random()<0.35:
        if np.random.random()>0.4:
            aug_img = get_random_shadow(aug_img)
        else:
            aug_img = get_random_glare(aug_img)
    if np.random.random()<0.6:
        if np.random.random()>0.5:
            aug_img = blur(aug_img, (9,9))
        else:
            aug_img = motion_blur(aug_img, (3,3))
    if np.random.random()<prob:
        m = get_projective_matrix((aug_img.shape[0],aug_img.shape[1]), (4,4))
        aug_img = projective_transform(aug_img, m)
    if np.random.random()<prob:
        shape = aug_img.shape
        if np.random.random()>0.5:
            aug_img = rescale(aug_img)
        else:
            aug_img = horisontal_resize(aug_img)     
        aug_img = cv2.resize(aug_img, (shape[1], shape[0]))
    if np.random.random()<0.5:
        aug_img = rand_br(aug_img, (-0.4, 0.1))
        aug_img = camera_noise(aug_img)
    if np.random.random()<0.3:
        aug_img = compression(aug_img)
    return(aug_img)