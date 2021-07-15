import os
import cv2
import torch
import random
import numpy as np
import torchvision.transforms.functional as FT

from math import sin, cos, radians, degrees, atan, sqrt, pi
from PIL import Image, ImageDraw
from .image_transformer import ImageTransformer
from .augment_full import add_glares, motion_blur

path_to_bg_imgs = "/general/prj/pasp_detector/room_scenes/"
bg_imgs_list = [path_to_bg_imgs + x for x in os.listdir(path_to_bg_imgs) if '.jpg' in x]
total_bg_imgs = len(bg_imgs_list)
def load_bg_img():
    img_path = bg_imgs_list[np.random.choice(total_bg_imgs)]
    img = np.array(Image.open(img_path))
    return img


def perspective_rotation(img, box, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, padding=True):
    #gamma is simple angle
    it = ImageTransformer()
    
    img_mask, object_bnd = None, None
    if padding:
        old_h, old_w, _ = img.shape
        pad_v = int(max(old_h, old_w) * (np.sqrt(2)-1) / 2)+1
        img_mask = np.ones(img.shape, dtype=np.uint8)*255
        img_mask = cv2.copyMakeBorder(img_mask, pad_v, pad_v, pad_v, pad_v, cv2.BORDER_CONSTANT)
        img = cv2.copyMakeBorder(img, pad_v, pad_v, pad_v, pad_v, cv2.BORDER_CONSTANT)
        box = [crd + pad_v for crd in box]
        img_mask = it.rotate_along_axis(img_mask, theta, phi, gamma, dx, dy, dz)
        img_mask = np.where(img_mask>126, 255, 0).astype(np.uint8)
        y_with_non_zero_entry = np.array([np.any(img_mask[i,:]>0) for i in range(img_mask.shape[0])])
        x_with_non_zero_entry = np.array([np.any(img_mask[:,i]>0) for i in range(img_mask.shape[1])])
        y_with_non_zero_entry = np.where(y_with_non_zero_entry == True)[0]
        x_with_non_zero_entry = np.where(x_with_non_zero_entry == True)[0]
        y_top, y_bot = y_with_non_zero_entry[0], y_with_non_zero_entry[-1]
        x_left, x_right = x_with_non_zero_entry[0], x_with_non_zero_entry[-1]
        object_bnd = [y_top, y_bot, x_left, x_right]
    
    transformed_img = it.rotate_along_axis(img, theta, phi, gamma, dx, dy, dz)
    
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    
    xs1, ys1, s1 = np.dot(it.mat, np.array([[x1], [y1], [1]])).flatten()
    new_x1, new_y1 = int(np.round(xs1/s1)), int(np.round(ys1/s1))
    xs2, ys2, s2 = np.dot(it.mat, np.array([[x2], [y2], [1]])).flatten()
    new_x2, new_y2 = int(np.round(xs2/s2)), int(np.round(ys2/s2))
    xs3, ys3, s3 = np.dot(it.mat, np.array([[x3], [y3], [1]])).flatten()
    new_x3, new_y3 = int(np.round(xs3/s3)), int(np.round(ys3/s3))
    xs4, ys4, s4 = np.dot(it.mat, np.array([[x4], [y4], [1]])).flatten()
    new_x4, new_y4 = int(np.round(xs4/s4)), int(np.round(ys4/s4))
    
    new_box = [new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4]
    
    return transformed_img, new_box, img_mask, object_bnd


def align_box(img, angle=0, box=None):
    w, h = img.shape[:2]
    center = w / 2, h / 2
    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1)
    
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    xx = np.array([x1,x2,x3,x4])
    yy = np.array([y1,y2,y3,y4])
    
    new_x1, new_y1 = np.dot(rotation_mat, np.array([[x1], [y1], [1]])).flatten()
    new_x2, new_y2 = np.dot(rotation_mat, np.array([[x2], [y2], [1]])).flatten()
    new_x3, new_y3 = np.dot(rotation_mat, np.array([[x3], [y3], [1]])).flatten()
    new_x4, new_y4 = np.dot(rotation_mat, np.array([[x4], [y4], [1]])).flatten()
    
    xx_inds = np.argsort([new_x1, new_x2, new_x3, new_x4])
    yy_inds = np.argsort([new_y1, new_y2, new_y3, new_y4])
    top_inds, bot_inds = set(yy_inds[:2]), set(yy_inds[2:])
    left_inds, right_inds = set(xx_inds[:2]), set(xx_inds[2:])
    
    tl_ind = (top_inds.intersection(left_inds)).pop()
    tl_x, tl_y = xx[tl_ind], yy[tl_ind]
    
    tr_ind = (top_inds.intersection(right_inds)).pop()
    tr_x, tr_y = xx[tr_ind], yy[tr_ind]
    
    br_ind = (bot_inds.intersection(right_inds)).pop()
    br_x, br_y = xx[br_ind], yy[br_ind]
    
    bl_ind = (bot_inds.intersection(left_inds)).pop()
    bl_x, bl_y = xx[bl_ind], yy[bl_ind]
    
    #TopLeft,TopRight,BottomRight,BottomLeft
    return [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]


def transform(path2img, box, angle, orient, split, cut=False):
    """
    Apply the transformations above.

    :param path2img: a path to image, str
    :param box: bounding boxes in boundary coordinates, a list of dimensions (8)
    :param angle: current image tilt angle , a scalar (float)
    :param orient: current image orientation, a scalar (int)
    :param split : current sample, str
    :return: transformed image, transformed bounding box coordinates, transformed angle
    """
    assert split in {'TRAIN', 'TEST', 'VAL'}
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = cv2.imread(path2img)
    box = align_box(img, angle - orient, box)
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    
    #fist - resize initial image to reduce computational costs
    if cut:
        if np.random.rand()<0.5:
            img, box = aug_in_the_wild(img, box)
        else:
            img, box = aug_a4(img, box)
    
    else:
        if np.random.rand()<0.5:
            img, box = flip_transform(img, box)
    
    if img.shape != (300,300,3):
        img, box = resize_obj_img(img, (300, 300), box[:8])
        cxcywh = get_cxcywh(box)
        box = box + cxcywh
    
    x1, y1, x2, y2, x3, y3, x4, y4, cx, cy, w, h = box
    image = Image.fromarray(img[:, :, ::-1].astype(np.uint8))
    box8points = torch.FloatTensor([x1, y1, x2, y2, x3, y3, x4, y4])
    target = torch.FloatTensor([cx, cy, w, h, 0, 0, 0, 1])
    
    #if split == 'TRAIN':
    #    image = photometric_distort(image)
    
    target_angle = get_angle(box)
    target[4:-1] = torch.FloatTensor([target_angle, sin(radians(target_angle)), cos(radians(target_angle))])
    
    #Normalization
    #draw_io(image, box8points, target, path2img)
    image = FT.to_tensor(image)
    image = FT.normalize(image, mean=mean, std=std)
    return image, box8points, target


def sample_angles():
    #theta = random.randint(-5, 5)
    #phi = random.randint(-5, 5)
    theta, phi = 0, 0
    gamma = random.randint(0, 359)
    return theta, phi, gamma


def aug_a4(obj_img, box):
    bg_img = load_random_color_bg()
    h_img, w_img = bg_img.shape[:2]
    
    h_obj, w_obj = obj_img.shape[:2]
    relative_size = (h_img*w_img)/(h_obj*w_obj)
    target_relative_size = 3.5+1.*np.random.rand()
    compress = np.sqrt(relative_size/target_relative_size)
    diag_side_ratio = min(w_img, h_img)/np.sqrt(h_obj**2+w_obj**2)
    compress = diag_side_ratio if compress>diag_side_ratio else compress
    new_w_obj, new_h_obj = int(w_obj*compress), int(h_obj*compress)
    
    obj_img, box = resize_obj_img(obj_img, (new_w_obj, new_h_obj), box)
    rotated_img, obj_mask, box = random_rotation(obj_img, box)
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    h_obj, w_obj = rotated_img.shape[:2]
    
    shift_x, shift_y = random.randint(0, w_img - w_obj), random.randint(0, h_img - h_obj)
    temp_obj_img = np.zeros((h_img, w_img, 3))
    temp_obj_img[shift_y:shift_y+h_obj, shift_x:shift_x+w_obj] = rotated_img
    temp_obj_mask = np.zeros((h_img, w_img, 3))
    temp_obj_mask[shift_y:shift_y+h_obj, shift_x:shift_x+w_obj] = obj_mask/255
    rotated_img = temp_obj_img * temp_obj_mask + bg_img * (1 - temp_obj_mask)
    x1, x2, x3, x4 = [x + shift_x for x in [x1, x2, x3, x4]]
    y1, y2, y3, y4 = [y + shift_y for y in [y1, y2, y3, y4]]
    box = [x1, y1, x2, y2, x3, y3, x4, y4]
    cxcywh = get_cxcywh(box)
    target_box = box + cxcywh
    return rotated_img, target_box


def flip_transform(img, box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    h, w = img.shape[:2]
    img = cv2.flip(img, -1)
    x1, x2, x3, x4 = [w - x for x in [x1, x2, x3, x4]]
    y1, y2, y3, y4 = [h - y for y in [y1, y2, y3, y4]]
    box = [x1, y1, x2, y2, x3, y3, x4, y4]
    cxcywh = get_cxcywh(box)
    box = box + cxcywh
    return img, box


def load_random_color_bg():
    t = np.random.rand()
    if t<0.5:
        compress_coef = np.random.choice([0.7265, 0.7071, 0.7269])
        w_img = int(300 / compress_coef)
        h_img = 300
    else:
        compress_coef = np.random.choice([0.7265, 0.7071, 0.7269])
        w_img = 300
        h_img = int(300 / compress_coef)
    
    t = np.random.rand()
    bg = np.zeros((h_img,w_img,3))
    if t<0.5:
        bg[:,:] = np.random.randint(0,255,3)
    else:
        hue = np.random.randint(0,255)
        sat = np.random.randint(0,10)
        val = np.random.randint(230,255)
        hsv_color = np.uint8([[[hue,sat,val]]])  
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        bg[:,:] = bgr_color
    return bg


def aug_in_the_wild(obj_img, box):
    bg_img = load_bg_img()
    h_img, w_img = 300, 300
    t = np.random.rand()
    if t<0.5:
        compress_coef = np.random.choice([0.5625, 0.75, 1])
        w_img = int(300 / compress_coef)
        h_img = 300
    else:
        compress_coef = np.random.choice([0.5625, 0.75, 1])
        w_img = 300
        h_img = int(300 / compress_coef)
    bg_img = cv2.resize(bg_img, (w_img, h_img), cv2.INTER_CUBIC)
    
    h_obj, w_obj = obj_img.shape[:2]
    relative_size = (h_img*w_img)/(h_obj*w_obj)
    target_relative_size = 1.2+1.3*np.random.rand()   # maintain relative size in range 1.2 to 2.5
    compress = np.sqrt(relative_size/target_relative_size)
    diag_side_ratio = min(w_img, h_img)/np.sqrt(h_obj**2+w_obj**2)
    compress = diag_side_ratio if compress>diag_side_ratio else compress
    new_w_obj, new_h_obj = int(w_obj*compress), int(h_obj*compress)
    
    obj_img, box = resize_obj_img(obj_img, (new_w_obj, new_h_obj), box)
    rotated_img, obj_mask, box = random_rotation(obj_img, box)
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    h_obj, w_obj = rotated_img.shape[:2]
    
    if np.random.rand()<0.8:
        obj_mask = crop(obj_mask, size=(30, 60), n_crops=2)
    
    shift_x, shift_y = random.randint(0, w_img - w_obj), random.randint(0, h_img - h_obj)
    
    if np.random.rand()<0.5:
        obj_cx, obj_cy = int(shift_x + w_obj/2), int(shift_y + h_obj/2)
        rotated_img = cv2.seamlessClone(rotated_img, bg_img, obj_mask, (obj_cx, obj_cy), cv2.NORMAL_CLONE)
        #x_l, y_t, x_r, y_t, x_r, y_b, x_l, y_b
        #TopLeft,TopRight,BottomRight,BottomLeft
        x1, x2, x3, x4 = [x_crd + shift_x for x_crd in (x1, x2, x3, x4)]
        y1, y2, y3, y4 = [y_crd + shift_y for y_crd in (y1, y2, y3, y4)]
        box = [x1, y1, x2, y2, x3, y3, x4, y4]
        cxcywh = get_cxcywh(box)
        target_box = box + cxcywh
    
    else:
        temp_obj_img = np.zeros((h_img, w_img, 3))
        temp_obj_img[shift_y:shift_y+h_obj, shift_x:shift_x+w_obj] = rotated_img
        temp_obj_mask = np.zeros((h_img, w_img, 3))
        temp_obj_mask[shift_y:shift_y+h_obj, shift_x:shift_x+w_obj] = obj_mask/255
        rotated_img = temp_obj_img * temp_obj_mask + bg_img * (1 - temp_obj_mask)
        x1, x2, x3, x4 = [x + shift_x for x in [x1, x2, x3, x4]]
        y1, y2, y3, y4 = [y + shift_y for y in [y1, y2, y3, y4]]
        box = [x1, y1, x2, y2, x3, y3, x4, y4]
        cxcywh = get_cxcywh(box)
        target_box = box + cxcywh
    
    rotated_img = add_glares(rotated_img, number_of_glares=4, size_range=(50, 80), value_range=(-100, 200))
    if np.random.rand()<0.6:
        rotated_img = motion_blur(rotated_img, direction=(6, 6))
    return rotated_img, target_box


def crop(img, size=(30, 60), n_crops=4):
    h, w = img.shape[:2]
    for i in range(n_crops):
        crop_size = int(size[0]+size[1]*np.random.rand())
        shift_x, shift_y = random.randint(0, w - crop_size), random.randint(0, h - crop_size)
        img[shift_y:shift_y+crop_size, shift_y:shift_y+crop_size] = 0
    return img


def get_cxcywh(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    cx1, cy1 = np.array([x1,y1]) + np.array([x3-x1,y3-y1])/2
    cx2, cy2 = np.array([x2,y2]) + np.array([x4-x2,y4-y2])/2
    cx, cy = np.mean([cx1, cx2]), np.mean([cy1, cy2])
    w = np.mean([np.linalg.norm([x1-x2, y1-y2]), np.linalg.norm([x3-x4, y3-y4])])
    h = np.mean([np.linalg.norm([x1-x4, y1-y4]), np.linalg.norm([x2-x3, y2-y3])])
    return [cx, cy, w, h]


def random_rotation(obj_img, box):
    theta, phi, gamma = sample_angles()
    rotated_img, box, obj_mask, object_bnd = perspective_rotation(obj_img, box, theta, phi, gamma)
    if len(obj_mask.shape) < 3:
        obj_mask = np.expand_dims(obj_mask, -1)
    if obj_mask.shape[-1] == 1:
        obj_mask = np.repeat(obj_mask, 3)
    y_bnd_top, y_bnd_bot, x_bnd_left, x_bnd_right = object_bnd
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    rotated_img = rotated_img[y_bnd_top:y_bnd_bot, x_bnd_left:x_bnd_right]
    obj_mask = obj_mask[y_bnd_top:y_bnd_bot, x_bnd_left:x_bnd_right]
    x1, x2, x3, x4 = [ x - x_bnd_left for x in [x1, x2, x3, x4]]
    y1, y2, y3, y4 = [ y - y_bnd_top for y in [y1, y2, y3, y4]]
    new_box = [x1, y1, x2, y2, x3, y3, x4, y4]
    return rotated_img, obj_mask, new_box


def resize_obj_img(obj_img, new_wh, box):
    x_compress, y_compress = new_wh[0]/obj_img.shape[1], new_wh[1]/obj_img.shape[0]
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    obj_img = cv2.resize(obj_img, new_wh, cv2.INTER_CUBIC)
    x1, x2, x3, x4 = [ np.round(int(x*x_compress)) for x in [x1, x2, x3, x4]]
    y1, y2, y3, y4 = [ np.round(int(y*y_compress)) for y in [y1, y2, y3, y4]]
    new_box = [x1, y1, x2, y2, x3, y3, x4, y4]
    return obj_img, new_box


def get_angle(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    v1 = np.array([x1-x2, y1-y2])
    v1_ort = np.array([v1[1], -v1[0]])
    v2 = np.array([x1-x2, y1-y2])
    v2_ort = np.array([v2[1], -v2[0]])
    mean_vector = (v1_ort+v2_ort)/2
    target_vector = np.array([0,1])
    angle = np.degrees(np.arccos(np.sum(mean_vector*target_vector)/np.linalg.norm(mean_vector)))
    if mean_vector[0]<0:
        angle = 360-angle
    return angle


def create_prior_boxes(angle):
    """
    :return: prior boxes in top_left-bottom_right coordinates, a tensor of dimensions (250, 4)
    """
    fmap_dims = {'conv8_2': 10, 'conv9_2': 5}

    obj_scales = {'conv8_2': 0.2, 'conv9_2': 0.4}

    aspect_ratios = {'conv8_2': 6, 'conv9_2': 5}

    fmaps = list(fmap_dims.keys())

    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                ratio = aspect_ratios[fmap]

                prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

    prior_boxes = torch.FloatTensor(prior_boxes)  # (250, 4)
    prior_boxes.clamp_(0, 1)  # (250, 4)
    n_priors = prior_boxes.shape[0]
    # torch.randint(-180, 180, (n_priors, 1), dtype=torch.float32)
    return torch.cat([prior_boxes, angle * torch.ones(size=(n_priors, 1))], 1)


def photometric_distort(image):
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                adjust_factor = random.uniform(-.25, .25)
            elif d.__name__ is 'adjust_saturation':
                adjust_factor = random.uniform(.0, .5)
            elif d.__name__ is 'adjust_brightness':
                adjust_factor = random.uniform(.5, 1.2)
            else:
                adjust_factor = random.uniform(.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def cxcya_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h,sin(angle),cos(angle)) to
    boundary coordinates (TopLeft,TopRight, BottomRight,LeftBottom).

    :param cxcy: bounding boxes in center-size coordinates with angle degree, a tensor of size (n_boxes, 5)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 8)
    """

    top_left = cxcy[:, :2] - (cxcy[:, 2:4] / 2)
    bottom_right = cxcy[:, :2] + (cxcy[:, 2:4] / 2)
    top_right = torch.cat([bottom_right[:, 0].view(-1, 1), top_left[:, 1].view(-1, 1)], 1)
    left_bottom = torch.cat([top_left[:, 0].view(-1, 1), bottom_right[:, 1].view(-1, 1)], 1)

    cx, cy = cxcy[:, 0].view(-1, 1), cxcy[:, 1].view(-1, 1)
    angles = pi * cxcy[:, -1].view(-1, 1) / 180

    sin = torch.sin(angles)
    cos = torch.cos(angles)

    rotation_matrix_x = torch.cat([cos, sin, (- cos + 1) * cx - sin * cy], 1)
    rotation_matrix_y = torch.cat([-sin, cos, sin * cx + (- cos + 1) * cy], 1)

    n_priors = cxcy.shape[0]

    x1 = (rotation_matrix_x * torch.cat([top_left, torch.ones(size=(n_priors, 1))], 1)).sum(axis=1).view(-1, 1)
    y1 = (rotation_matrix_y * torch.cat([top_left, torch.ones(size=(n_priors, 1))], 1)).sum(axis=1).view(-1, 1)
    x2 = (rotation_matrix_x * torch.cat([top_right, torch.ones(size=(n_priors, 1))], 1)).sum(axis=1).view(-1, 1)
    y2 = (rotation_matrix_y * torch.cat([top_right, torch.ones(size=(n_priors, 1))], 1)).sum(axis=1).view(-1, 1)
    x3 = (rotation_matrix_x * torch.cat([bottom_right, torch.ones(size=(n_priors, 1))], 1)).sum(axis=1).view(-1, 1)
    y3 = (rotation_matrix_y * torch.cat([bottom_right, torch.ones(size=(n_priors, 1))], 1)).sum(axis=1).view(-1, 1)
    x4 = (rotation_matrix_x * torch.cat([left_bottom, torch.ones(size=(n_priors, 1))], 1)).sum(axis=1).view(-1, 1)
    y4 = (rotation_matrix_y * torch.cat([left_bottom, torch.ones(size=(n_priors, 1))], 1)).sum(axis=1).view(-1, 1)

    return torch.cat([x1, y1, x2, y2, x3, y3, x4, y4], 1)


def draw_io(image, box8points, target, path2img):
    """

    :param image:
    :param box8points:
    :param target:
    :return:
    """

    wPATH = "./result_augmentation"
    os.makedirs(wPATH,exist_ok=True)

    cx, cy, w, h, angle = target[:5].tolist()
    gt = ((cx, cy), (w, h), -angle)

    original_dims = torch.FloatTensor([300] * 4).unsqueeze(0)
    priors_cxcy = create_prior_boxes(angle)
    priors_cxcy[:, :4] = priors_cxcy[:, :4] * original_dims
    priors8points = priors_cxcy.clone()
    priors8points = cxcya_to_xy(priors8points)
    priors_cxcy = priors_cxcy[:, :4].numpy()

    index_max_iou = 0
    gt_iou = []
    points = []
    ious = []

    rect_a_area = gt[1][0] * gt[1][1]

    for i, (cx, cy, w, h) in enumerate(priors_cxcy):

        r, v = cv2.rotatedRectangleIntersection(gt, ((cx, cy), (w, h), -angle))

        if r:

            intersection_area = cv2.contourArea(v)

            rect_b_area = w * h

            iou = intersection_area / (rect_a_area + rect_b_area - intersection_area)
            ious.append(iou)

            if iou > 0.4:
                gt_iou.append(i)
                points.append(v)

    print(path2img)
    print(gt_iou)
    print(max(ious))
    print(angle)

    draw = ImageDraw.Draw(image)
    # ground true rotated rectangle
    draw.polygon(xy=box8points.tolist(), outline="red")

    for i, index in enumerate(gt_iou):
        # priors having max intersection with ground true
        draw.polygon(xy=priors8points[index].tolist(), outline="blue")
        # area intersection
        draw.polygon(xy=points[i].flatten().tolist(), outline="green", fill='green')
    # ground true center
    draw.ellipse((gt[0][0] - 2, gt[0][1] - 2, gt[0][0] + 2, gt[0][1] + 2), fill="red", outline="red")

    image.save(os.path.join(wPATH, f"{angle:.3f}_{os.path.split(path2img)[1]}"))

    return priors8points[index_max_iou]