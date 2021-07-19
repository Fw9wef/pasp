import os
import torch
import pandas as pd
import numpy as np

from torchvision import transforms
from PIL import Image
import cv2

from mrz_model import MRZdetector
from pasp_model import MRZdetector as PaspDetector
from utils.utils import label_color_map, rev_label_map, device, convert2degree

wr_path = './a4_final_test_output'
os.makedirs(wr_path, exist_ok=True)
data_folder = './annotation'

def load_models(mrz_checkpoint = './mrz_lib/mrz_model.pth', pasp_checkpoint = './mrz_lib/pasp_model.pth'):
    pasp_checkpoint = torch.load(pasp_checkpoint)
    pasp_model = PaspDetector()
    pasp_model.load_state_dict(pasp_checkpoint)
    pasp_model = pasp_model.to(device)
    pasp_model.eval()

    mrz_checkpoint = torch.load(mrz_checkpoint)
    mrz_model = MRZdetector(n_classes=len(label_color_map))
    mrz_model.load_state_dict(mrz_checkpoint)
    mrz_model = mrz_model.to(device)
    mrz_model.eval()
    return pasp_model, mrz_model

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def get_projection(img):
    hist = [np.sum(line) for line in img]
    total = np.sum(hist)
    hist = [x/total for x in hist]
    return hist


def calc_energy(img):
    hist = get_projection(img)
    energy = np.sum([x*x for x in hist])
    return energy


def value_binarization(img, threshold=120):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    value_map = img[:,:,2]
    otsu_threshold, image_result = cv2.threshold(value_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - image_result


def rotate(img, angle):
    h, w = img.shape[:2]
    center = w / 2, h / 2
    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1)
    img = cv2.warpAffine(img, rotation_mat, (w,h))#, borderValue=[255,255,255])
    return img


def resize_crop(img):
    img_h, img_w = img.shape[:2]
    new_h = 50
    new_w = int(new_h*img_w/img_h)
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_CUBIC)
    return img


def get_best_angle_in_range(img, max_angle, delta, const_angle, base_enrgy):
    best_enrg, delta_angle = base_enrgy, 0
    
    for angle in np.arange(delta, max_angle, delta):
        rotated_img_1 = rotate(img, angle+const_angle)
        rotated_img_2 = rotate(img, -angle+const_angle)
        
        enrg_1, enrg_2 = calc_energy(rotated_img_1), calc_energy(rotated_img_2)
        for i, enrg in enumerate([enrg_1, enrg_2]):
            if enrg > best_enrg:
                best_enrg = enrg
                delta_angle = ((-1)**i) * angle
    
    return delta_angle


def get_delta_angle(img):
    img = resize_crop(img)
    img = value_binarization(img).astype(np.uint8)
    
    base_enrgy = calc_energy(img)
    
    delta_angle = 0
    for max_angle, delta in zip([10,2],[2,0.25]):
        delta_angle += get_best_angle_in_range(img, max_angle, delta, delta_angle, base_enrgy)
    
    return delta_angle


def resize_angle(angle, orig_w, orig_h, curr_w = 300, curr_h = 300):
    if angle > 270:
        const_angle = 270
        angle -= 270
        tg_alpha = np.tan(np.deg2rad(angle))
        sq_h, sq_w = 1, tg_alpha
        
        k, m = orig_w/curr_w, orig_h/curr_h
        
        angle = np.rad2deg(np.arctan((sq_w*k)/(sq_h*m)))
    
    
    elif angle>180:
        const_angle = 180
        angle -= 180
        tg_alpha = np.tan(np.deg2rad(angle))
        sq_h, sq_w = tg_alpha, 1
        
        k, m = orig_w/curr_w, orig_h/curr_h
        
        angle = np.rad2deg(np.arctan((sq_h*m)/(sq_w*k)))
    
    
    elif angle>90:
        const_angle = 90
        angle -= 90
        tg_alpha = np.tan(np.deg2rad(angle))
        sq_h, sq_w = 1, tg_alpha
    
        k, m = orig_w/curr_w, orig_h/curr_h
    
        angle = np.rad2deg(np.arctan((sq_w*k)/(sq_h*m)))
    
    
    else:
        const_angle = 0
        tg_alpha = np.tan(np.deg2rad(angle))
        sq_h, sq_w = tg_alpha, 1
        
        k, m = orig_w/curr_w, orig_h/curr_h
        
        angle = np.rad2deg(np.arctan((sq_h*m)/(sq_w*k)))
    
    return const_angle+angle


def detect(original_image, pasp_model, mrz_model, min_score=0.2, max_overlap=0, top_k=1):
    orig_w, orig_h = original_image.width, original_image.height
    image = normalize(to_tensor(resize(original_image)))
    image = image.to(device)
    
    with torch.no_grad():
        predicted_locs, predicted_scores = mrz_model(image.unsqueeze(0))
        det_boxes, det_labels = mrz_model.detect_objects(predicted_locs,
                                                       predicted_scores,
                                                       min_score=min_score,
                                                       max_overlap=max_overlap,
                                                       top_k=top_k)
        pasp_locs = pasp_model(image.unsqueeze(0))
    det_boxes = det_boxes[0].to('cpu')
    pasp_locs = pasp_locs.to('cpu')*300
    pasp_locs = pasp_locs.tolist()
    
    
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    if det_labels == ['background']:
        return None
    
    cx, cy, w, h, sin_alpha, cos_alpha = det_boxes[0].tolist()
    angle = convert2degree(sin_alpha, cos_alpha)
    angle = resize_angle(angle, orig_w, orig_h)
    angle = -angle
    
    box = [cx, cy, w, h, orig_w, orig_h]
    pasp_box = pasp_locs[:4] + [orig_w, orig_h]
    
    mrz = get_mrz(original_image, angle, box)
    
    delta_angle = get_delta_angle(mrz)
    angle = angle+delta_angle
    
    mrz = get_mrz(original_image, angle, pasp_box, pasp_mode=True)
    return mrz


def get_mrz(original_image, angle, box, pasp_mode=False):
    cx, cy, w_box, h_box, orig_w, orig_h = box
    ratio_w, ratio_h = orig_w/300, orig_h/300
    
    cx, cy = int(cx*ratio_w), int(cy*ratio_h)
    center = (cx, cy)
    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1)
    abs_cos = np.abs(rotation_mat[0, 0])
    abs_sin = np.abs(rotation_mat[0, 1])
    w, h = int((orig_h * abs_sin) + (abs_cos * orig_w)), int((orig_h * abs_cos) + (abs_sin * orig_w))
    rotation_mat[0, 2] += (w / 2) - center[0]
    rotation_mat[1, 2] += (h / 2) - center[1]
    rotated_img = cv2.warpAffine(np.array(original_image), rotation_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    cx, cy = int(w/2), int(h/2)
    
    rot_h, rot_w = rotated_img.shape[:2]
    #half_w_box, half_h_box = int(w_box*rot_w/300/2), int(h_box*rot_h/300/2)
    #half_w_box, half_h_box = int(w_box*orig_w/300/2), int(h_box*orig_h/300/2)
    if angle > 270:
        res_angle = angle - 270
    elif angle > 180:
        res_angle = angle - 180
    elif angle > 90:
        res_angle = angle - 90
    else:
        res_angle = angle
    
    w_box = w_box*np.sqrt((np.sin(np.deg2rad(res_angle))*ratio_h)**2 + (np.cos(np.deg2rad(res_angle))*ratio_w)**2)
    if w_box>=orig_w:
        w_box = orig_w-1
    h_box = h_box*np.sqrt((np.cos(np.deg2rad(res_angle))*ratio_h)**2 + (np.sin(np.deg2rad(res_angle))*ratio_w)**2)
    if h_box>=orig_h:
        h_box = orig_h-1
    half_w_box, half_h_box = int(w_box/2), int(h_box/2)
    
    #pasp_h = int(w_box / pasp_ratio - half_h_box)
    
    #if pasp_mode:
    #    mrz = rotated_img[cy-pasp_h:cy+half_h_box, cx-half_w_box:cx+half_w_box]
    #else:
    mrz = rotated_img[cy-half_h_box:cy+half_h_box, cx-half_w_box:cx+half_w_box]
    
    #import matplotlib.pyplot as plt
    #print(half_w_box, half_h_box)
    #fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(18,18))
    #ax1.imshow(original_image)
    #ax2.imshow(rotated_img)
    #ax3.imshow(mrz)
    
    return mrz


def main():
    
    d_model, r_model = load_models()
    
    data = pd.read_csv(os.path.join(data_folder, "a4_final_TEST_annotation.csv"))
    df = pd.DataFrame(data)
    images = df['ПУТЬ К ФАЙЛУ']
    
    print(f"Number of images - {len(images)}")
    
    for i, path2image in enumerate(images):
        original_image = Image.open(os.path.join('/fulldisk',path2image), mode='r')
        #original_image, _, _ = transform(os.path.join('/fulldisk',path2image),
        #                                 [int(x) for x in df.loc[i,"КООРДИНАТЫ МРЗ"][1:-1].split(', ')],
        #                                 float(df.loc[i,"УГОЛ НАКЛОНА"]),
        #                                 float(df.loc[i,"ОРИЕНТАЦИЯ"]),
        #                                 'TEST')
        #original_image = cv2.imread(os.path.join('/fulldisk',path2image))
        original_image = original_image.convert('RGB')
        #original_image = cv2.resize(original_image, (300,300), cv2.INTER_CUBIC)
        #original_image = np.array(original_image)[:,:,::-1]
        #original_image = Image.fromarray(original_image)

        annotated_image = detect(original_image, d_model, r_model, min_score=0.2, max_overlap=0, top_k=1)

        if annotated_image is not None:
            annotated_image.save(os.path.join(wr_path, os.path.split(path2image)[1]))

        if (i + 1) % 10 == 0:
            print(f"Image {i + 1} done!")

    print(f"FPS = {len(times) / sum(times)}")


if __name__ == '__main__':
    main()
