import cv2
import torch
import numpy as np

from math import cos, pi, asin, acos, degrees
from terminaltables import AsciiTable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
#device = torch.device('cpu')

label_names = ("mrz",)
label_map = {k: v + 1 for v, k in enumerate(label_names)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
distinct_colors = ('#e6194b', '#3cb44b', '#ffe119',
                   '#0082c8', '#f58231', '#911eb4',
                   '#46f0f0', '#f032e6', '#00FF00',
                   '#ff00ff', '#0000ff', '#ff0000')
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).
    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.
    In the model, we are predicting bounding box coordinates in this encoded form.
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.
    They are decoded into center-size coordinates.
    This is the inverse of the function above.
    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_io_rotated_boxes(ground_truths, priors_cxcy):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in center-size coordinates.

    :param ground_truths: (n1,5)
    :param priors_cxcy: (n2,4)

    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    ground_truths = ground_truths.to("cpu")
    priors_cxcy = priors_cxcy.to("cpu")

    n1 = ground_truths.size(0)
    n2 = priors_cxcy.size(0)

    iou_tensor = torch.zeros((n1, n2), dtype=torch.float).to('cpu')

    for i, ground_truth in enumerate(ground_truths):

        cx, cy, w, h, angle = ground_truth.tolist()
        gt = ((int(cx), int(cy)), (int(w), int(h)), -angle)
        rect_a_area = gt[1][0] * gt[1][1]

        for j, (cx, cy, w, h) in enumerate(priors_cxcy):
            
            print(gt)
            print(((int(cx), int(cy)), (int(w), int(h)), -angle))
            ret_value, region = cv2.rotatedRectangleIntersection(gt, ((int(cx), int(cy)), (int(w), int(h)), -angle))

            if cv2.INTERSECT_NONE == ret_value:
                iou = 0
            elif cv2.INTERSECT_FULL == ret_value:
                iou = 1.0
            else:

                intersection_area = cv2.contourArea(region)
                rect_b_area = w * h
                iou = intersection_area / (rect_a_area + rect_b_area - intersection_area)

            iou_tensor[i, j] = iou

    return iou_tensor.to(device)


def find_io(set_a, set_b):
    """

    :param set_a:
    :param set_b:
    :return:
    """

    set_a = set_a.to("cpu")
    set_b = set_b.to("cpu")

    n1 = set_a.size(0)
    n2 = set_b.size(0)

    iou_tensor = torch.zeros((n1, n2), dtype=torch.float).to('cpu')

    for i, (cx_a, cy_a, w_a, h_a, sin_a, cos_a) in enumerate(set_a):

        angle_a = convert2degree(sin_a, cos_a)
        box_a = ((int(cx_a), int(cy_a)), (int(w_a), int(h_a)), -angle_a)
        a_area = w_a * h_a

        for j, (cx_b, cy_b, w_b, h_b, sin_b, cos_b) in enumerate(set_b):

            angle_b = convert2degree(sin_b, cos_b)

            ret_value, region = cv2.rotatedRectangleIntersection(box_a, ((int(cx_b), int(cy_b)), (int(w_b), int(h_b)), -angle_b))

            if cv2.INTERSECT_NONE == ret_value:
                iou = 0
            elif cv2.INTERSECT_FULL == ret_value:
                iou = 1.0
            else:
                intersection_area = cv2.contourArea(region)
                b_area = w_b * h_b
                iou = intersection_area / (a_area + b_area - intersection_area)

            iou_tensor[i, j] = iou

    return iou_tensor


def nms(boxes, scores, nms_thresh):
    """

    :param boxes:
    :param scores:
    :param nms_thresh:
    :return:
    """

    index = torch.argsort(scores, descending=True)
    sorted_boxes = boxes[index]

    keep_boxes = []

    while sorted_boxes.size(0):
        large_overlap = find_io(sorted_boxes[0].unsqueeze(0), sorted_boxes).squeeze(0) > nms_thresh
        keep_boxes += [sorted_boxes[0]]
        sorted_boxes = sorted_boxes[~large_overlap]

    return keep_boxes


def convert2degree(sin_alpha, cos_alpha):
    """

    :param sin_alpha:
    :param cos_alpha:
    :return:
    """
    sin_degree = degrees(asin(sin_alpha))
    cos_degree = degrees(acos(cos_alpha))

    degree = 0
    if sin_alpha > 0 and cos_alpha > 0:
        return (sin_degree + cos_degree) / 2
    elif sin_alpha > 0 > cos_alpha:
        return (180 - sin_degree + cos_degree) / 2
    elif sin_alpha < 0 and cos_alpha < 0:
        return (540 - sin_degree - cos_degree) / 2
    elif sin_alpha < 0 < cos_alpha:
        return (720 + sin_degree - cos_degree) / 2
    return degree


def cxcy_4points(box, width, height):
    """
    :param box : list , [cx, cy, w, h, sin_alpha, cos_alpha]
    :return: TopLeft,RightTop,RightBottom,LeftBottom,
    """

    cx, cy, w, h, sin_alpha, cos_alpha = box

    angle = convert2degree(sin_alpha, cos_alpha)

    box = [cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2]
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    rotation_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    # get coord mrz
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    # calculate new coord mrz
    new_x1, new_y1 = np.dot(rotation_mat, np.array([[x1], [y1], [1]])).flatten()
    new_x2, new_y2 = np.dot(rotation_mat, np.array([[x2], [y2], [1]])).flatten()
    new_x3, new_y3 = np.dot(rotation_mat, np.array([[x3], [y3], [1]])).flatten()
    new_x4, new_y4 = np.dot(rotation_mat, np.array([[x4], [y4], [1]])).flatten()

    box = np.array([new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4])
    dims = np.array([width, height, width, height, width, height, width, height])
    box *= dims

    return box


def adjust_learning_rate(optimizer, i_epoch, epochs, lr_base):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param i_epoch:  current epoch.
    :param epochs: number epochs.
    :param lr_base: default learning rate.
    """
    scale = 1.1 + cos(pi * i_epoch / epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_base * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def save_checkpoint(epoch, model, optimizer, folder_name, evaluation_metrics, best_f1, is_best):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    :param folder_name: a folder name for saving model
    :param evaluation_metrics: validation recall,precision,f1 per class
    :param best_f1: best validation average f1
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             }

    for metric, classes in evaluation_metrics:
        for k, v in classes.items():
            if k not in state:
                state[k] = []
            state[k].append((metric, v))

    print("Saving model...")

    filename = f'{epoch}.pth.tar'
    torch.save(state, f"./{folder_name}/{filename}")
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        state['best_f1'] = best_f1
        torch.save(state, f'./BEST_model.pth.tar')


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def show_metrics(precision, recall, f1, accuracy_angle, epoch=None, mode="val", logger=None, print_ascii_table=True):
    """

    :param precision:
    :param recall:
    :param f1:
    :param  accuracy_angle:
    :param epoch:
    :param mode:
    :param logger:
    :param print_ascii_table:

    """

    if logger is not None:
        evaluation_metrics = [
            (f"{mode}/precision", {label_name: precision[i] for i, label_name in enumerate(label_names)}),
            (f"{mode}/recall", {label_name: recall[i] for i, label_name in enumerate(label_names)}),
            (f"{mode}/f1", {label_name: f1[i] for i, label_name in enumerate(label_names)}),
            (f"{mode}/accuracy_angle", {label_name: accuracy_angle[i] for i, label_name in enumerate(label_names)}),

        ]

        logger.list_of_scalars_summary(evaluation_metrics, epoch, average=True)

    if print_ascii_table:
        print("\nValidation" if mode == "val" else "\nTest")
        if epoch is not None:
            print(f'Epoch number:{epoch}')
        ap_table = [["Class name", "Precision", "Recall", "f1","Accuracy angle"]]
        for i, label_name in enumerate(label_names):
            ap_table += [[label_name, f"{precision[i]:.5f}", f"{recall[i]:.5f}", f"{f1[i]:.5f}",f"{accuracy_angle[i]:.5f}"]]
        print(AsciiTable(ap_table).table)
        del ap_table

    print(f"Average: F1 = {f1.mean():.3f},Recall = {recall.mean():.3f},"
          f"Precision ={precision.mean():.3f},Accuracy angle = {accuracy_angle.mean():.3f}")


def show_confusion_matrix(confusion_matrix, all_labels):
    """

    :param confusion_matrix:
    :param all_labels:
    :return:precision, recall, f1

    """

    precision = torch.zeros(len(label_names))
    recall = torch.zeros(len(label_names))
    accuracy_angle = torch.zeros(len(label_names))

    ap_table = [["Prd\GT", *label_names, "Bg", "TP angle", "TP", "FP", "Total GT" ]]

    for i, label_name in enumerate(label_names):
        n_gt = int((all_labels == (i + 1)).sum())
        n_prd = int(confusion_matrix[i,:-1].sum())
        n_tp = int(confusion_matrix[i, i])
        n_tp_angle = int(confusion_matrix[i, -1])

        precision[i] = n_tp / n_prd if n_prd != 0 else 0
        recall[i] = n_tp / n_gt
        accuracy_angle[i] = n_tp_angle / n_tp if n_tp != 0 else 0

        ap_table += [[label_name, *confusion_matrix[i].tolist(), n_tp, n_prd - n_tp, n_gt]]

    print(AsciiTable(ap_table).table)

    f1 = 2 * precision * recall / (precision + recall + 1e-16)

    return precision, recall, f1, accuracy_angle
