import torch

from utils.utils import find_io, convert2degree, label_names


def get_confusion_matrix(predict_boxes, predict_labels, boxes, labels, angles, iou_threshold=0.5, angle_threshold=5.):
    """
    Create confusion matrix

    :param predict_boxes: list of tensors, one tensor for each image containing prediction objects' bounding boxes
    :param predict_labels: list of tensors, one tensor for each image containing prediction objects' labels' scores
    :param boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param labels: list of tensors,one tensor for each image containing actual objects' labels
    :param angles :list of tensors,one tensor for each image containing actual objects' angle
    :param iou_threshold: threshold for intersection over union metrics between predict_boxes and boxes
    :param  angle_threshold:

    :return: confusion_matrix, a tensor of size (n_class,n_class+bg+angle)

    """

    assert len(predict_boxes) == len(predict_labels)

    batch_size = len(predict_boxes)
    n = len(label_names)
    confusion_matrix = torch.zeros(size=(n, n + 2), dtype=torch.int)
    diff_angle = []

    for sample_i in range(batch_size):

        prd_boxes = predict_boxes[sample_i]
        prd_labels = predict_labels[sample_i]

        gt_boxes = boxes[sample_i]
        gt_labels = labels[sample_i]
        gt_angles = angles[sample_i]

        detected_boxes = []

        for prd_box, prd_label in zip(prd_boxes, prd_labels):

            iou, box_index = find_io(prd_box.unsqueeze(0), gt_boxes).squeeze(0).max(0)

            if iou >= iou_threshold and box_index not in detected_boxes and int(prd_label) != 0:
                confusion_matrix[int(prd_label) - 1, int(gt_labels[box_index]) - 1] += 1
                if int(prd_label) == int(gt_labels[box_index]):
                    detected_boxes.append(box_index)
                    prd_sin_a, prd_cos_a = prd_box[..., -2], prd_box[..., -1]
                    prd_angle = convert2degree(prd_sin_a, prd_cos_a)
                    gt_angle = gt_angles[box_index]
                    arc_a = (abs(prd_angle - gt_angle)).int().item()
                    arc_b = 360 - arc_a
                    arc = min(arc_a, arc_b)
                    diff_angle.append(arc)
                    if arc < angle_threshold:
                        confusion_matrix[int(prd_label) - 1, -1] += 1
            elif int(prd_label) != 0:
                confusion_matrix[int(prd_label) - 1, -2] += 1

        del detected_boxes

    return confusion_matrix,diff_angle
