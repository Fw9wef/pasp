import torch
import numpy as np
from numpy import degrees
from math import asin, acos

from tqdm import tqdm

from utils.utils import device
from utils.datasets import ListDataset

# Parameters
best_checkpoint = './checkpoints/91.pth'
data_folder = f'./annotation'
batch_size = 32
workers = 32
add_txt_files_name = 'super_'


@np.vectorize
def get_angle(sin_alpha, cos_alpha):
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


def min_arc(deg_a, deg_b):
    arc_a = np.abs(deg_a - deg_b)
    arc_b = 360 - arc_a
    return np.min(np.stack([arc_a, arc_b], axis=-1), axis=-1)


def evaluate(model, test_loader):
    n = 0
    sae_sin = 0
    sae_cos = 0
    sae_angle = 0
    sae_mse = 0

    model.eval()

    with torch.no_grad():
        # Batches
        for i, (images, _, target) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)
            gt_angle = target[:, 4].view(-1).detach().to('cpu').numpy()
            gt_sin = target[:, 5].view(-1).detach().to('cpu').numpy()
            gt_cos = target[:, 6].view(-1).detach().to('cpu').numpy()

            gt_xywh = target[:, :4].detach().to('cpu').numpy()
            #gt_sc = target[:, -3:-1].detach().to('cpu').numpy()
            #gt = np.concatenate([gt_xywh, gt_sc], axis=-1)

            # Forward prop.
            pred = model(images)
            pred_sin = pred[:, -2].view(-1).detach().to('cpu').numpy()
            pred_cos = pred[:, -1].view(-1).detach().to('cpu').numpy()
            pred_angle = get_angle(pred_sin, pred_cos)

            pred = pred.detach().to('cpu').numpy()
            mse = np.sum(np.sqrt(np.sum((gt_xywh-pred[:, :4])**2, axis=-1)))

            n += int(gt_sin.shape[0])
            sae_sin += np.sum(np.abs(gt_sin - pred_sin))
            sae_cos += np.sum(np.abs(gt_cos - pred_cos))
            sae_angle += np.sum(min_arc(gt_angle, pred_angle))
            sae_mse += mse

    mae_sin = sae_sin / n
    mae_cos = sae_cos / n
    mae_angle = sae_angle / n
    mae_mse = sae_mse / n
    print("MEAN ABSOLUTE SIN ERROR : %f" % mae_sin)
    print("MEAN ABSOLUTE COS ERROR : %f" % mae_cos)
    print("MEAN ABSOLUTE ANGLE ERROR : %f" % mae_angle)
    print("MSE : %f" % mae_mse)
    f = open(add_txt_files_name + 'mae_sin.txt', 'a')
    f.write(str(mae_sin) + '\n')
    f.close()
    f = open(add_txt_files_name + 'mae_cos.txt', 'a')
    f.write(str(mae_cos) + '\n')
    f.close()
    f = open(add_txt_files_name + 'mae_angle.txt', 'a')
    f.write(str(mae_angle) + '\n')
    f.close()
    return mae_mse


def main():
    # Load model checkpoint that is to be evaluated
    model = MRZdetector()
    checkpoint = torch.load(best_checkpoint)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # Switch to eval mode
    model.eval()

    # Load test data
    test_dataset = ListDataset(data_folder, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=workers, pin_memory=True)

    # evaluating model
    e_loss = evaluate(model, test_loader)
    print("Test loss: ", e_loss)

if __name__ == '__main__':
    from pasp_model import MRZdetector

    main()
