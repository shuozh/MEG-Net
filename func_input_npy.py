import cv2
import numpy as np


def image_input_full(root, image_path, view_n, scale):
    gt_image = np.float32(np.load(root + image_path))
    if np.max(gt_image) > 1.0:
        gt_image /= 255.0
    if view_n < 9:
        gt_image_input = angular_resolution_changes(gt_image, 9, view_n)
    else:
        gt_image_input = gt_image

    image_h = int(gt_image_input.shape[2] / scale)
    image_w = int(gt_image_input.shape[3] / scale)
    gt_image_input = gt_image_input[:, :, 0:image_h * scale, 0:image_w * scale]

    lr_image_input = np.zeros((view_n, view_n, int(image_h), int(image_w)), dtype=np.float32)

    for i in range(0, view_n, 1):
        for j in range(0, view_n, 1):
            img = cv2.blur(gt_image_input[i, j, :, :], (scale, scale))
            img_tmp = img[scale // 2::scale, scale // 2::scale]

            lr_image_input[i, j, :, :] = img_tmp

    return lr_image_input, gt_image_input


def angular_resolution_changes(image, view_num_ori, view_num_new):
    n_view = (view_num_ori + 1 - view_num_new) // 2
    return image[n_view:n_view + view_num_new, n_view:n_view + view_num_new, :, :]
