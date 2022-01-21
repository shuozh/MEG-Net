import numpy as np
import torch.optim.lr_scheduler as lrs
import cv2
import skimage.color as color
import imageio


def make_scheduler(my_optimizer):
    # learning rate decay per N epochs
    lr_decay = 200
    # learning rate decay factor for step decay
    gamma = 0.5
    scheduler = lrs.StepLR(my_optimizer, step_size=lr_decay, gamma=gamma)
    return scheduler


def get_135_position(view_n):
    start_position_list = []
    for i in range(view_n):
        start_position_list.append(([i], [0]))
    for j in range(1, view_n):
        start_position_list.append(([0], [j]))
    for item in start_position_list:
        while item[0][-1] < view_n - 1 and item[1][-1] < view_n - 1:
            item[0].append(item[0][-1] + 1)
            item[1].append(item[1][-1] + 1)
    return start_position_list


def get_45_position(view_n):
    start_position_list = []
    for i in range(view_n):
        start_position_list.append(([i], [0]))
    for j in range(1, view_n):
        start_position_list.append(([view_n - 1], [j]))
    for item in start_position_list:
        while item[0][0] > 0 and item[1][0] < view_n - 1:
            item[0].insert(0, item[0][0] - 1)
            item[1].insert(0, item[1][0] + 1)
    return start_position_list


def image_input(image_path, scale, view_ori, view_n):
    """
    prepare ground truth and test data
    :param image_path: loading 4D training LF from this path
    :param scale: spatial upsampling scale
    :param view_ori: original angular resolution
    :param view_n: crop Length from Initial LF for test
    :return: ground truth and test data with YCbCr
    """

    gt_image = imageio.imread(image_path)
    gt_image_ycbcr = color.rgb2ycbcr(gt_image[:, :, :3])

    # change the angular resolution of LF images for different input
    num_vew_gap = (view_ori + 1 - view_n) // 2

    image_h = gt_image_ycbcr.shape[0] // view_ori
    image_w = gt_image_ycbcr.shape[1] // view_ori
    channel_n = gt_image_ycbcr.shape[2]

    # cut the extra pixels
    if image_h % scale != 0:
        gt_image_ycbcr = gt_image_ycbcr[:-(image_h % scale) * view_ori, :, :]
        image_h -= image_h % scale
    if image_w % scale != 0:
        gt_image_ycbcr = gt_image_ycbcr[:, :-(image_w % scale) * view_ori, :]
        image_w -= image_w % scale

    # downsampling with interpolation
    gt_ycbcr = np.zeros((1, view_n, view_n, image_h, image_w, channel_n), dtype=np.float32)
    lr_ycbcr = np.zeros((1, view_n, view_n, image_h // scale, image_w // scale, channel_n), dtype=np.float32)

    for i in range(0, view_n, 1):
        for j in range(0, view_n, 1):
            gt_ycbcr[0, i, j, :, :, :] = gt_image_ycbcr[i + num_vew_gap::view_ori, j + num_vew_gap::view_ori, :]
            # interpolation with blur
            gt_ycbcr_blur = cv2.blur(gt_image_ycbcr[i + num_vew_gap::view_ori, j + num_vew_gap::view_ori, :],
                                     (scale, scale))
            lr_ycbcr[0, i, j, :, :, :] = gt_ycbcr_blur[scale // 2::scale, scale // 2::scale, :]

    return gt_ycbcr[:, :, :, :, :, 0], lr_ycbcr


def get_parameter_number(net):
    print(net)
    parameter_list = [p.numel() for p in net.parameters()]
    print(parameter_list)
    total_num = sum(parameter_list)
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
