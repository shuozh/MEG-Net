import torch
import os
import time
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import skimage.color as color

import utils
from model import HVLF


def test_main(image_path, model_path, view_n_ori, view_n, scale, cut_margin=15, gpu_no=0, is_fine_tune=False,
              save_img=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)
    torch.backends.cudnn.benchmark = True

    print('=' * 40)
    print('build network...')
    model = HVLF(scale=scale)
    utils.get_parameter_number(model)
    # return
    model.cuda()
    model.eval()
    print('done')

    print('=' * 40)
    print('load model...')
    if is_fine_tune:
        state_dict = torch.load(model_path + 'HVLF_' + str(scale) + '_' + str(view_n) + '_ft.pkl')
    else:
        state_dict = torch.load(model_path + 'HVLF_' + str(scale) + '_' + str(view_n) + '.pkl')
    model.load_state_dict(state_dict)
    print('done')

    print('=' * 40)
    print('create save directory...')
    if is_fine_tune:
        save_path = '../result/scale{}view{}_fine_tune_result/'.format(scale, view_n)
    else:
        save_path = '../result/scale{}view{}_result/'.format(scale, view_n)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('path: ', save_path)
    print('done')

    print('=' * 40)
    print('predict image...')

    xls_list = []
    psnr_list = []
    ssim_list = []
    time_list = []

    image_list = os.listdir(image_path)
    image_list.sort()

    for index, image_name in enumerate(image_list):
        print('-' * 100)
        print('[{}/{}]'.format(index + 1, len(image_list)), image_name)
        gt_hr_y, lr_ycbcr = utils.image_input(image_path + image_name, scale, view_n_ori, view_n)
        lr_y, gt_hr_y, lr_cbcr = lr_ycbcr[:, :, :, :, :, 0] / 255.0, gt_hr_y / 255.0, lr_ycbcr[:, :, :, :, :, 1:3]

        hr_y, time_ = predict_y(lr_y, model)

        time_list.append(time_)

        hr_y = hr_y.cpu().numpy()
        hr_y = np.clip(hr_y, 16.0 / 255.0, 235.0 / 255.0)

        gt_hr_y = np.clip(gt_hr_y, 16.0 / 255.0, 235.0 / 255.0)

        psnr_view_list = []
        ssim_view_list = []

        for i in range(view_n):
            for j in range(view_n):
                if cut_margin == 0:
                    psnr_view = peak_signal_noise_ratio(hr_y[0, i, j, :, :],
                                                        gt_hr_y[0, i, j, :, :], data_range=1)
                    psnr_view_list.append(psnr_view)
                    ssim_view = structural_similarity(hr_y[0, i, j, :, :],
                                                      gt_hr_y[0, i, j, :, :], data_range=1)
                    ssim_view_list.append(ssim_view)
                else:
                    psnr_view = peak_signal_noise_ratio(hr_y[0, i, j, cut_margin:-cut_margin, cut_margin:-cut_margin],
                                                        gt_hr_y[0, i, j, cut_margin:-cut_margin,
                                                        cut_margin:-cut_margin], data_range=1)
                    psnr_view_list.append(psnr_view)
                    ssim_view = structural_similarity(hr_y[0, i, j, cut_margin:-cut_margin, cut_margin:-cut_margin],
                                                      gt_hr_y[0, i, j, cut_margin:-cut_margin, cut_margin:-cut_margin],
                                                      data_range=1)
                    ssim_view_list.append(ssim_view)
                print('{:6.4f}/{:6.4f}'.format(psnr_view, ssim_view), end='\t\t')
            print('')

        if save_img:
            result_image_path = save_path + image_name[0:-4] + '/'
            if not os.path.exists(result_image_path):
                os.makedirs(result_image_path)

            hr_cbcr = predict_cbcr(lr_cbcr, scale, view_n)
            for i in range(view_n):
                for j in range(view_n):
                    hr_y_item = np.clip(hr_y[0, i, j, :, :] * 255.0, 16.0, 235.0)
                    hr_y_item = hr_y_item[:, :, np.newaxis]
                    hr_cb_item = hr_cbcr[0, i, j, :, :, 0:1]
                    hr_cr_item = hr_cbcr[0, i, j, :, :, 1:2]
                    hr_ycbcr_item = np.concatenate((hr_y_item, hr_cb_item, hr_cr_item), 2)
                    hr_rgb_item = color.ycbcr2rgb(hr_ycbcr_item) * 255.0
                    hr_rgb_item = hr_rgb_item[:, :, ::-1]
                    img_save_path = result_image_path + str(i) + str(j) + '.png'
                    cv2.imwrite(img_save_path, hr_rgb_item)
        psnr_ = np.mean(psnr_view_list)
        psnr_list.append(psnr_)
        ssim_ = np.mean(ssim_view_list)
        ssim_list.append(ssim_)

        print('PSNR: {:.4f} SSIM: {:.4f} time: {:.4f}'.format(psnr_, ssim_, time_))
        xls_list.append([image_name, psnr_, ssim_, time_])

    xls_list.append(['average', np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)])
    xls_list = np.array(xls_list)

    result = pd.DataFrame(xls_list, columns=['image', 'psnr', 'ssim', 'time'])
    result.to_csv(save_path + 'result_s{}n{}_{}.csv'.format(scale, view_n, int(time.time())))

    print('-' * 100)
    print('Average: PSNR: {:.4f}, SSIM: {:.4f}, TIME: {:.4f}'.format(np.mean(psnr_list), np.mean(ssim_list),
                                                                     np.mean(time_list)))
    print('all done')


def predict_y(lr_y, model):
    with torch.no_grad():
        lr_y = torch.from_numpy(lr_y.copy())
        lr_y = lr_y.cuda()
        time_item_start = time.time()
        hr_y = model(lr_y)
        return hr_y, time.time() - time_item_start


def predict_cbcr(lr_cbcr, scale, view_n):
    hr_cbcr = np.zeros((1, view_n, view_n, lr_cbcr.shape[3] * scale, lr_cbcr.shape[4] * scale, 2))

    for i in range(view_n):
        for j in range(view_n):
            image_bicubic = cv2.resize(lr_cbcr[0, i, j, :, :, :],
                                       (lr_cbcr.shape[4] * scale, lr_cbcr.shape[3] * scale),
                                       interpolation=cv2.INTER_CUBIC)
            hr_cbcr[0, i, j, :, :, :] = image_bicubic
    return hr_cbcr


if __name__ == '__main__':
    test_main(image_path='../../Dataset/test_hci1/',  # .png
              model_path='../models/',
              view_n_ori=9,
              view_n=3,
              scale=2,
              gpu_no=3,
              is_fine_tune=True,
              save_img=False)
