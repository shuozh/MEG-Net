import torch
import time
import os
import numpy as np
from math import log10
import utils
from initializers import weights_init_xavier
from model import HVLF
from func_input_npy import image_input_full
from SRLF import SRLF_Dataset_new

MAX_EPOCH = 1701


def main(params):
    print(params)
    view_n = params['view_n'] if 'view_n' in params else 3
    scale = params['scale'] if 'scale' in params else 2
    n_seb = params['n_seb'] if 'n_seb' in params else 4
    n_sab = params['n_sab'] if 'n_sab' in params else 4
    n_feats = params['n_feats'] if 'n_feats' in params else 16
    dir_LF_training = params['dir_LF_training'] if 'dir_LF_training' in params else None
    dir_LF_testing = params['dir_LF_testing'] if 'dir_LF_testing' in params else None
    dir_model = params['dir_model'] if 'dir_model' in params else None
    base_lr = params['base_lr'] if 'base_lr' in params else 0.001
    batch_size = params['batch_size'] if 'batch_size' in params else 32
    repeat_size = params['repeat_size'] if 'repeat_size' in params else 32
    crop_size = params['crop_size'] if 'crop_size' in params else 24
    gpu_no = params['gpu_no'] if 'gpu_no' in params else 1
    current_iter = params['current_iter'] if 'current_iter' in params else 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)

    ''' Define Model(set parameters)'''
    start_time = time.strftime('%m%d%H', time.localtime(time.time()))
    criterion = torch.nn.MSELoss()
    criterion_train = torch.nn.L1Loss()

    model = HVLF(scale=scale, n_seb=n_seb, n_sab=n_sab, n_feats=n_feats)

    model.apply(weights_init_xavier)
    utils.get_parameter_number(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)

    ''' Loading the trained model'''
    if dir_model is not None:
        ''' Loading the trained model'''
        if not os.path.exists(f'../net_store/{dir_model}/'):
            os.makedirs(f'../net_store/{dir_model}/')
        state_dict = torch.load(f'../net_store/{dir_model}/HVLF_{scale}_{view_n}.pkl.pkl')
        model.load_state_dict(state_dict)

    ''' Create the save path for training models and record the training and test loss'''

    dir_save_name = '../net_store/scale{}view{}'.format(scale, view_n)

    dir_save_name += f'_{start_time}'

    print(dir_save_name)

    if not os.path.exists(dir_save_name):
        os.makedirs(dir_save_name)

    best_psnr = 0
    test_psnr_list = []
    train_loss_list = []

    train_dataset = SRLF_Dataset_new(dir_LF_training, repeat_size=repeat_size, view_n=view_n, scale=scale,
                                     crop_size=crop_size, if_flip=True, if_rotation=True, if_test=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(current_iter, MAX_EPOCH):

        # torch.cuda.empty_cache()
        ''' Validation during the training process'''
        if epoch % 200 == 0:
            torch.save(model.state_dict(), dir_save_name + '/HVLF_' + str(epoch) + '.pkl')
            best_psnr_new, test_psnr, test_loss = test_res(dir_LF_testing, model, criterion, view_n, scale,
                                                           best_psnr)

            if test_psnr > best_psnr:
                torch.save(model.state_dict(),
                           dir_save_name + '/HVLF_{}_{}.pkl'.format(scale, str(view_n)))

            best_psnr = best_psnr_new
            test_psnr_list.append(test_psnr)

        ''' Training begin'''
        current_iter, train_loss = train_res(train_loader, model, epoch, criterion_train, optimizer, scheduler,
                                             current_iter)

        train_loss_list.append(train_loss)

        np.save(dir_save_name + '/psnr.npy', test_psnr_list)
        np.save(dir_save_name + '/loss.npy', train_loss_list)


def train_res(train_loader, model, epoch, criterion, optimizer, scheduler, current_iter):
    lr = scheduler.get_lr()[0]
    model.train()
    torch.backends.cudnn.benchmark = True  # speed up

    time_start = time.time()
    total_loss = 0
    count = 0

    for i, (train_data, gt_data) in enumerate(train_loader):
        train_data, gt_data = train_data.cuda(), gt_data.cuda()

        # Forward pass: Compute predicted y by passing x to the model

        gt_pred = model(train_data)

        # Compute and print loss
        loss = criterion(gt_pred, gt_data[:, :, :, :, :])
        total_loss += loss.item()
        count += 1

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    current_iter += 1
    scheduler.step()

    time_end = time.time()
    print('=========================================================================')
    print('Train Epoch: {} Learning rate: {:.2e} Time: {:.2f}s Average Loss: {:.6f} '
          .format(epoch, lr, time_end - time_start, total_loss / count))
    return current_iter, total_loss / count


def test_res(dir_Test_LFimages, model, criterion, view_n, scale, best_psnr):
    avg_psnr = 0
    avg_loss = 0
    image_num = 10
    time_start = time.time()
    for root, dirs, files in os.walk(dir_Test_LFimages):
        model.eval()
        if len(files) == 0:
            break

        for i in range(image_num):
            image_path = 'general_' + str(i + 1) + '.npy'
            # for image_path in files:

            torch.cuda.empty_cache()

            train_data, gt_data = image_input_full(root, image_path, view_n, scale)
            train_data, gt_data = input_prepare(train_data, gt_data)

            with torch.no_grad():
                # Forward pass: Compute predicted y by passing x to the model
                # chop for less memory consumption during test

                gt_pred = overlap_crop_forward(train_data, scale, model, shave=10)
                loss = criterion(gt_pred[:, :, :, 15:-15, 15:-15], gt_data[:, :, :, 15:-15, 15:-15])

                avg_loss += loss.item()
                psnr = 10 * log10(1 / loss.item())
                avg_psnr += psnr

                print('Test Loss: {:.6f}, PSNR: {:.4f} in {}'.format(loss.item(), psnr, image_path))

        break

    if (avg_psnr / image_num) > best_psnr:
        best_psnr = avg_psnr / image_num

    print('===> Avg. PSNR: {:.4f} dB / BEST {:.4f} dB Avg. Loss: {:.6f}. Time: {:.6f}'
          .format(avg_psnr / image_num, best_psnr, avg_loss / image_num, time.time() - time_start))

    return best_psnr, avg_psnr / image_num, avg_loss / image_num


def input_prepare(train_data, gt_data):
    train_data = train_data[np.newaxis, :, :, :, :]
    gt_data = gt_data[np.newaxis, :, :, :, :]
    train_data, gt_data = torch.from_numpy(train_data.copy()), torch.from_numpy(gt_data.copy())
    train_data, gt_data = train_data.cuda(), gt_data.cuda()
    return train_data, gt_data


def overlap_crop_forward(x, scale, model, shave=10):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 1
    b, u, v, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, :, 0:h_size, 0:w_size],
        x[:, :, :, 0:h_size, (w - w_size):w],
        x[:, :, :, (h - h_size):h, 0:w_size],
        x[:, :, :, (h - h_size):h, (w - w_size):w]]

    sr_list = []
    for i in range(0, 4, n_GPUs):
        lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
        sr_batch_temp = model(lr_batch)

        if isinstance(sr_batch_temp, list):
            sr_batch = sr_batch_temp[-1]
        else:
            sr_batch = sr_batch_temp

        sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, u, v, h, w)
    output[:, :, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, :, 0:h_half, 0:w_half]
    output[:, :, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


if __name__ == '__main__':
    params = {
        'view_n': 7,
        'scale': 2,
        'dir_LF_training': '../../Dataset/180LF/',  # .npy
        "dir_LF_testing": '../../Dataset/test_general/',  # .npy
        'gpu_no': 3,
    }
    main(params)
