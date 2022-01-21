import common
import torch
import torch.nn as nn
import utils


class HVLF(nn.Module):
    def __init__(self, scale=2, n_seb=4, n_sab=4, n_feats=16, is_large_kernel=False):
        super(HVLF, self).__init__()

        # 4 resblock in each image stack
        # n_seb = 4 #5
        if is_large_kernel:
            kernel_size = (3, 5, 5)
            padding = (1, 2, 2)
        else:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)

        # define body module
        m_horizontal_first = [
            nn.Conv3d(1, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=True)]
        m_horizontal = [
            common.ResBlockc3d(n_feats, is_large_kernel=is_large_kernel) for _ in range(n_seb)
        ]

        m_vertical_first = [nn.Conv3d(1, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=True)]
        m_vertical = [
            common.ResBlockc3d(n_feats, is_large_kernel=is_large_kernel) for _ in range(n_seb)
        ]

        m_45_first = [nn.Conv3d(1, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=True)]
        m_45 = [
            common.ResBlockc3d(n_feats, is_large_kernel=is_large_kernel) for _ in range(n_seb)
        ]

        m_135_first = [nn.Conv3d(1, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=True)]
        m_135 = [
            common.ResBlockc3d(n_feats, is_large_kernel=is_large_kernel) for _ in range(n_seb)
        ]

        s_list = [common.ResBlock2d(4 * n_feats, 4 * n_feats, kernel_size=(1, 3, 3)) for _ in range(n_sab)]  # 4
        a_list = [common.ResBlock2d(4 * n_feats, 4 * n_feats, kernel_size=(1, 3, 3)) for _ in range(n_sab)]  # 4

        m_upsample = [
            nn.ConvTranspose3d(4 * n_feats, n_feats, kernel_size=(1, scale + 2, scale + 2), stride=(1, scale, scale),
                               # 4
                               padding=(0, 1, 1), output_padding=(0, 0, 0), bias=True),
            nn.Conv3d(n_feats, 1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)]

        m_upsample_main = [
            nn.ConvTranspose3d(1, 1, kernel_size=(1, scale + 2, scale + 2), stride=(1, scale, scale), padding=(0, 1, 1),
                               output_padding=(0, 0, 0), bias=True)]

        self.horizontal_first = nn.Sequential(*m_horizontal_first)
        self.horizontal = nn.Sequential(*m_horizontal)
        self.vertical_first = nn.Sequential(*m_vertical_first)
        self.vertical = nn.Sequential(*m_vertical)
        self.s45_first = nn.Sequential(*m_45_first)
        self.s45 = nn.Sequential(*m_45)
        self.s135_first = nn.Sequential(*m_135_first)
        self.s135 = nn.Sequential(*m_135)

        self.s_body_list = nn.ModuleList(s_list)
        self.a_body_list = nn.ModuleList(a_list)

        self.upsample = nn.Sequential(*m_upsample)
        self.upsample_main = nn.Sequential(*m_upsample_main)
        self.scale = scale
        self.n_feats = n_feats
        self.n_sab = n_sab

    def forward(self, train_data):

        # extract the central view from the image stack
        ''' super-resolution horizontally '''
        batch_size = (train_data.shape[0])
        view_n = (train_data.shape[1])
        image_h = int(train_data.shape[3])
        image_w = int(train_data.shape[4])  # train_data.shape[4]

        horizontal = torch.zeros((batch_size, self.n_feats, view_n, view_n, int(image_h), int(image_w)),
                                 dtype=torch.float32)
        horizontal = horizontal.cuda()
        for i in range(0, view_n, 1):
            train_cut = train_data[:, i:i + 1, :, :, :]
            train_cut = self.horizontal_first(train_cut)
            train_cut = train_cut + self.horizontal(train_cut)  # (7,7,32,32)
            horizontal[:, :, i:i + 1, :, :, :] = train_cut.view(batch_size, self.n_feats, 1, view_n,
                                                                int(image_h), int(image_w))
        horizontal = horizontal.view(-1, self.n_feats, view_n * view_n, image_h, image_w)  # (1,49,64,64)

        ''' super-resolution vertically '''
        vertical = torch.zeros((batch_size, self.n_feats, view_n, view_n, int(image_h), int(image_w)),
                               dtype=torch.float32)
        vertical = vertical.cuda()
        for i in range(0, view_n, 1):
            train_cut = train_data[:, :, i:i + 1, :, :]
            train_cut = train_cut.permute(0, 2, 1, 3, 4)
            train_cut = self.vertical_first(train_cut)
            train_cut = train_cut + self.vertical(train_cut)  # (7,7,32,32)
            vertical[:, :, :, i:i + 1, :, :] = train_cut.view(batch_size, self.n_feats, view_n, 1,
                                                              int(image_h), int(image_w))
        vertical = vertical.view(-1, self.n_feats, view_n * view_n, image_h, image_w)  # (1,49,64,64)

        ''' super-resolution 45'''
        s45 = torch.zeros((batch_size, self.n_feats, view_n, view_n, int(image_h), int(image_w)), dtype=torch.float32)
        s45 = s45.cuda()
        position_45 = utils.get_45_position(view_n)
        for item in position_45:
            s45_cut = train_data[:, item[0], item[1], :, :]
            s45_cut = s45_cut.view(batch_size, 1, len(item[0]), image_h, image_w)
            s45_cut = self.s45_first(s45_cut)
            s45_cut = s45_cut + self.s45(s45_cut)
            for i in range(len(item[0])):
                s45[:, :, item[0][i], item[1][i], :, :] = s45_cut[:, :, i, :, :]
        s45 = s45.view(-1, self.n_feats, view_n * view_n, image_h, image_w)

        ''' super-resolution 135'''
        s135 = torch.zeros((batch_size, self.n_feats, view_n, view_n, int(image_h), int(image_w)), dtype=torch.float32)
        s135 = s135.cuda()
        position_135 = utils.get_135_position(view_n)
        for item in position_135:
            s135_cut = train_data[:, item[0], item[1], :, :].view(batch_size, 1, len(item[0]), image_h, image_w)
            s135_cut = self.s135_first(s135_cut)
            s135_cut = s135_cut + self.s135(s135_cut)
            for i in range(len(item[0])):
                s135[:, :, item[0][i], item[1][i], :, :] = s135_cut[:, :, i, :, :]

        s135 = s135.view(-1, self.n_feats, view_n * view_n, image_h, image_w)

        # residual part
        train_data = train_data.view(batch_size, 1, view_n * view_n, int(image_h), int(image_w))
        train_data = self.upsample_main(train_data)

        full_up = torch.cat((horizontal, vertical, s45, s135), 1)  # (4*n_feats,49,64,64)

        for i in range(self.n_sab):
            full_up = self.s_body_list[i](full_up)
            full_up = full_up.permute(0, 1, 3, 4, 2)
            full_up = full_up.view(-1, 4 * self.n_feats, image_h * image_w, view_n, view_n)  # 4
            full_up = self.a_body_list[i](full_up)
            full_up = full_up.permute(0, 1, 3, 4, 2)
            full_up = full_up.view(-1, 4 * self.n_feats, view_n * view_n, image_h, image_w)  # 4


        full_up = self.upsample(full_up)
        full_up += train_data
        full_up = full_up.view(-1, view_n, view_n, image_h * self.scale, image_w * self.scale)  # (7,7,h,w)->(1,49,h,w)

        return full_up

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
