import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform_for_SR, get_transform_for_SR_x4
from data.image_folder import make_dataset
from PIL import Image
import random

class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_D = os.path.join(opt.dataroot, opt.phase + 'D')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.C_paths = make_dataset(self.dir_C)
        self.D_paths = make_dataset(self.dir_D)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.C_paths = sorted(self.C_paths)
        self.D_paths = sorted(self.D_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)
        self.transform = get_transform(opt)
        self.transform_for_SR = get_transform_for_SR(opt)
        self.transform_for_SR_x4 = get_transform_for_SR_x4(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
            index_C = index % self.C_size
            index_D = index % self.D_size
        else:
            index_B = random.randint(0, self.B_size - 1)
            index_C = random.randint(0, self.C_size - 1)
            index_D = random.randint(0, self.D_size - 1)
        B_path = self.B_paths[index_B]
        C_path = self.C_paths[index_C]
        D_path = self.D_paths[index_D]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('RGB')
        D_img = Image.open(D_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        C = self.transform_for_SR(C_img)
        D = self.transform_for_SR(D_img)
        D_x4 = self.transform_for_SR_x4(D_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B, 'C': C, 'D': D, 'D_x4': D_x4,
                'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'D_paths': D_path}

    def __len__(self):
        return max(self.A_size, self.B_size, self.C_size, self.D_size)

    def name(self):
        return 'UnalignedDataset'
