import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.transforms as transforms
from PIL import Image
import scipy.misc as sm
import numpy as np
import pdb

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_C', type=float, default=10.0, help='weight for cycle loss (C -> D -> C)')
            parser.add_argument('--lambda_D', type=float, default=10.0, help='weight for cycle loss (D -> C -> D)')            
            #parser.add_argument('--lambda_rain', type=float, default=100000.0, help='weight for rain loss (Normlization)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'D_C', 'G_C', 'cycle_C', 'idt_C', 'D_D', 'G_D', 'cycle_D', 'idt_D']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        visual_names_C = ['real_C', 'fake_D', 'rec_C']
        visual_names_D = ['real_D', 'fake_C', 'rec_D']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')
            visual_names_C.append('idt_C')
            visual_names_D.append('idt_D')

        self.visual_names = visual_names_A + visual_names_B + visual_names_C + visual_names_D
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            self.model_names_for_SR = ['G_C', 'G_D', 'D_C', 'D_D']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'G_C', 'G_D']
            self.model_names_for_SR = ['G_C', 'G_D']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_C = networks.define_EDSR(init_type='normal', init_gain=0.02, gpu_ids=[])
        self.netG_D = networks.define_G_for_SR(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)        

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_C = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_D = networks.define_D(opt.input_nc, opt.ndf, opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)            

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_C_pool = ImagePool(opt.pool_size)
            self.fake_D_pool = ImagePool(opt.pool_size)            
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_for_SR = torch.optim.Adam(itertools.chain(self.netG_C.parameters(), self.netG_D.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_for_SR = torch.optim.Adam(itertools.chain(self.netD_C.parameters(), self.netD_D.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))            
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_G_for_SR)
            self.optimizers.append(self.optimizer_D_for_SR)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        #self.real_C = input['C'].to(self.device)
        self.real_D = input['D'].to(self.device)        
        self.real_D_x4 = input['D_x4'].to(self.device)        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

        self.real_C = self.fake_B.detach()
        self.fake_D = self.netG_C(self.real_C) #****#
        self.rec_C = self.netG_D(self.fake_D)

        self.fake_C = self.netG_D(self.real_D)
        self.rec_D = self.netG_C(self.fake_C)


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_D_C(self):
        fake_D = self.fake_D_pool.query(self.fake_D)
        self.loss_D_C = self.backward_D_basic(self.netD_C, self.real_D, fake_D)

    def backward_D_D(self):
        fake_C = self.fake_C_pool.query(self.fake_C)
        self.loss_D_D = self.backward_D_basic(self.netD_D, self.real_C, fake_C)                

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
        lambda_D = self.opt.lambda_D        
        #lambda_rain = self.opt.lambda_rain
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            # G_C should be identity if real_D is fed.
            self.idt_C = self.netG_C(self.real_D_x4)
            self.loss_idt_C = self.criterionIdt(self.idt_C, self.real_D) * lambda_D * lambda_idt
            # G_D should be identity if real_C is fed.
            #pdb.set_trace()

            #Strategy-1:
            # self.real_C_copy = self.real_C
            # image_numpy = self.real_C_copy.data[0].cpu().float().numpy()
            # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            # image_numpy.astype(np.uint8)
            # self.real_C_resize = sm.imresize(image_numpy, (self.opt.fineSize*4, self.opt.fineSize*4))
            # self.real_C_tensor = Image.fromarray(self.real_C_resize)
            # self.trans = transforms.Compose([transforms.ToTensor()])            
            # self.C_final = torch.unsqueeze(self.trans(self.real_C_tensor), 0).to(self.device)
            # self.idt_D = self.netG_D(self.C_final.detach())
            # self.loss_idt_D = self.criterionIdt(self.idt_D, self.real_C) * lambda_C * lambda_idt

            #Strategy-2:    
            # self.real_C_copy = self.real_C
            # self.real_C_copy = (self.real_C_copy.data[0].cpu() + 1) / 2.0 * 255.0
            # self.real_C_copy = transforms.ToPILImage()(self.real_C_copy).convert('RGB')
            # self.trans = transforms.Compose([transforms.Resize([self.opt.fineSize*4, self.opt.fineSize*4], Image.BICUBIC), transforms.ToTensor()])            
            # self.C_final = torch.unsqueeze(self.trans(self.real_C_copy), 0).to(self.device)
            # self.idt_D = self.netG_D(self.C_final.detach())
            # self.loss_idt_D = self.criterionIdt(self.idt_D, self.real_C) * lambda_C * lambda_idt     

            #Strategy-3:      
            self.real_C_copy = self.real_C
            image_numpy = (self.real_C_copy.data[0].cpu().float().numpy() + 1) / 2.0 * 255.0
            image_numpy = np.transpose(image_numpy, (1,2,0))
            self.real_C_tensor = Image.fromarray(image_numpy.astype(np.uint8))
            self.trans = transforms.Compose([transforms.Resize([self.opt.fineSize*4, self.opt.fineSize*4], Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])            
            self.C_final = torch.unsqueeze(self.trans(self.real_C_tensor), 0).to(self.device)
            self.idt_D = self.netG_D(self.C_final.detach())
            self.loss_idt_D = self.criterionIdt(self.idt_D, self.real_C) * lambda_C * lambda_idt                
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            self.loss_idt_C = 0
            self.loss_idt_D = 0
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # GAN loss D_C(G_C(C))
        self.loss_G_C = self.criterionGAN(self.netD_C(self.fake_D), True)
        # GAN loss D_D(G_D(D))
        self.loss_G_D = self.criterionGAN(self.netD_D(self.fake_C), True)        
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # Forward cycle loss
        self.loss_cycle_C = self.criterionCycle(self.rec_C, self.real_C) * lambda_C
        # Backward cycle loss
        self.loss_cycle_D = self.criterionCycle(self.rec_D, self.real_D) * lambda_D        
        
        # Myself Rain loss
        # self.rain_sub = self.real_A - self.fake_B
        # self.rain_add = self.fake_A - self.real_B
        # self.loss_rain = self.criterionCycle(self.rain_sub, self.rain_add) / lambda_rain

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B        
        self.loss_G.backward(retain_graph=True)
        # combined loss for SR
        self.loss_G_for_SR = self.loss_G_C + self.loss_G_D + self.loss_cycle_C + self.loss_cycle_D + self.loss_idt_C + self.loss_idt_D       
        self.loss_G_for_SR.backward()        

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B and G_C and G_D
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D], False)
        self.optimizer_G.zero_grad()
        self.optimizer_G_for_SR.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_G_for_SR.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D], True)
        self.optimizer_D.zero_grad()
        self.optimizer_D_for_SR.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.backward_D_C()
        self.backward_D_D()
        self.optimizer_D.step()
        self.optimizer_D_for_SR.step()
