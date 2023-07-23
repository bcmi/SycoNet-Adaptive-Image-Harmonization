import torch
from .base_model import BaseModel
from . import modules
from torch import cuda

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class LABASELUTSModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.visual_names = ['mask', 'real', 'transfer_img_c']
        self.model_names = ['Er']

        self.LUT_num = opt.LUT_num

        self.baseLUTs = []
        for _ in range(self.LUT_num):
            self.baseLUTs.append(modules.Get3DLUT_identity(dim=17).to(self.device))
        self.LUT_num = len(self.baseLUTs)
        print('total LUT numbers: %d' % self.LUT_num)

        self.netEr = modules.define_E(opt.input_nc, opt.nz, opt.nef, opt.nwf, opt.netEr, opt.norm,
                                          'lrelu', self.gpu_ids, True, self.LUT_num)


        self.TV3 = modules.TV_3D().to(self.device)
        self.TV3.weight_r = self.TV3.weight_r.type(Tensor)
        self.TV3.weight_g = self.TV3.weight_g.type(Tensor)
        self.TV3.weight_b = self.TV3.weight_b.type(Tensor)
        self.trilinear_ = modules.TrilinearInterpolation()


    def set_input(self, tgt_input):
        self.mask = tgt_input['mask'].to(self.device)
        self.real = tgt_input['real'].to(self.device)
        self.inputs = torch.cat([self.real,self.mask], 1)

        self.real_raw = tgt_input['real_raw'].to(self.device)
        self.mask_raw = tgt_input['mask_raw'].to(self.device)


    def get_z_random(self, size, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(size) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(size)
        return z.detach().to(self.device)
   

    def generator_eval(self, pred, img, mask, LUTs):

        pred = pred.squeeze()

        for ii in range(self.LUT_num):
            self.baseLUTs[ii].eval()

        LUT = pred[0] * self.baseLUTs[0].LUT
        for idx in range(1, self.LUT_num):
            LUT += pred[idx] * self.baseLUTs[idx].LUT


        _, combine_A = self.trilinear_(LUT,img)

        combine_A = combine_A * mask + img * (1 - mask)
        combine_A = torch.clamp(combine_A, 0, 1)

        return combine_A


    def forward(self):
        self.z_size = [self.inputs.size(0), self.opt.nz, 1, 1]
        self.z_random = self.get_z_random(self.z_size)

        self.features_c, self.weightPred_c = self.netEr(self.inputs, self.z_random)
        if self.opt.keep_res:
            self.transfer_img_c = self.generator_eval(self.weightPred_c, self.real_raw, self.mask_raw, self.baseLUTs)
        else:
            self.transfer_img_c = self.generator_eval(self.weightPred_c, self.real, self.mask, self.baseLUTs)


