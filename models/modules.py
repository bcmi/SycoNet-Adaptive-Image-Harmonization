import torch
import functools
import trilinear
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer



def init_net(net, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net



def define_E(input_nc, output_nc, nef, nwf, netE, norm='batch', nl='lrelu', gpu_ids=[], linear=True, LUT_num=5):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)

    if netE == 'Syco': #Joy
        net = SycoNet(input_nc, output_nc, nef, nwf, n_blocks=5, norm_layer=norm_layer, nl_layer=nl_layer, linear=linear, LUT_num=LUT_num)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % netE)

    return init_net(net, gpu_ids)



def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw,
                           stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out



class SycoNet(nn.Module):
    def __init__(self, input_nc=3, nz=1, nef=64, nwf=128, n_blocks=4, norm_layer=None, nl_layer=None, linear=False, LUT_num=5):
        super(SycoNet, self).__init__()
        self.isLinear = linear
        max_nef = 4
        self.block0 = nn.Conv2d(input_nc+nz, nef, kernel_size=4, stride=2, padding=1, bias=True)

        self.block1 = BasicBlock(nef * min(max_nef, 1)+nz, nef * min(max_nef, 2), norm_layer, nl_layer)
        self.block2 = BasicBlock(nef * min(max_nef, 2)+nz, nef * min(max_nef, 3), norm_layer, nl_layer)
        self.block3 = BasicBlock(nef * min(max_nef, 3)+nz, nef * min(max_nef, 4), norm_layer, nl_layer)
        self.block4 = BasicBlock(nef * min(max_nef, 4)+nz, nef * min(max_nef, 4), norm_layer, nl_layer)

        self.nl = nn.Sequential(nl_layer(), nn.AvgPool2d(8))

        self.weight_predictor = nn.Conv2d(nwf, LUT_num, 1, padding=0)

    def forward(self, img_input, random_z):
        z_img = random_z.expand(random_z.size(0), random_z.size(1), img_input.size(2), img_input.size(3))
        inputs = torch.cat([img_input, z_img], 1)
        x0 = self.block0(inputs)
        z0 = random_z.expand(random_z.size(0), random_z.size(1), x0.size(2), x0.size(3))
        x1 = torch.cat([x0, z0], 1)
        x1 = self.block1(x1)
        z1 = random_z.expand(random_z.size(0), random_z.size(1), x1.size(2), x1.size(3))
        x2 = torch.cat([x1, z1], 1)
        x2 = self.block2(x2)
        z2 = random_z.expand(random_z.size(0), random_z.size(1), x2.size(2), x2.size(3))
        x3 = torch.cat([x2, z2], 1)
        x3 = self.block3(x3)
        z3 = random_z.expand(random_z.size(0), random_z.size(1), x3.size(2), x3.size(3))
        x4 = torch.cat([x3, z3], 1)
        x4 = self.block4(x4)
        features = self.nl(x4)
        outputs = self.weight_predictor(features)
        if self.isLinear:
            outputs = F.softmax(outputs, dim=1)
        return features, outputs

class Get3DLUT_identity(nn.Module):
    def __init__(self, dim=17):
        super(Get3DLUT_identity, self).__init__()
        buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n = i * dim*dim + j * dim + k
                    #x = lines[n].split()
                    buffer[0,i,j,k] = 1.0/(dim-1)*k #float(x[0]) 
                    buffer[1,i,j,k] = 1.0/(dim-1)*j #float(x[1])
                    buffer[2,i,j,k] = 1.0/(dim-1)*i #float(x[2])
                    #print(i,j,k,":",buffer[0,i,j,k],buffer[1,i,j,k],buffer[2,i,j,k])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        return output

class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim-1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)
        
        assert 1 == trilinear.forward(lut, x, output, dim, shift, binsize, W, H, batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        
        ctx.save_for_backward(*variables)
        
        return lut, output
    
    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
            
        assert 1 == trilinear.backward(x, x_grad, lut_grad, dim, shift, binsize, W, H, batch)
        return lut_grad, x_grad



class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D,self).__init__()

        self.weight_r = torch.ones(3,dim,dim,dim-1, dtype=torch.float)
        self.weight_r[:,:,:,(0,dim-2)] *= 2.0
        self.weight_g = torch.ones(3,dim,dim-1,dim, dtype=torch.float)
        self.weight_g[:,:,(0,dim-2),:] *= 2.0
        self.weight_b = torch.ones(3,dim-1,dim,dim, dtype=torch.float)
        self.weight_b[:,(0,dim-2),:,:] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):

        dif_r = LUT.LUT[:,:,:,:-1] - LUT.LUT[:,:,:,1:]
        dif_g = LUT.LUT[:,:,:-1,:] - LUT.LUT[:,:,1:,:]
        dif_b = LUT.LUT[:,:-1,:,:] - LUT.LUT[:,1:,:,:]
        tv = torch.mean(torch.mul((dif_r ** 2),self.weight_r)) + torch.mean(torch.mul((dif_g ** 2),self.weight_g)) + torch.mean(torch.mul((dif_b ** 2),self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn