import mxnet as mx

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable
import _ext as _backend
#
# """
# demo symbol of using modulated deformable convolution
# """
# def modulated_deformable_conv(data, name, num_filter, stride, lr_mult=0.1):
#     weight_var = mx.sym.Variable(name=name+'_conv2_offset_weight', init=mx.init.Zero(), lr_mult=lr_mult)#创建变量,权重
#     bias_var = mx.sym.Variable(name=name+'_conv2_offset_bias', init=mx.init.Zero(), lr_mult=lr_mult)#创建变量，偏置
#     conv2_offset = mx.symbol.Convolution(name=name + '_conv2_offset', data=data, num_filter=27,
#                        pad=(1, 1), kernel=(3, 3), stride=stride, weight=weight_var, bias=bias_var, lr_mult=lr_mult)
#     conv2_offset_t = mx.sym.slice_axis(conv2_offset, axis=1, begin=0, end=18)#直接在某一维上切割，选择整行或整列，axis:要切篇的轴为1时时列，0为行，begin end :左开右闭，
#     conv2_mask =  mx.sym.slice_axis(conv2_offset, axis=1, begin=18, end=None)
#     conv2_mask = 2 * mx.sym.Activation(conv2_mask, act_type='sigmoid')
#
#     conv2 = mx.contrib.symbol.ModulatedDeformableConvolution(name=name + '_conv2', data=data, offset=conv2_offset_t, mask=conv2_mask,
#                        num_filter=num_filter, pad=(1, 1), kernel=(3, 3), stride=stride,
#                        num_deformable_group=1, no_bias=True)
#     return conv2

"""
demo symbol of using modulated deformable RoI pooling
"""
# def modulated_deformable_roi_pool(data, rois, spatial_scale, imfeat_dim=256, deform_fc_dim=1024, roi_size=7, trans_std=0.1):
#     roi_align = mx.contrib.sym.DeformablePSROIPooling(name='roi_align',
#                         data=data,
#                         rois=rois,
#                         group_size=1,
#                         pooled_size=roi_size,
#                         sample_per_part=2,
#                         no_trans=True,
#                         part_size=roi_size,
#                         output_dim=imfeat_dim,
#                         spatial_scale=spatial_scale)
#
#     feat_deform = mx.symbol.FullyConnected(name='fc_deform_1', data=roi_align, num_hidden=deform_fc_dim)
#     feat_deform = mx.sym.Activation(data=feat_deform, act_type='relu', name='fc_deform_1_relu')
#
#     feat_deform = mx.symbol.FullyConnected(name='fc_deform_2', data=feat_deform, num_hidden=deform_fc_dim)
#     feat_deform = mx.sym.Activation(data=feat_deform, act_type='relu', name='fc_deform_2_relu')
#
#     feat_deform = mx.symbol.FullyConnected(name='fc_deform_3', data=feat_deform, num_hidden=roi_size * roi_size * 3)
#
#     roi_offset = mx.sym.slice_axis(feat_deform, axis=1, begin=0, end=roi_size * roi_size * 2)
#     roi_offset = mx.sym.reshape(roi_offset, shape=(-1, 2, roi_size, roi_size))
#
#     roi_mask = mx.sym.slice_axis(feat_deform, axis=1, begin=roi_size * roi_size * 2, end=None)
#     roi_mask_sigmoid = mx.sym.Activation(roi_mask, act_type='sigmoid')
#     roi_mask_sigmoid = mx.sym.reshape(roi_mask_sigmoid, shape=(-1, 1, roi_size, roi_size))
#
#     deform_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deform_roi_pool',
#                         data=data,
#                         rois=rois,
#                         trans=roi_offset,
#                         group_size=1,
#                         pooled_size=roi_size,
#                         sample_per_part=2,
#                         no_trans=False,
#                         part_size=roi_size,
#                         output_dim=imfeat_dim,
#                         spatial_scale=spatial_scale,
#                         trans_std=trans_std)
#
#     modulated_deform_roi_pool = mx.sym.broadcast_mul(deform_roi_pool, roi_mask_sigmoid)
#     return modulated_deform_roi_pool



class _DCNv2(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride, padding, dilation, deformable_groups):
        ctx.stride = _pair(stride)#步长
        ctx.padding = _pair(padding)#输入的两边都加上了零填充
        ctx.dilation = _pair(dilation)#卷积核之间的间距
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups#从输入通道到输出通道的阻塞连接数
        output = _backend.dcn_v2_forward(input, weight, bias,
                                         offset, mask,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.deformable_groups)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
            _backend.dcn_v2_backward(input, weight,
                                     bias,
                                     offset, mask,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.deformable_groups)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias,\
            None, None, None, None,


dcn_v2_conv = _DCNv2.apply

class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            mask.shape[1]
        return dcn_v2_conv(input, offset, mask,
                           self.weight,
                           self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)


class DCN(DCNv2):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask,
                           self.weight, self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)