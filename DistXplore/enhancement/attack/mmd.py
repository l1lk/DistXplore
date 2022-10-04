import torch
import torch.nn as nn


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # print("source {}".format(source))
    # print("target {}".format(target))
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    # print("total.size(0) {}".format(total.size(0)))
    # print("total.size(1) {}".format(total.size(1)))
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # print("L2_distance {}".format(L2_distance))
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    # print("bandwidth {}".format(bandwidth))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # for bandwidth_temp in bandwidth_list:
    #     print("bandwidth_temp {}".format(bandwidth_temp))
    #     print("-L2_distance {}".format(-L2_distance))
    #     print("result {}".format(torch.exp(-L2_distance / bandwidth_temp)))
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # print("kernels val {}".format(kernel_val))
    return sum(kernel_val)


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                  fix_sigma=self.fix_sigma)
        # print("kernels {}".format(kernels))
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        # print("xx {}".format(XX))
        # print("yy {}".format(YY))
        # print("XY {}".format(XY))
        # print("YX {}".format(YX))
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss
