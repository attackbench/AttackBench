import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)


class SparsifyBase(nn.Module):
    def __init__(self, sparse_ratio=0.5):
        super(SparsifyBase, self).__init__()
        self.sr = sparse_ratio
        self.preact = None
        self.act = None

    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()

        return hook

    def record_activation(self):
        self.register_forward_hook(self.get_activation())


class Sparsify1D(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify1D, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        k = int(self.sr * x.shape[1])
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify1D_kactive(SparsifyBase):
    def __init__(self, k=1):
        super(Sparsify1D_kactive, self).__init__()
        self.k = k

    def forward(self, x):
        k = self.k
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D, self).__init__()
        self.sr = sparse_ratio

        self.preact = None
        self.act = None

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        tmpx = x.flatten(start_dim=2)
        topval = tmpx.topk(k, dim=2)[0][:, :, -1]
        topval = topval.expand(x.shape[2], x.shape[3], x.shape[0], x.shape[1]).permute(2, 3, 0, 1)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D_vol(SparsifyBase):
    '''cross channel sparsify'''

    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_vol, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        size = x.shape[1] * x.shape[2] * x.shape[3]
        k = int(self.sr * size)

        tmpx = x.flatten(start_dim=1)
        topval = tmpx.topk(k, dim=1)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).reshape_as(x)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D_kactive(SparsifyBase):
    '''cross channel sparsify'''

    def __init__(self, k):
        super(Sparsify2D_vol, self).__init__()
        self.k = k

    def forward(self, x):
        k = self.k
        tmpx = x.flatten(start_dim=1)
        topval = tmpx.topk(k, dim=1)[0][:, -1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1, 0).reshape_as(x)
        comp = (x >= topval).to(x)
        return comp * x


class Sparsify2D_abs(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_abs, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        absx = torch.abs(x)
        tmpx = absx.flatten(start_dim=2)
        topval = tmpx.topk(k, dim=2)[0][:, :, -1]
        topval = topval.expand(absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]).permute(2, 3, 0, 1)
        comp = (absx >= topval).to(x)
        return comp * x


class Sparsify2D_invabs(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_invabs, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        layer_size = x.shape[2] * x.shape[3]
        k = int(self.sr * layer_size)
        absx = torch.abs(x)
        tmpx = absx.flatten(start_dim=2)
        topval = tmpx.topk(k, dim=2, largest=False)[0][:, :, -1]
        topval = topval.expand(absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]).permute(2, 3, 0, 1)
        comp = (absx >= topval).to(x)
        return comp * x


class breakReLU(nn.Module):
    def __init__(self, sparse_ratio=5):
        super(breakReLU, self).__init__()
        self.h = sparse_ratio
        self.thre = nn.Threshold(0, -self.h)

    def forward(self, x):
        return self.thre(x)


class SmallCNN(nn.Module):
    def __init__(self, fc_in=3136, n_classes=10):
        super(SmallCNN, self).__init__()

        self.module_list = nn.ModuleList([nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                                          Flatten(),
                                          nn.Linear(fc_in, 100), nn.ReLU(),
                                          nn.Linear(100, n_classes)])

    def forward(self, x):
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
        return x

    def forward_to(self, x, layer_i):
        for i in range(layer_i):
            x = self.module_list[i](x)
        return x


sparse_func_dict = {
    'reg': Sparsify2D,  # top-k value
    'abs': Sparsify2D_abs,  # top-k absolute value
    'invabs': Sparsify2D_invabs,  # top-k minimal absolute value
    'vol': Sparsify2D_vol,  # cross channel top-k
    'brelu': breakReLU,  # break relu
    'kact': Sparsify2D_kactive,
    'relu': nn.ReLU
}

import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SparseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sparsity=0.5, use_relu=True, sparse_func='reg', bias=False):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.use_relu = use_relu
        self.sparse1 = sparse_func_dict[sparse_func](sparsity)
        self.sparse2 = sparse_func_dict[sparse_func](sparsity)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bias=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SparseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, sparsity=0.5, use_relu=True, sparse_func='reg', bias=True):
        super(SparseBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()

        self.sparse1 = sparse_func_dict[sparse_func](sparsity)
        self.sparse2 = sparse_func_dict[sparse_func](sparsity)
        self.sparse3 = sparse_func_dict[sparse_func](sparsity)

        self.use_relu = use_relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse1(out)

        out = self.bn2(self.conv2(out))
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse2(out)

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        if self.use_relu:
            out = self.relu(out)
        out = self.sparse3(out)
        return out


class SparseResNet(nn.Module):
    def __init__(self, block, num_blocks, sparsities, num_classes=10, use_relu=True, sparse_func='reg', bias=True):
        super(SparseResNet, self).__init__()
        self.in_planes = 64
        self.use_relu = use_relu
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, sparsity=sparsities[0],
                                       sparse_func=sparse_func, bias=bias)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, sparsity=sparsities[1],
                                       sparse_func=sparse_func, bias=bias)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, sparsity=sparsities[2],
                                       sparse_func=sparse_func, bias=bias)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, sparsity=sparsities[3],
                                       sparse_func=sparse_func, bias=bias)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()

        return hook

    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))

    def _make_layer(self, block, planes, num_blocks, stride, sparsity=0.5, sparse_func='reg', bias=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, sparsity, self.use_relu, sparse_func=sparse_func, bias=bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.flatten(start_dim=1)
        out = self.linear(out)
        return out


class SparseResNet_ImageNet(nn.Module):
    def __init__(self, block, num_blocks, sparsities, num_classes=1000, sparse_func='vol', bias=False):
        super(SparseResNet_ImageNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, sparsity=sparsities[0],
                                       sparse_func=sparse_func, bias=bias)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, sparsity=sparsities[1],
                                       sparse_func=sparse_func, bias=bias)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, sparsity=sparsities[2],
                                       sparse_func=sparse_func, bias=bias)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, sparsity=sparsities[3],
                                       sparse_func=sparse_func, bias=bias)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.sp = sparse_func_dict[sparse_func](sparsities[0])

        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()

        return hook

    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))

    def _make_layer(self, block, planes, num_blocks, stride, sparsity=0.5, sparse_func='reg', bias=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, sparsity, use_relu=False, sparse_func=sparse_func, bias=bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.sp(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(start_dim=1)
        out = self.linear(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()

        return hook

    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.flatten(start_dim=1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def SparseResNet18(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBasicBlock, [2, 2, 2, 2], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)


def SparseResNet34(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBasicBlock, [3, 4, 6, 3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)


def SparseResNet50(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBottleneck, [3, 4, 6, 3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)


def SparseResNet101(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBottleneck, [3, 4, 23, 3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)


def SparseResNet152(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBottleneck, [3, 8, 36, 3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)


def SparseResNet152_ImageNet(relu=False, sparsities=[0.5, 0.4, 0.3, 0.2], sparse_func='reg', bias=False):
    return SparseResNet_ImageNet(SparseBottleneck, [3, 8, 36, 3], sparsities, sparse_func=sparse_func, bias=bias)


########### End resnet related ##################
sparse_func_dict = {
    'reg': Sparsify2D,  # top-k value
    'abs': Sparsify2D_abs,  # top-k absolute value
    'invabs': Sparsify2D_invabs,  # top-k minimal absolute value
    'vol': Sparsify2D_vol,  # cross channel top-k
    'brelu': breakReLU,  # break relu
    'kact': Sparsify2D_kactive,
    'relu': nn.ReLU
}
