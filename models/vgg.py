'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from pprint import pprint

"""
Number of layers of architecture - break-up ->
Total = Num Conv Layers * 1 + Num Fully Connected Layers * 1.
Thus, the totla number of layers having parameters.
The below nomenclature is probably wrong as per the above definition.
"""
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        pprint("out = self.features(x) -> out.shape {}".format(out.shape))

        # flatten the array
        out = out.view(out.size(0), -1)
        print(" out = out.view(out.size(0), -1) = {}".format(out.shape))
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # The below filter has input & output of same shape?
        # refer my own answer below link/issue reported in same repo - 
        # https://github.com/kuangliu/pytorch-cifar/issues/110
        # Refer experimental_code.py file for details.
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        """
        NOTE : This architecture is almost like a Fully Conv Network. If we make 
        the avgPooling layer as GlobalAvg2D, it becomes an FCN. Also, if you want 
        to double the input size (while keeping the same avgPooling layer), make
        the kernel_size=2 and stride = 2 and so on.
        """
        # layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    # pprint(x)
    pprint("x = torch.randn(2,3,32,32) = {}".format(x.shape))
    y = net(x)
    print(y.size())

# test()

from pytorch_model_summary import summary

model_name = 'VGG19'
print(model_name)
net = VGG(model_name)
batch_size = 2
# print(summary(net, torch.randn(2,3,32,32), show_input=True))
# print(summary(net, torch.randn(2,3,64,64), batch_size=2, show_input=True, show_hierarchical=True, max_depth=True, show_parent_layers=True))
print(summary(net, torch.randn(batch_size,3,32,32), batch_size=batch_size, show_input=True, show_hierarchical=True, max_depth=True, show_parent_layers=True))