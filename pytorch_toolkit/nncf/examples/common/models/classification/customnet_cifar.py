import torch.nn as nn
import torch.nn.functional as F

class CustomNetClassifierCifar(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(CustomNetClassifierCifar, self).__init__()
        self.conv = nn.Conv2d(3, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)
        self.name = 'customnet_cifar'

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 4 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

def customnet(num_classes, pretrained=False, **kwargs):
    model = CustomNetClassifierCifar(num_classes, pretrained, **kwargs)

    if pretrained:
        print("CustomNetClassifierCifar has no pretrained weights")

    return model

# import os
# from distutils.spawn import find_executable
# from torchsummary import summary

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = CustomNetClassifierCifar().to(device)
# summary(net, input_size=(3, 32, 32))

# dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True).to(device)
# result = net(dummy_input)

# torch.onnx.export(net,
#                   dummy_input,
#                   net.name+'.onnx',
#                   input_names=['input'],
#                   output_names=['output'])

# command = 'mo_onnx.py'
# if find_executable(command) is not None:
#     command += ' -m ' + net.name + '.onnx --data_type=FP32 --generate_deprecated_IR_V7'
#     os.system(command)
# else:
#     raise Exception('Model Optimizer not found, exiting...')
