import torch.nn as nn
from dropout import ExampleTiedDropout

class Mul(nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
            nn.Conv2d(channels_in, channels_out,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=groups, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
    )

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x


class ResNet9_dropout(nn.Module):
    def __init__(self, p_fixed = 0.2, p_mem = 0.1, num_batches = 100, drop_mode = "train", in_channels = 3, fac = 1, NUM_CLASSES = 10):
        super(ResNet9_dropout, self).__init__()
        self.p_fixed = p_fixed
        self.p_mem = p_mem
        self.num_batches = num_batches
        self.drop_mode = drop_mode

        dims = [in_channels, 64, 128, 128, 128, 256, 256, 256,128]
        dims = [int(d*fac) for d in dims]
        dims[0] = in_channels
        self.dims = dims

        self.dropout = self.get_dropout(p_fixed, p_mem, num_batches, drop_mode)

        self.conv1 = conv_bn(dims[0], dims[1], kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_bn(dims[1], dims[2], kernel_size=5, stride=2, padding=2)
        self.res1 = Residual(nn.Sequential(conv_bn(dims[2], dims[3]), conv_bn(dims[3], dims[4])))
        self.conv3 = conv_bn(dims[4], dims[5], kernel_size=3, stride=1, padding=1)
        self.res2 = Residual(nn.Sequential(conv_bn(dims[5], dims[6]), conv_bn(dims[6], dims[7])))
        self.conv4 = conv_bn(dims[7], dims[8], kernel_size=3, stride=1, padding=0)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = Flatten()
        self.linear = nn.Linear(dims[8], NUM_CLASSES, bias=False)
        self.mul  = Mul(0.2)

    def get_dropout(self, p_fixed, p_mem, num_batches, drop_mode):
        # return ExampleTiedDropout(p_fixed=p_fixed, p_mem=p_mem,num_batches=num_batches, drop_mode = drop_mode)
        return nn.Dropout(p=p_fixed)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.dropout(x)

        x = self.res1(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)

        x = self.maxpool(x)
        x = self.res2(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.dropout(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.mul(x)
        return x

def ResNet9(NUM_CLASSES = 10, in_channels = 3):
    return nn.Sequential(
        #[bs, 3, n, n]
        conv_bn(in_channels, 64, kernel_size=3, stride=1, padding=1),
        #[bs, 64, n, n]
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        #[bs, 128, n/2, n/2]
        Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        #[bs, 128, n/2, n/2]
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        #[bs, 256, n/2, n/2]
        nn.MaxPool2d(2),
        #[bs, 256, n/4, n/4]
        Residual(nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        #[bs, 256, n/4, n/4]
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        #[bs, 128, n/4, n/4]
        nn.AdaptiveMaxPool2d((1, 1)),
         #[bs, 128, 1, 1]
        Flatten(),
        nn.Linear(128, NUM_CLASSES, bias=False),
        Mul(0.2)
    )

def ResNet9_sink(NUM_CLASSES = 10, in_channels = 3, sink = 0):
    dims = [in_channels, 64, 128, 128, 128, 256, 256, 256,128]
    dims[sink] = dims[sink]*10
    return nn.Sequential(
        #[bs, 3, n, n]
        conv_bn(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
        #[bs, 64, n, n]
        conv_bn(dims[1], dims[2], kernel_size=5, stride=2, padding=2),
        #[bs, 128, n/2, n/2]
        Residual(nn.Sequential(conv_bn(dims[2], dims[3]), conv_bn(dims[3], dims[4]))),
        #[bs, 128, n/2, n/2]
        conv_bn(dims[4], dims[5], kernel_size=3, stride=1, padding=1),
        #[bs, 256, n/2, n/2]
        nn.MaxPool2d(2),
        #[bs, 256, n/4, n/4]
        Residual(nn.Sequential(conv_bn(dims[5], dims[6]), conv_bn(dims[6], dims[7]))),
        #[bs, 256, n/4, n/4]
        conv_bn(dims[7], dims[8], kernel_size=3, stride=1, padding=0),
        #[bs, 128, n/4, n/4]
        nn.AdaptiveMaxPool2d((1, 1)),
         #[bs, 128, 1, 1]
        Flatten(),
        nn.Linear(dims[8], NUM_CLASSES, bias=False),
        Mul(0.2)
    )

def ResNet5(NUM_CLASSES = 10, in_channels = 3):
    return nn.Sequential(
        #[bs, 3, n, n]
        conv_bn(in_channels, 128, kernel_size=5, stride=2, padding=2),
        #[bs, 128, n/2, n/2]
        Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        #[bs, 128, n/2, n/2]
        nn.MaxPool2d(2),
        #[bs, 128, n/4, n/4]
        Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        #[bs, 128, n/4, n/4]
        nn.AdaptiveMaxPool2d((1, 1)),
         #[bs, 128, 1, 1]
        Flatten(),
        nn.Linear(128, NUM_CLASSES, bias=False),
        Mul(0.2)
    )


def LeNet(NUM_CLASSES = 10, in_channels = 1):
    return nn.Sequential(
                nn.Conv2d(in_channels, 32, 5, padding = 2), 
                nn.ReLU(), 
                nn.MaxPool2d(2, 2), 
                nn.Conv2d(32, 64, 5, padding = 2), 
                nn.ReLU(), 
                nn.MaxPool2d(2, 2), 
                Flatten(), 
                nn.Linear(7*7*64, 1024), 
                nn.ReLU(), 
                nn.Linear(1024, NUM_CLASSES)
        )

def ResNet50_old(NUM_CLASSES = 10, in_channels = 3):
    from custom_resnet import CustomResNet50
    # For 32*32 images we use a smaller kernel size 
    # in the first layer to get good performance.
    model = CustomResNet50(num_classes=NUM_CLASSES,ks=3,in_channels=in_channels)
    return model

def ResNet50(NUM_CLASSES = 10, in_channels = 3):
    import torchvision
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(2048, NUM_CLASSES)
    if in_channels!=3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input

    # model = model.cuda()
    return model

def get_resnet_18(NUM_CLASSES, in_channels):
    from torchvision import models
    model = models.resnet18()
    model.fc = nn.Linear(512, NUM_CLASSES)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model

def get_vgg11(NUM_CLASSES, in_channels):
    from torchvision import models
    model = models.vgg11_bn()
    model.classifier[-1] = nn.Linear(4096, NUM_CLASSES)
    if in_channels != 3:
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    return model

def get_vgg16(NUM_CLASSES, in_channels):
    from torchvision import models
    model = models.vgg16_bn()
    model.classifier[-1] = nn.Linear(4096, NUM_CLASSES)
    if in_channels != 3:
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    return model

def get_vit(NUM_CLASSES, in_channels, image_size = 32):
    from vit_small import ViT

    model = ViT(
        image_size = image_size,
        patch_size = 2,
        num_classes = 10,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels = in_channels
    )

    return model

def get_resnet9_sink(sink):
    def new_f(NUM_CLASSES, in_channels):
        return ResNet9_sink(NUM_CLASSES = NUM_CLASSES, in_channels = in_channels, sink = sink)
    return new_f



def get_model(model_type, in_channels = 3, NUM_CLASSES=10, sink = 0, image_size = 32):
    model_mapper = {"resnet9": ResNet9, 
                    "resnet9_sink":get_resnet9_sink(sink), 
                    "lenet": LeNet, 
                    "resnet50":ResNet50, 
                    "resnet18":get_resnet_18, 
                    "vgg11":get_vgg11,
                    "vgg16":get_vgg16,
                    "vit":get_vit}
    if model_type == "vit":
        model = model_mapper[model_type](NUM_CLASSES, in_channels, image_size = image_size).cuda()
    else:
        model = model_mapper[model_type](NUM_CLASSES, in_channels).cuda()
    
    return model


