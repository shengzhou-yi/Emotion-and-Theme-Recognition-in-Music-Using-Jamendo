import torch.nn as nn
from torch.autograd import Variable

class AudioNet(nn.Module):
    def __init__(self, num_class=15):
        super(AudioNet, self).__init__()

        # init bn
        self.bn_init = nn.BatchNorm2d(1)

        #block 1
        in_width = 1
        out_width = 64
        self.conv_1 = nn.Conv2d(in_width, out_width, 5, padding = 3)
        self.bn_1 = nn.BatchNorm2d(out_width)

        self.mp_1 = nn.MaxPool2d((3, 5))

        #block 2
        in_width = 64
        out_width = 64
        self.conv_2 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.bn_2 = nn.BatchNorm2d(out_width)
        self.conv_3 = nn.Conv2d(out_width, out_width, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(out_width)
        self.conv_4 = nn.Conv2d(out_width, out_width, 1, padding=0)
        self.bn_4 = nn.BatchNorm2d(out_width)

        #block 3
        in_width = 64
        out_width = 128
        self.conv_5 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.bn_5 = nn.BatchNorm2d(out_width)
        self.conv_6 = nn.Conv2d(out_width, out_width, 3, padding=1, stride = 2)
        self.bn_6 = nn.BatchNorm2d(out_width)
        self.conv_7 = nn.Conv2d(out_width, out_width, 1, padding=0)
        self.bn_7 = nn.BatchNorm2d(out_width)

        self.sc_conv_1 = nn.Conv2d(in_width, out_width, 1, padding=0, stride = 2)
        self.sc_bn_1 = nn.BatchNorm2d(out_width)
        self.mp_2 = nn.MaxPool2d((2, 4))

        #block 4
        in_width = 128
        out_width = 128
        self.conv_8 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.bn_8 = nn.BatchNorm2d(out_width)
        self.conv_9 = nn.Conv2d(out_width, out_width, 3, padding=1)
        self.bn_9 = nn.BatchNorm2d(out_width)
        self.conv_10 = nn.Conv2d(out_width, out_width, 1, padding=0)
        self.bn_10 = nn.BatchNorm2d(out_width)

        #block 5
        in_width = 128
        out_width = 256
        self.conv_11 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.bn_11 = nn.BatchNorm2d(out_width)
        self.conv_12 = nn.Conv2d(out_width, out_width, 3, padding=1, stride = 2)
        self.bn_12 = nn.BatchNorm2d(out_width)
        self.conv_13 = nn.Conv2d(out_width, out_width, 1, padding=0)
        self.bn_13 = nn.BatchNorm2d(out_width)

        self.sc_conv_2 = nn.Conv2d(in_width, out_width, 1, padding=0, stride = 2)
        self.sc_bn_2 = nn.BatchNorm2d(out_width)
        self.mp_3 = nn.MaxPool2d((2, 4))

        #block 6
        in_width = 256
        out_width = 256
        self.conv_14 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.bn_14 = nn.BatchNorm2d(out_width)
        self.conv_15 = nn.Conv2d(out_width, out_width, 3, padding=1)
        self.bn_15 = nn.BatchNorm2d(out_width)
        self.conv_16 = nn.Conv2d(out_width, out_width, 1, padding=0)
        self.bn_16 = nn.BatchNorm2d(out_width)

        self.mp_4 = nn.MaxPool2d((2, 4))

        #classifier
        self.dense = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)

        #init bn
        x = self.bn_init(x)

        #block 1
        x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))

        #block 2
        out = x

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = nn.ELU()(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = nn.ELU()(x)
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = nn.ELU()(out + x)

        #block 3
        out = self.sc_conv_1(x)
        out = self.sc_bn_1(out)

        x = self.conv_5(x)
        x = self.bn_5(x)
        x = nn.ELU()(x)
        x = self.conv_6(x)
        x = self.bn_6(x)
        x = nn.ELU()(x)
        x = self.conv_7(x)
        x = self.bn_7(x)
        x = nn.ELU()(out + x)

        x = self.mp_2(x)

        #block 4
        out = x

        x = self.conv_8(x)
        x = self.bn_8(x)
        x = nn.ELU()(x)
        x = self.conv_9(x)
        x = self.bn_9(x)
        x = nn.ELU()(x)
        x = self.conv_10(x)
        x = self.bn_10(x)
        x = nn.ELU()(out + x)

        #block 5
        out = self.sc_conv_2(x)
        out = self.sc_bn_2(out)

        x = self.conv_11(x)
        x = self.bn_11(x)
        x = nn.ELU()(x)
        x = self.conv_12(x)
        x = self.bn_12(x)
        x = nn.ELU()(x)
        x = self.conv_13(x)
        x = self.bn_13(x)
        x = nn.ELU()(out + x)

        x = self.mp_3(x)

        #block 6
        out = x

        x = self.conv_14(x)
        x = self.bn_14(x)
        x = nn.ELU()(x)
        x = self.conv_15(x)
        x = self.bn_15(x)
        x = nn.ELU()(x)
        x = self.conv_16(x)
        x = self.bn_16(x)
        x = nn.ELU()(out + x)

        x = self.mp_4(x)

        # classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logit = nn.Sigmoid()(self.dense(x))

        return logit

class CNN_SMALL(nn.Module):
    def __init__(self, num_class=15):
        super(CNN_SMALL, self).__init__()

        # init bn
        self.bn_init = nn.BatchNorm2d(1)

        # layer 1
        self.conv_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.mp_1 = nn.MaxPool2d((2, 2))

        # layer 2
        self.conv_2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.mp_2 = nn.MaxPool2d((2, 4))

        # layer 3
        self.conv_3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.mp_3 = nn.MaxPool2d((2, 2))

        # layer 4
        self.conv_4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(256)
        self.mp_4 = nn.MaxPool2d((2, 4))

        # layer 5
        self.conv_5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(512)
        self.mp_5 = nn.MaxPool2d((3, 5))

        # layer 6
        self.conv_6 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn_6 = nn.BatchNorm2d(256)
        self.mp_6 = nn.MaxPool2d((2, 4))

        # classifier
        self.dense = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)

        # init bn
        x = self.bn_init(x)

        # layer 1
        x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))

        # layer 2
        x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))

        # layer 3
        x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))

        # layer 4
        x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))

        # layer 5
        x = self.mp_5(nn.ELU()(self.bn_5(self.conv_5(x))))

        # layer 6
        x = self.mp_6(nn.ELU()(self.bn_6(self.conv_6(x))))

        # classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logit = nn.Sigmoid()(self.dense(x))

        return logit

class CNN_Plain(nn.Module):
    def __init__(self, num_class=15):
        super(CNN_Plain, self).__init__()

        # init bn
        self.bn_init = nn.BatchNorm2d(1)

        #block 1
        in_width = 1
        out_width = 64
        self.conv_1 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.bn_1 = nn.BatchNorm2d(out_width)
        self.conv_2 = nn.Conv2d(out_width, out_width, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_width)
        self.conv_3 = nn.Conv2d(out_width, out_width, 1, padding=0)
        self.bn_3 = nn.BatchNorm2d(out_width)

        self.sc_1 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.scb_1 = nn.BatchNorm2d(out_width)
        self.mp_1 = nn.MaxPool2d((2, 2))

        #block 2
        in_width = 64
        out_width = 64
        self.conv_4 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.bn_4 = nn.BatchNorm2d(out_width)
        self.conv_5 = nn.Conv2d(out_width, out_width, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(out_width)
        self.conv_6 = nn.Conv2d(out_width, out_width, 1, padding=0)
        self.bn_6 = nn.BatchNorm2d(out_width)

        self.mp_2 = nn.MaxPool2d((2, 4))

        #block 3
        in_width = 64
        out_width = 128
        self.conv_7 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.bn_7 = nn.BatchNorm2d(out_width)
        self.conv_8 = nn.Conv2d(out_width, out_width, 3, padding=1)
        self.bn_8 = nn.BatchNorm2d(out_width)
        self.conv_9 = nn.Conv2d(out_width, out_width, 1, padding=0)
        self.bn_9 = nn.BatchNorm2d(out_width)

        self.sc_3 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.scb_3 = nn.BatchNorm2d(out_width)
        self.mp_3 = nn.MaxPool2d((2, 2))

        #block 4
        in_width = 128
        out_width = 128
        self.conv_10 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.bn_10 = nn.BatchNorm2d(out_width)
        self.conv_11 = nn.Conv2d(out_width, out_width, 3, padding=1)
        self.bn_11 = nn.BatchNorm2d(out_width)
        self.conv_12 = nn.Conv2d(out_width, out_width, 1, padding=0)
        self.bn_12 = nn.BatchNorm2d(out_width)

        self.mp_4 = nn.MaxPool2d((2, 4))

        #block 5
        in_width = 128
        out_width = 256
        self.conv_13 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.bn_13 = nn.BatchNorm2d(out_width)
        self.conv_14 = nn.Conv2d(out_width, out_width, 3, padding=1)
        self.bn_14 = nn.BatchNorm2d(out_width)
        self.conv_15 = nn.Conv2d(out_width, out_width, 1, padding=0)
        self.bn_15 = nn.BatchNorm2d(out_width)

        self.sc_5 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.scb_5 = nn.BatchNorm2d(out_width)
        self.mp_5 = nn.MaxPool2d((3, 5))

        #block 6
        in_width = 256
        out_width = 256
        self.conv_16 = nn.Conv2d(in_width, out_width, 1, padding=0)
        self.bn_16 = nn.BatchNorm2d(out_width)
        self.conv_17 = nn.Conv2d(out_width, out_width, 3, padding=1)
        self.bn_17 = nn.BatchNorm2d(out_width)
        self.conv_18 = nn.Conv2d(out_width, out_width, 1, padding=0)
        self.bn_18 = nn.BatchNorm2d(out_width)

        self.mp_6 = nn.MaxPool2d((2, 4))

        #classifier
        self.dense = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)

        #init bn
        x = self.bn_init(x)

        #block 1
        out2 = self.sc_1(x)
        out2 = self.scb_1(out2)

        out1 = self.conv_1(x)
        out1 = self.bn_1(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_2(out1)
        out1 = self.bn_2(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_3(out1)
        out1 = self.bn_3(out1)

        x = nn.ELU()(out1 + out2)
        x = self.mp_1(x)

        #block 2
        out2 = x

        out1 = self.conv_4(x)
        out1 = self.bn_4(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_5(out1)
        out1 = self.bn_5(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_6(out1)
        out1 = self.bn_6(out1)

        x = nn.ELU()(out1 + out2)
        x = self.mp_2(x)

        #block 3
        out1 = self.conv_7(x)
        out1 = self.bn_7(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_8(out1)
        out1 = self.bn_8(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_9(out1)
        out1 = self.bn_9(out1)

        out2 = self.sc_3(x)
        out2 = self.scb_3(out2)

        x = nn.ELU()(out1 + out2)
        x = self.mp_3(x)

        #block 4
        out2 = x

        out1 = self.conv_10(x)
        out1 = self.bn_10(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_11(out1)
        out1 = self.bn_11(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_12(out1)
        out1 = self.bn_12(out1)

        x = nn.ELU()(out1 + out2)
        x = self.mp_4(x)

        #block 5
        out1 = self.conv_13(x)
        out1 = self.bn_13(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_14(out1)
        out1 = self.bn_14(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_15(out1)
        out1 = self.bn_15(out1)

        out2 = self.sc_5(x)
        out2 = self.scb_5(out2)

        x = nn.ELU()(out1 + out2)
        x = self.mp_5(x)

        #block 6
        out2 = x

        out1 = self.conv_16(x)
        out1 = self.bn_16(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_17(out1)
        out1 = self.bn_17(out1)
        out1 = nn.ELU()(out1)
        out1 = self.conv_18(out1)
        out1 = self.bn_18(out1)

        x = nn.ELU()(out1 + out2)
        x = self.mp_6(x)

        # classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logit = nn.Sigmoid()(self.dense(x))

        return logit

class CNN(nn.Module):
    def __init__(self, num_class=15):
        super(CNN, self).__init__()

        # init bn
        self.bn_init = nn.BatchNorm2d(1)

        # layer 1
        self.conv_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.mp_1 = nn.MaxPool2d((2, 4))

        # layer 2
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.mp_2 = nn.MaxPool2d((2, 4))

        # layer 3
        self.conv_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.mp_3 = nn.MaxPool2d((2, 4))

        # layer 4
        self.conv_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.mp_4 = nn.MaxPool2d((3, 5))

        # layer 5
        self.conv_5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)
        self.mp_5 = nn.MaxPool2d((4, 4))

        # classifier
        self.dense = nn.Linear(64, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)

        # init bn
        x = self.bn_init(x)

        # layer 1
        x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))

        # layer 2
        x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))

        # layer 3
        x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))

        # layer 4
        x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))

        # layer 5
        x = self.mp_5(nn.ELU()(self.bn_5(self.conv_5(x))))

        # classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logit = nn.Sigmoid()(self.dense(x))

        return logit

class FCN5(nn.Module):
    def __init__(self, num_class=56):
        super(FCN5, self).__init__()

        # init bn
        self.bn_init = nn.BatchNorm2d(1)

        # layer 1
        self.conv_1 = nn.Conv2d(1, 128, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(128)
        self.mp_1 = nn.MaxPool2d((2, 4))

        # layer 2
        self.conv_2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(256)
        self.mp_2 = nn.MaxPool2d((2, 4))

        # layer 3
        self.conv_3 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(512)
        self.mp_3 = nn.MaxPool2d((2, 4))

        # layer 4
        self.conv_4 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(1024)
        self.mp_4 = nn.MaxPool2d((3, 5))

        # layer 5
        self.conv_5 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(2048)
        self.mp_5 = nn.MaxPool2d((4, 4))

        # classifier
        self.dense = nn.Linear(2048, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)

        # init bn
        x = self.bn_init(x)

        # layer 1
        x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))

        # layer 2
        x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))

        # layer 3
        x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))

        # layer 4
        x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))

        # layer 5
        x = self.mp_5(nn.ELU()(self.bn_5(self.conv_5(x))))

        # classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logit = nn.Sigmoid()(self.dense(x))

        return logit

class AResNet(nn.Module):
    def __init__(self, num_class=15):
        super(AResNet, self).__init__()

        # init bn
        self.bn_init = nn.BatchNorm2d(1)

        #block 1
        self.conv_1 = nn.Conv2d(1, 64, 7, padding=3)
        self.bn_1 = nn.BatchNorm2d(64)

        self.mp_1 = nn.MaxPool2d((2, 4))

        #block 2
        self.conv_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(64)
        self.conv_4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.conv_5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)
        self.conv_6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_6 = nn.BatchNorm2d(64)
        self.conv_7 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_7 = nn.BatchNorm2d(64)

        self.mp_2 = nn.MaxPool2d((2, 4))

        #block 3
        self.conv_8 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_8 = nn.BatchNorm2d(128)
        self.conv_9 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_9 = nn.BatchNorm2d(128)
        self.conv_10 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_10 = nn.BatchNorm2d(128)
        self.conv_11 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_11 = nn.BatchNorm2d(128)
        self.conv_12 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_12 = nn.BatchNorm2d(128)
        self.conv_13 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_13 = nn.BatchNorm2d(128)

        self.mp_3 = nn.MaxPool2d((2, 4))

        #block 4
        self.conv_14 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_14 = nn.BatchNorm2d(256)
        self.conv_15 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_15 = nn.BatchNorm2d(256)
        self.conv_16 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_16 = nn.BatchNorm2d(256)
        self.conv_17 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_17 = nn.BatchNorm2d(256)
        self.conv_18 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_18 = nn.BatchNorm2d(256)
        self.conv_19 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_19 = nn.BatchNorm2d(256)

        self.mp_4 = nn.MaxPool2d((3, 5))

        #block 5
        self.conv_20 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn_20 = nn.BatchNorm2d(512)
        self.conv_21 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_21 = nn.BatchNorm2d(512)
        self.conv_22 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_22 = nn.BatchNorm2d(512)
        self.conv_23 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_23 = nn.BatchNorm2d(512)
        self.conv_24 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_24 = nn.BatchNorm2d(512)
        self.conv_25 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_25 = nn.BatchNorm2d(512)

        self.mp_5 = nn.MaxPool2d((4, 4))
        self.ap_5 = nn.AvgPool2d((4, 4))

        #classifier
        self.dense = nn.Linear(512, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)

        #init bn
        x = self.bn_init(x)

        #block 1
        x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))

        #block 2
        x = nn.ELU()(self.bn_2(self.conv_2(x)))
        x = nn.ELU()(self.bn_3(self.conv_3(x)))
        x = nn.ELU()(self.bn_4(self.conv_4(x)))
        x = nn.ELU()(self.bn_5(self.conv_5(x)))
        x = nn.ELU()(self.bn_6(self.conv_6(x)))
        x = nn.ELU()(self.bn_7(self.conv_7(x)))
        x = self.mp_2(x)

        #block 3
        x = nn.ELU()(self.bn_8(self.conv_8(x)))
        x = nn.ELU()(self.bn_9(self.conv_9(x)))
        x = nn.ELU()(self.bn_10(self.conv_10(x)))
        x = nn.ELU()(self.bn_11(self.conv_11(x)))
        x = nn.ELU()(self.bn_12(self.conv_12(x)))
        x = nn.ELU()(self.bn_13(self.conv_13(x)))
        x = self.mp_3(x)

        #block 4
        x = nn.ELU()(self.bn_14(self.conv_14(x)))
        x = nn.ELU()(self.bn_15(self.conv_15(x)))
        x = nn.ELU()(self.bn_16(self.conv_16(x)))
        x = nn.ELU()(self.bn_17(self.conv_17(x)))
        x = nn.ELU()(self.bn_18(self.conv_18(x)))
        x = nn.ELU()(self.bn_19(self.conv_19(x)))
        x = self.mp_4(x)

        #block 5
        x = nn.ELU()(self.bn_20(self.conv_20(x)))
        x = nn.ELU()(self.bn_21(self.conv_21(x)))
        x = nn.ELU()(self.bn_22(self.conv_22(x)))
        x = nn.ELU()(self.bn_23(self.conv_23(x)))
        x = nn.ELU()(self.bn_24(self.conv_24(x)))
        x = nn.ELU()(self.bn_25(self.conv_25(x)))
        #x = self.mp_5(x)
        x = self.ap_5(x)

        # classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logit = nn.Sigmoid()(self.dense(x))

        return logit

class AResNet_Residual(nn.Module):
    def __init__(self, num_class=15):
        super(AResNet_Residual, self).__init__()

        # init bn
        self.bn_init = nn.BatchNorm2d(1)

        #block 1
        self.conv_1 = nn.Conv2d(1, 64, 7, padding=3)
        self.bn_1 = nn.BatchNorm2d(64)

        self.mp_1 = nn.MaxPool2d((2, 4))

        #block 2
        self.conv_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(64)
        self.conv_4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.conv_5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)
        self.conv_6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_6 = nn.BatchNorm2d(64)
        self.conv_7 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_7 = nn.BatchNorm2d(64)

        self.mp_2 = nn.MaxPool2d((2, 4))

        #block 3
        self.conv_8 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_8 = nn.BatchNorm2d(128)
        self.conv_9 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_9 = nn.BatchNorm2d(128)
        self.conv_10 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_10 = nn.BatchNorm2d(128)
        self.conv_11 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_11 = nn.BatchNorm2d(128)
        self.conv_12 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_12 = nn.BatchNorm2d(128)
        self.conv_13 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_13 = nn.BatchNorm2d(128)

        self.mp_3 = nn.MaxPool2d((2, 4))

        #block 4
        self.conv_14 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_14 = nn.BatchNorm2d(256)
        self.conv_15 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_15 = nn.BatchNorm2d(256)
        self.conv_16 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_16 = nn.BatchNorm2d(256)
        self.conv_17 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_17 = nn.BatchNorm2d(256)
        self.conv_18 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_18 = nn.BatchNorm2d(256)
        self.conv_19 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_19 = nn.BatchNorm2d(256)

        self.mp_4 = nn.MaxPool2d((3, 5))

        #block 5
        self.conv_20 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn_20 = nn.BatchNorm2d(512)
        self.conv_21 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_21 = nn.BatchNorm2d(512)
        self.conv_22 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_22 = nn.BatchNorm2d(512)
        self.conv_23 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_23 = nn.BatchNorm2d(512)
        self.conv_24 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_24 = nn.BatchNorm2d(512)
        self.conv_25 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_25 = nn.BatchNorm2d(512)

        self.mp_5 = nn.MaxPool2d((4, 4))
        self.ap_5 = nn.AvgPool2d((4, 4))

        #classifier
        self.dense = nn.Linear(512, num_class)

    def forward(self, x):
        x = x.unsqueeze(1)

        #init bn
        x = self.bn_init(x)

        #block 1
        x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))

        #block 2
        out = nn.ELU()(self.bn_2(self.conv_2(x)))
        x = nn.ELU()(self.bn_3(self.conv_3(out)))
        identity = x
        out = nn.ELU()(self.bn_4(self.conv_4(x)))
        x = nn.ELU()(self.bn_5(self.conv_5(out)) + identity)
        identity = x
        out = nn.ELU()(self.bn_6(self.conv_6(x)))
        x = nn.ELU()(self.bn_7(self.conv_7(out)) + identity)
        x = self.mp_2(x)

        #block 3
        out = nn.ELU()(self.bn_8(self.conv_8(x)))
        x = nn.ELU()(self.bn_9(self.conv_9(out)))
        identity = x
        out = nn.ELU()(self.bn_10(self.conv_10(x)))
        x = nn.ELU()(self.bn_11(self.conv_11(out)) + identity)
        identity = x
        out = nn.ELU()(self.bn_12(self.conv_12(x)))
        x = nn.ELU()(self.bn_13(self.conv_13(out)) + identity)
        x = self.mp_3(x)

        #block 4
        out = nn.ELU()(self.bn_14(self.conv_14(x)))
        x = nn.ELU()(self.bn_15(self.conv_15(out)))
        identity = x
        out = nn.ELU()(self.bn_16(self.conv_16(x)))
        x = nn.ELU()(self.bn_17(self.conv_17(out)) + identity)
        identity = x
        out = nn.ELU()(self.bn_18(self.conv_18(x)))
        x = nn.ELU()(self.bn_19(self.conv_19(out)) + identity)
        x = self.mp_4(x)

        #block 5
        out = nn.ELU()(self.bn_20(self.conv_20(x)))
        x = nn.ELU()(self.bn_21(self.conv_21(out)))
        identity = x
        out = nn.ELU()(self.bn_22(self.conv_22(x)))
        x = nn.ELU()(self.bn_23(self.conv_23(out)) + identity)
        identity = x
        out = nn.ELU()(self.bn_24(self.conv_24(x)))
        x = nn.ELU()(self.bn_25(self.conv_25(out)) + identity)
        x = self.ap_5(x)

        # classifier
        x = x.view(x.size(0), -1)
        logit = nn.Sigmoid()(self.dense(x))

        return logit