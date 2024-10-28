import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernelSize=3):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernelSize, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernelSize, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)))


class Recurrent_block(nn.Module):
    def __init__(self,ch_out):
        super(Recurrent_block,self).__init__()
        self.ch_out = ch_out
        self.conv = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True)
        self.bn = nn.BatchNorm2d(ch_out)

    def forward(self,x):
        x1 = F.relu(self.bn(self.conv(x)))
        return F.relu(self.bn(self.conv(x + x1)))


class Attention_block(nn.Module):
    def __init__(self, query_in, key_in, out_channels):
        
        super(Attention_block,self).__init__()

        self.q_conv = nn.Conv2d(query_in, out_channels, kernel_size=1,stride=1,padding=0,bias=True)
        self.q_bn = nn.BatchNorm2d(out_channels)

        self.k_conv = nn.Conv2d(key_in, out_channels, kernel_size=1,stride=1,padding=0,bias=True)
        self.k_bn = nn.BatchNorm2d(out_channels)

        self.v_conv = nn.Conv2d(out_channels, 1, kernel_size=1,stride=1,padding=0,bias=True)
        self.v_bn = nn.BatchNorm2d(1)
        
    def forward(self, q, k):
        q1 = self.q_bn(self.q_conv(q))
        k1 = self.k_bn(self.k_conv(k))

        value = torch.sigmoid(self.v_bn(self.v_conv(F.relu(q1 + k1))))

        return k * value


class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(RRCNN_block, self).__init__()
        self.rec_block1 = Recurrent_block(ch_out)
        self.rec_block2 = Recurrent_block(ch_out)
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        return x+self.rec_block2(self.rec_block1(x))


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filterCount=16, dropoutRate=0.1):
        super(UNet, self).__init__()
        self.dropoutRate = dropoutRate
        self.epochs_done = 0
        self.best_acc = 0

        self.conv1 = ConvolutionBlock(in_channels, filterCount)
        self.conv2 = ConvolutionBlock(filterCount, filterCount*2)
        self.conv3 = ConvolutionBlock(filterCount*2, filterCount*4)
        self.conv4 = ConvolutionBlock(filterCount*4, filterCount*8)
        self.conv5 = ConvolutionBlock(filterCount*8, filterCount*16)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropoutRate)

        self.transpose_upconv1 = nn.ConvTranspose2d(filterCount*16, filterCount*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = ConvolutionBlock(filterCount * 16, filterCount*8)

        self.transpose_upconv2 = nn.ConvTranspose2d(filterCount*8, filterCount*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = ConvolutionBlock(filterCount * 8, filterCount*4)

        self.transpose_upconv3 = nn.ConvTranspose2d(filterCount * 4, filterCount * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = ConvolutionBlock(filterCount * 4, filterCount*2)

        self.transpose_upconv4 = nn.ConvTranspose2d(filterCount*2, filterCount, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv9 = ConvolutionBlock(filterCount*2, filterCount)

        self.out_conv = nn.Conv2d(filterCount, out_channels, kernel_size=1)

    def forward(self, x):
        encode1 = self.conv1(x)
        encode2 = self.conv2(self.dropout(self.pool(encode1)))
        encode3 = self.conv3(self.dropout(self.pool(encode2)))
        encode4 = self.conv4(self.dropout(self.pool(encode3)))
        encode5 = self.conv5(self.dropout(self.pool(encode4)))

        decode1 = self.transpose_upconv1(encode5)
        encode6 = self.conv6(self.dropout(torch.cat([decode1, encode4], dim=1)))

        decode2 = self.transpose_upconv2(encode6)
        encode7 = self.conv7(self.dropout(torch.cat([decode2, encode3], dim=1)))

        decode3 = self.transpose_upconv3(encode7)
        encode8 = self.conv8(self.dropout(torch.cat([decode3, encode2], dim=1)))

        decode4 = self.transpose_upconv4(encode8)
        encode9 = self.conv9(self.dropout(torch.cat([decode4, encode1], dim=1)))

        return torch.sigmoid(self.out_conv(encode9))


class RUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filterCount=16, dropoutRate=0.1):
        super(RUNet, self).__init__()
        self.dropoutRate = dropoutRate
        self.epochs_done = 0
        self.best_acc = 0

        self.rec_conv1 = RRCNN_block(in_channels, filterCount)
        self.rec_conv2 = RRCNN_block(filterCount, filterCount*2)
        self.rec_conv3 = RRCNN_block(filterCount*2, filterCount*4)
        self.rec_conv4 = RRCNN_block(filterCount*4, filterCount*8)
        self.rec_conv5 = RRCNN_block(filterCount*8, filterCount*16)

        self.dropout = nn.Dropout(dropoutRate)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.transpose_upconv1 = nn.ConvTranspose2d(filterCount*16, filterCount*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.rec_conv6 = RRCNN_block(filterCount * 16, filterCount*8)

        self.transpose_upconv2 = nn.ConvTranspose2d(filterCount*8, filterCount*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.rec_conv7 = RRCNN_block(filterCount * 8, filterCount*4)

        self.transpose_upconv3 = nn.ConvTranspose2d(filterCount * 4, filterCount * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.rec_conv8 = RRCNN_block(filterCount * 4, filterCount*2)

        self.transpose_upconv4 = nn.ConvTranspose2d(filterCount*2, filterCount, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.rec_conv9 = RRCNN_block(filterCount*2, filterCount)

        self.out_conv = nn.Conv2d(filterCount, out_channels, kernel_size=1)

    def forward(self, x):
        encode1 = self.rec_conv1(x)
        encode2 = self.rec_conv2(self.dropout(self.avgpool(encode1)))
        encode3 = self.rec_conv3(self.dropout(self.avgpool(encode2)))
        encode4 = self.rec_conv4(self.dropout(self.avgpool(encode3)))
        encode5 = self.rec_conv5(self.dropout(self.avgpool(encode4)))

        decode1 = self.transpose_upconv1(encode5)
        encode6 = self.rec_conv6(self.dropout(torch.cat([decode1, encode4], dim=1)))

        decode2 = self.transpose_upconv2(encode6)
        encode7 = self.rec_conv7(self.dropout(torch.cat([decode2, encode3], dim=1)))

        decode3 = self.transpose_upconv3(encode7)
        encode8 = self.rec_conv8(self.dropout(torch.cat([decode3, encode2], dim=1)))

        decode4 = self.transpose_upconv4(encode8)
        encode9 = self.rec_conv9(self.dropout(torch.cat([decode4, encode1], dim=1)))

        return torch.sigmoid(self.out_conv(encode9))


class AttUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, CellNum=16, dropout=0.1):
        super(AttUNet,self).__init__()

        self.dropout = dropout
        self.best_acc = 0
        self.Pool = nn.AvgPool2d(2)
        self.dropout = nn.Dropout(self.dropout)

        self.conv1 = ConvolutionBlock(in_channels, CellNum)
        self.conv2 = ConvolutionBlock(CellNum, CellNum*2)
        self.conv3 = ConvolutionBlock(CellNum*2, CellNum*4)
        self.conv4 = ConvolutionBlock(CellNum*4, CellNum*8)
        self.conv5 = ConvolutionBlock(CellNum*8, CellNum*16)

        self.transpose_upconv1 = nn.ConvTranspose2d(CellNum*16, CellNum*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_block1 = Attention_block(CellNum*8, CellNum*8, CellNum*4)
        self.conv6 = ConvolutionBlock(CellNum*16, CellNum*8)

        self.transpose_upconv2 = nn.ConvTranspose2d(CellNum*8, CellNum*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_block2 = Attention_block(CellNum*4, CellNum*4, CellNum*2)
        self.conv7 = ConvolutionBlock(CellNum*8, CellNum*4)
        
        self.transpose_upconv3 = nn.ConvTranspose2d(CellNum*4, CellNum*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_block3 = Attention_block(CellNum*2, CellNum*2, CellNum)
        self.conv8 = ConvolutionBlock(CellNum*4, CellNum*2)
        
        self.transpose_upconv4 = nn.ConvTranspose2d(CellNum*2, CellNum, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_block4 = Attention_block(CellNum, CellNum, CellNum//2)
        self.conv9 = ConvolutionBlock(CellNum*2, CellNum)

        self.out_conv = nn.Conv2d(CellNum, out_channels, kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.conv1(x)

        x2 = self.conv2(self.dropout(self.Pool(x1)))
        
        x3 = self.conv3(self.dropout(self.Pool(x2)))

        x4 = self.conv4(self.dropout(self.Pool(x3)))

        x5 = self.conv5(self.dropout(self.Pool(x4)))

        # decoding + concat path
        d5 = self.transpose_upconv1(x5)
        d5 = self.conv6(self.dropout(torch.cat((self.attention_block1(d5, x4), d5), dim=1)))
        
        d4 = self.transpose_upconv2(d5)
        d4 = self.conv7(self.dropout(torch.cat((self.attention_block2(d4, x3), d4), dim=1)))

        d3 = self.transpose_upconv3(d4)
        d3 = self.conv8(self.dropout(torch.cat((self.attention_block3(d3,x2), d3), dim=1)))

        d2 = self.transpose_upconv4(d3)
        d2 = self.conv9(self.dropout(torch.cat((self.attention_block4(d2,x1), d2), dim=1)))

        return torch.sigmoid(self.out_conv(d2))


class RAttUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1, CellNum=16, dropout=0.1):
        super(RAttUNet, self).__init__()
        
        self.dropout = dropout
        self.best_acc = 0
        self.Pool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.dropout = nn.Dropout(dropout)

        self.rec_conv1 = RRCNN_block(img_ch, CellNum)
        self.rec_conv2 = RRCNN_block(CellNum, CellNum*2)
        self.rec_conv3 = RRCNN_block(CellNum*2, CellNum*4)
        self.rec_conv4 = RRCNN_block(CellNum*4, CellNum*8)
        self.rec_conv5 = RRCNN_block(CellNum*8, CellNum*16)
        
        self.transpose_upconv1 = nn.ConvTranspose2d(CellNum*16, CellNum*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_block1 = Attention_block(CellNum*8, CellNum*8, CellNum*4)
        self.rec_conv6 = RRCNN_block(CellNum*16, CellNum*8)
        
        self.transpose_upconv2 = nn.ConvTranspose2d(CellNum*8, CellNum*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_block2 = Attention_block(CellNum*4, CellNum*4, CellNum*2)
        self.rec_conv7 = RRCNN_block(CellNum*8, CellNum*4)
        
        self.transpose_upconv3 = nn.ConvTranspose2d(CellNum*4, CellNum*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_block3 = Attention_block(CellNum*2, CellNum*2, CellNum)
        self.rec_conv8 = RRCNN_block(CellNum*4, CellNum*2)
        
        self.transpose_upconv4 = nn.ConvTranspose2d(CellNum*2, CellNum, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention_block4 = Attention_block(CellNum, CellNum, CellNum//2)
        self.rec_conv9 = RRCNN_block(CellNum*2, CellNum)

        self.out_conv = nn.Conv2d(CellNum, output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.rec_conv1(x)
        
        x2 = self.rec_conv2(self.dropout(self.Pool(x1)))
        
        x3 = self.rec_conv3(self.dropout(self.Pool(x2)))

        x4 = self.rec_conv4(self.dropout(self.Pool(x3)))

        x5 = self.rec_conv5(self.dropout(self.Pool(x4)))

        # decoding + concat path
        d5 = self.transpose_upconv1(x5)
        d5 = self.rec_conv6(self.dropout(torch.cat((self.attention_block1(d5, x4), d5), dim = 1)))
        
        d4 = self.transpose_upconv2(d5)
        d4 = self.rec_conv7(self.dropout(torch.cat((self.attention_block2(d4, x3), d4), dim = 1)))

        d3 = self.transpose_upconv3(d4)
        d3 = self.rec_conv8(self.dropout(torch.cat((self.attention_block3(d3, x2), d3), dim = 1)))

        d2 = self.transpose_upconv4(d3)
        d2 = self.rec_conv9(self.dropout(torch.cat((self.attention_block4(d2, x1), d2), dim = 1)))

        return torch.sigmoid(self.out_conv(d2))

