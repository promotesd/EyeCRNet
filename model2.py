import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv
import ellipse
import copy

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Custom Unet with SE blocks and Attention blocks
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, use_se=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_se:
            layers.append(SEBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))
        dec4 = self.upconv4(bottleneck)
        enc4 = self.att4(dec4, enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        enc3 = self.att3(dec3, enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        enc2 = self.att2(dec2, enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        out = self.final_conv(dec1)
        return out

# PupilNet class using the custom Unet for pupil detection
class PupilNet(nn.Module):
    def __init__(self):
        super(PupilNet, self).__init__()
        self.backbone = Unet(in_channels=1, out_channels=1)  # Adjust channel numbers as needed

    def forward(self, inputs):
        return self.backbone(inputs)

    def predict(self, inputs, sfac=None, offsets=None):
        with torch.no_grad():
            pred = self.forward(inputs)
            pred = nn.Sigmoid()(pred).detach().cpu().squeeze()
            if pred.ndim == 2:
                pred = pred.unsqueeze(0)
            mask = pred > 0.99
            pupils = []
            for i, m in enumerate(mask):
                pup = cv.detect_pupil_from_thresholded(m.numpy().astype('uint8')*255, symmetry_tresh=0.3, kernel=cv.kernel_pup2)
                pupil = {
                    'centroid': (pup['ellipse'][0]),
                    'axis_major_radius': pup['ellipse'][1][0]/2,
                    'axis_minor_radius': pup['ellipse'][1][1]/2,
                    'orientation': pup['ellipse'][2]
                }
                max_rad = max(pupil['axis_major_radius'], pupil['axis_minor_radius'])
                pupil['too_close_edge'] = pupil['centroid'][0] < max_rad or pupil['centroid'][0] > m.shape[1] or pupil['centroid'][1] < max_rad or pupil['centroid'][1] > m.shape[0]
                if (not np.isnan(pupil['centroid'][0])) and (sfac is not None):
                    el = ellipse.my_ellipse((*(pupil['centroid']), pupil['axis_major_radius'], pupil['axis_minor_radius'], pupil['orientation']/180*np.pi))
                    tform = ellipse.scale_2d(sfac, sfac)
                    nelpar = el.transform(tform)[0][:-1]
                    pupil['oripupil'] = copy.deepcopy(pupil)
                    pupil['centroid'] = (nelpar[0], nelpar[1])
                    pupil['axis_major_radius'] = nelpar[2]
                    pupil['axis_minor_radius'] = nelpar[3]
                    pupil['orientation'] = nelpar[4]/np.pi*180
                if (not np.isnan(pupil['centroid'][0])) and (offsets[i] is not None):
                    pupil['oripupil'] = copy.deepcopy(pupil)
                    pupil['centroid'] = tuple(x+y for x, y in zip(pupil['centroid'], offsets[i].numpy().flatten()))
                pupil["mask"] = m
                pupil["pmap"] = pred[i]
                pupils.append(pupil)
            return pupils


