from encoder import EncoderBase
import numpy as np
import util, data_transport
import scipy.interpolate
import matplotlib.pyplot as plt

# format
# 11 bits WIDTH
# 11 bits HEIGHT
# 4 bits DECIMATE
# 3 bits B
class ColorDecimateEncoder(EncoderBase):
    def __init__(self, binarize = False, b_thresh = 160):
        self.binarize = binarize
        self.b_thresh = b_thresh

    def imgToBits(self, img, downsample = 1, b = 1):
        img_filt = np.zeros(img.shape, dtype=float)
        img_filt[:,:,:] = img[:,:,:]

        img_up = np.zeros(img.shape, dtype=float)
        img_send2 = img_filt[::downsample,::downsample,:]
        img_send = np.round(img_send2 / (255. / float(2**b-1))).astype(int)
        if self.binarize:
            img_send = np.where(img_send2 > self.b_thresh, 1, 0).astype(int)
            b = 1

        img_str = "".join([str(bin(c))[2:].zfill(b) for c in np.reshape(img_send[:,:,0], (-1))]) + \
                    "".join([str(bin(c))[2:].zfill(b) for c in np.reshape(img_send[:,:,1], (-1))]) + \
                    "".join([str(bin(c))[2:].zfill(b) for c in np.reshape(img_send[:,:,2], (-1))])
        img_str = str(bin(img.shape[0]))[2:].zfill(11) + str(bin(img.shape[1]))[2:].zfill(11) + \
                    str(bin(downsample))[2:].zfill(4) + str(bin(b-1))[2:].zfill(3) + \
                    img_str

        return img_str + '0' * ((8-len(img_str)%8)%8)

    def bitsToImg(self, bits, show=False, filter=False):
        img_w = int(bits[0:11], 2)
        img_h = int(bits[11:22], 2)
        img = np.zeros((img_w, img_h, 3)) #DUMMY
        img_up = np.zeros((img_w, img_h, 3))
        downsample = int(bits[22:26], 2)
        ds_w = img_w / downsample + (1 if img_w % downsample else 0)
        ds_h = img_h / downsample + (1 if img_h % downsample else 0)
        b = int(bits[26:29], 2) + 1
        img_bits_R = bits[29 : 29 + b * ds_w * ds_h]
        img_bits_G = bits[29 + b * ds_w * ds_h : 29 + 2 * b * ds_w * ds_h]
        img_bits_B = bits[29 + 2 * b * ds_w * ds_h : 29 + 3 * b * ds_w * ds_h]

        img_recv_R = np.array([int(img_bits_R[b*i : b*i+b], 2) for i in range(ds_w * ds_h)])
        img_recv_R = ((255. / float(2**b - 1))) * np.reshape(img_recv_R, (ds_w, ds_h))
        img_recv_G = np.array([int(img_bits_G[b*i : b*i+b], 2) for i in range(ds_w * ds_h)])
        img_recv_G = ((255. / float(2**b - 1))) * np.reshape(img_recv_G, (ds_w, ds_h))
        img_recv_B = np.array([int(img_bits_B[b*i : b*i+b], 2) for i in range(ds_w * ds_h)])
        img_recv_B = ((255. / float(2**b - 1))) * np.reshape(img_recv_B, (ds_w, ds_h))

        img_up[::downsample,::downsample,0] = img_recv_R
        img_up[::downsample,::downsample,1] = img_recv_G
        img_up[::downsample,::downsample,2] = img_recv_B

        img_up_filt = np.zeros(img_up.shape, dtype=float)

        inter_method = 'linear'
        interR = scipy.interpolate.interp2d(np.r_[0:img.shape[1]:downsample], np.r_[0:img.shape[0]:downsample], img_recv_R, kind=inter_method)
        interG = scipy.interpolate.interp2d(np.r_[0:img.shape[1]:downsample], np.r_[0:img.shape[0]:downsample], img_recv_G, kind=inter_method)
        interB = scipy.interpolate.interp2d(np.r_[0:img.shape[1]:downsample], np.r_[0:img.shape[0]:downsample], img_recv_B, kind=inter_method)

        img_up_filt[:,:,0] = interR(np.r_[:img.shape[1]], np.r_[:img.shape[0]])
        img_up_filt[:,:,1] = interG(np.r_[:img.shape[1]], np.r_[:img.shape[0]])
        img_up_filt[:,:,2] = interB(np.r_[:img.shape[1]], np.r_[:img.shape[0]])

        recon = img_up_filt
        recon = np.where(recon > 255, 255, recon)
        recon = np.where(recon < 0, 0, recon)

        if show:
            plt.figure(figsize=(16,16))
            plt.imshow(recon.astype(np.uint8))

        return recon.astype(np.uint8)
