import numpy as np
import scipy.misc
import scipy.signal
import matplotlib.pyplot as plt

import pyaudio
import Queue
import threading
import time
import zlib

def PSNR(i1, i2):
    i1 = i1.astype(np.int64)
    i2 = i2.astype(np.int64)
    mse = np.mean((i1 - i2)**2)
    return 10 * np.log10(255**2 / mse)

def PSNR_official(im_truth, im_test, maxval=255.):
    mse = np.linalg.norm(im_truth.astype(np.float64) - im_test.astype(np.float64))**2 / np.prod(np.shape(im_truth))
    return 10 * np.log10(maxval**2 / mse)

def avgColorError(img):
    avg = np.mean(img).astype(np.uint8)
    return avg, np.sum((img - avg.astype(float)) ** 2)
def bestCut(img, granularity = 1):
    h = img.shape[0]
    w = img.shape[1]

    best_err = 1e99
    best_err_ind = ()
    best_cut = 0
    best_cut_is_vert = False
    for i in np.r_[1:h:granularity]:
        col1, err1 = avgColorError(img[:i,:])
        col2, err2 = avgColorError(img[i:,:])
        tot_err = err1 + err2
        if tot_err < best_err:
            best_err_ind = (err1, err2)
            best_err = tot_err
            best_cut = i

    for i in np.r_[1:w:granularity]:
        col1, err1 = avgColorError(img[:,:i])
        col2, err2 = avgColorError(img[:,i:])
        tot_err = err1 + err2
        if tot_err < best_err:
            best_err_ind = (err1, err2)
            best_err = tot_err
            best_cut = i
            best_cut_is_vert = True

    if len(best_err_ind) == 0:
        import pdb; pdb.set_trace()
    return best_cut, best_cut_is_vert, best_err_ind

def encodeImage(img, x, y, stop = 1e3):
    if img.shape[0] == 1 and img.shape[1] == 1:
        return [((x, y), (1, 1), img[0][0])]
    cut, is_vert, err = bestCut(img, granularity = max(1, int(img.shape[0] / 50)))
    chunks = []
    if is_vert:
        chunk1 = img[:,:cut]
        chunk2 = img[:,cut:]
        x2 = x + cut
        y2 = y
    else:
        chunk1 = img[:cut,:]
        chunk2 = img[cut:,:]
        x2 = x
        y2 = y + cut
    if err[0] < stop:
        chunks += [((x, y), (chunk1.shape[0], chunk1.shape[1]), avgColorError(chunk1)[0])]
    else:
        chunks += encodeImage(chunk1, x, y, stop)
    if err[1] < stop:
        chunks += [((x2, y2), (chunk2.shape[0], chunk2.shape[1]), avgColorError(chunk2)[0])]
    else:
        chunks += encodeImage(chunk2, x2, y2, stop)

    return chunks

def decodeImage(chunks, h, w, init = 0):
    recon = init * np.ones((h, w))
    for coor, size, val in chunks:
        recon[coor[1]:coor[1]+size[0],coor[0]:coor[0]+size[1]] = val
    return recon

## CHUNK FORMAT:
## 11 bits | 11 bits | 11 bits | 11 bits | 8 bits
## x pos   | y pos   | x size  | y size  | color
def packChunkToBits(chunks):
    bits = ""
    for chunk in chunks:
        curChunkBits = (chunk[0][0] << 41) + (chunk[0][1] << 30) + (chunk[1][0] << 19) + (chunk[1][1] << 8) + chunk[2]
        bits += str(bin(curChunkBits))[2:].zfill(52)
    return bits

def bitsToChunk(bits):
    bitStr = "".join(bits)
    col = int(bitStr[44:52], 2)
    ySiz = int(bitStr[33:44], 2)
    xSiz = int(bitStr[22:33], 2)
    yPos = int(bitStr[11:22], 2)
    xPos = int(bitStr[0:11], 2)
    return ((xPos, yPos), (xSiz, ySiz), col)

# IMAGE format
# 11 bits: x, 11 bits: y
# variable: chunks (multiple of 52)
class CartoonEncoder(object):
    def __init__(self, cutoff_c = 1e4, chunk_limit = 3500):
        self.cutoff_c = cutoff_c
        self.chunk_limit = chunk_limit

    def imgToChunks(self, img):
        dim = img.shape
        chunks = encodeImage(img, 0, 0, stop=(dim[0] * dim[1] / self.cutoff_c)**(1/0.3))
        nz_chunks = chunks
        cutoff = 255
        while len(nz_chunks) > self.chunk_limit:
            nz_chunks = [chunk for chunk in chunks if chunk[2] < cutoff]
            cutoff -= 1

        return nz_chunks

    def imgToBits(self, img):
        x = str(bin(img.shape[0]))[2:].zfill(11)
        y = str(bin(img.shape[1]))[2:].zfill(11)
        chunks = self.imgToChunks(img)
        bits = x + y + packChunkToBits(chunks)
        bits = bits + '0' * ((8 - len(bits)%8)%8)
        return bits

    def bitsToImg(self, bits):
        bits = np.array(list(bits), dtype=str)
        dim = (int("".join(bits[0:11]), 2), int("".join(bits[11:22]), 2))
        bits = bits[22:]
        bits = bits[:-(bits.shape[0]%52)]
        received_chunks = [bitsToChunk(b) for b in np.reshape(bits, (-1, 52))]
        grayscale = decodeImage(received_chunks, dim[0], dim[1], init = 255)
        decoded = np.zeros((grayscale.shape[0], grayscale.shape[1], 3))
        decoded[:,:,0] = grayscale
        decoded[:,:,1] = grayscale
        decoded[:,:,2] = grayscale
        plt.figure(figsize=(12,12))
        plt.imshow(decoded.astype(np.uint8))
        return decoded.astype(np.uint8)
