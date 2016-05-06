from encoder import EncoderBase
import numpy as np
import scipy.misc
import scipy.signal
from scipy.fftpack import dct, idct
import PIL.Image
import util, data_transport
import jpeg_huffman
import matplotlib.pyplot as plt

# format
# 11 bits ROW
# 11 bits COL 
# 7  bits quality factor q 
# ZRL 

# Quantization matrices for YUV 
Ql = np.array([[16, 11, 10, 16, 24, 40, 51, 61],     \
               [12, 12, 14, 19, 26, 58, 60, 55],     \
               [14, 13, 16, 24, 40, 57, 69 ,56],     \
               [14, 17, 22, 29, 51, 87, 80, 62],     \
               [18, 22, 37, 56, 68, 109, 103, 77],   \
               [24, 35, 55, 64, 81, 104, 113, 92],   \
               [49, 64, 78, 87, 103, 121, 120, 101], \
               [72, 92, 95, 98, 112, 100, 103, 99]])
    
Qc = np.array([[17, 18, 24, 47, 99, 99, 99, 99], \
               [18, 21, 26, 66, 99, 99, 99, 99], \
               [24, 26, 56, 99, 99, 99, 99, 99], \
               [47, 66, 99, 99, 99, 99, 99, 99], \
               [99, 99, 99, 99, 99, 99, 99, 99], \
               [99, 99, 99, 99, 99, 99, 99, 99], \
               [99, 99, 99, 99, 99, 99, 99, 99], \
               [99, 99, 99, 99, 99, 99, 99, 99]])

# imgToBits: splits images into 8x8 tiles (one color channel)
def blockSplit(image, n=8): 
    row, col = image.shape
    row_pad, col_pad = 0, 0
    if row%n != 0: row_pad = (int(row/n)+1)*n-row
    if col%n != 0: col_pad = (int(col/n)+1)*n-col
    image_pad, columns = np.copy(image), image[:, col-1]
    for _ in range(col_pad): 
        image_pad = np.column_stack((image_pad, columns))
    rows = image_pad[row-1, :]
    for _ in range(row_pad):
        image_pad = np.vstack((image_pad, rows))
    tiles = []
    r, c = image_pad.shape
    for i in np.r_[0:r:n]: 
        for j in np.r_[0:c:n]: 
            tiles.append(image_pad[i:i+n, j:j+n])
    return tiles

# imgToBits: quantize each tile specified by y, u, v
def quantizeTile(tile, color, q):
    alpha = 50.0/q if q<=50 and q>=1 else 2-q/50.0
    if color == 'y': return np.rint(tile/(alpha*Ql)).astype(int)
    else: return np.rint(tile/(alpha*Qc)).astype(int)

# imgToBits: DCT and then quantize (one color channel)
def dctQuantize(tiles, color, q=20): 
    quantized_tiles = []
    for tile in tiles: 
        dct_tile = dct(dct(tile.T,  norm='ortho').T, norm='ortho')
        quantized_tiles.append(quantizeTile(dct_tile, color, q))
    return quantized_tiles
# ZRL: zigzags through tile particular jpeg rle order
def zigZagTile(tile):
    coord = [(0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),\
           (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),\
           (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),\
           (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),\
           (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),\
           (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),\
           (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),\
           (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)]
    data = []
    for i in range(64):
        data.append(tile[coord[i][0], coord[i][1]])
    return np.array(data)

# imgToBits: encodes one tile in zero rle
def rleHelper(walkedValues, prevDC):
    num_zeros, rle, dc = 0, [], walkedValues[0] - prevDC
    rle.append((0, dc))
    for i in range(1, len(walkedValues)):
        if walkedValues[i] == 0: 
            if num_zeros < 15: num_zeros +=1
            else: 
                rle.append((num_zeros, 0))
                num_zeros = 0
        else: 
            rle.append((num_zeros, walkedValues[i]))
            num_zeros = 0
    if num_zeros < 15: 
        rle.append((0,0))
    idx = 1
    for i in range(0, len(rle)): 
        cur_idx = len(rle) - i - 1
        if rle[cur_idx][1] != 0: 
            idx = cur_idx
            break
    rle = rle[:idx+1]
    rle.append((0,0))
    return rle, walkedValues[0] 
         
# imgToBits: encodes tile coefficients in ZRL 
def rleTiles(tiles): 
    rle_lst, prev_dc = [], 0
    for tile in tiles:
        rle, prev_dc = rleHelper(zigZagTile(tile), prev_dc)
        rle_lst.append(rle)
    return rle_lst

# bitsToImg: reformat from zigzag pattern to standard coordinate format
def unZigzagTile(zigzag_tile): 
    coord = [(0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),\
           (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),\
           (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),\
           (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),\
           (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),\
           (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),\
           (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),\
           (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)]
    block = np.zeros((8,8), dtype=int)
    for i in range(64): 
        block[coord[i][0], coord[i][1]] = zigzag_tile[i]
    return block

# bitsToImg: extract quantized values and previous dc value
def decodeFromZRL(tile, prev_dc): 
    vals = []
    vals.append(tile[0][1] + prev_dc)
    for i in range(1, len(tile)): 
        num_zeros, x = tile[i][0], tile[i][1]
        if num_zeros == 0 and x == 0:
            vals_len = len(vals)
            for j in range(64-vals_len): 
                vals.append(0)
            break
        else: 
            for j in range(num_zeros):
                vals.append(0)
            vals.append(x)
    return vals, tile[0][1] + prev_dc

# bitsToImg: extract yuv quantized values
def decodeFromZRLTiles(zrl_tiles): 
    decoded_tiles = []
    y_prevDC, u_prevDC, v_prevDC = 0, 0, 0
    for tile in zrl_tiles: 
        y_tile, u_tile, v_tile = tile[0], tile[1], tile[2]
        y_vals, y_prevDC = decodeFromZRL(y_tile, y_prevDC)
        u_vals, u_prevDC = decodeFromZRL(u_tile, u_prevDC)
        v_vals, v_prevDC = decodeFromZRL(v_tile, v_prevDC)
        decoded_tiles.append([unZigzagTile(y_vals), unZigzagTile(u_vals), unZigzagTile(v_vals)])        
    return decoded_tiles

# bitsToImg: unquantize each tile given, based on y, u, v
def unquantizeTile(tile, color, q):
    alpha = 50.0/q if q<=50 and q>=1 else 2-q/50.0
    if color == 'y': return tile*(alpha*Ql)
    else: return tile*(alpha*Qc)

# bitsToImg: combine tiles into (padded) image
def combineTiles(tiles, image_shape, n=8):
    row, col = image_shape
    image_rows = []
    for i in range(row): 
        rows = tiles[col*i]
        for j in range(1, col): 
            rows = np.column_stack((rows, tiles[col*i+j]))
        image_rows.append(rows)
    combined_image = image_rows[0]
    for i in range(1, len(image_rows)): 
        combined_image = np.vstack((combined_image, image_rows[i]))
    return combined_image

# bitsToImg: reconstruct from yuv tiles, outputs rgb image
def reconstruct(yuv_tiles, original_shape, q=20): 
    recon = []
    row, col = int(original_shape[0]/8), int(original_shape[1]/8)
    if original_shape[0]%8 != 0: row+= 1
    if original_shape[1]%8 != 0: col+= 1
    
    for i in range(len(yuv_tiles)):
        if i==0: color = 'y'
        if i==1: color = 'u'
        if i==2: color = 'v'
        color_tiles = []
        for tile in yuv_tiles[i]:
            unquantized_tile = unquantizeTile(tile, color, q)
            idct_tile = idct(idct(unquantized_tile.T,  norm='ortho').T, norm='ortho')
            color_tiles.append(idct_tile)
        recon.append(combineTiles(color_tiles, (row, col), 8))
        
    reconstructed = np.zeros((row*8, col*8, 3),  dtype=float)
    reconstructed[:,:,0], reconstructed[:,:,1],  reconstructed[:,:,2] = recon[0], recon[1], recon[2]
    cropped = reconstructed[:original_shape[0], :original_shape[1]]
    return cropped

class JpegEncoder(EncoderBase):
    def __init__(self):
        pass 

    def imgToBits(self, img, qualityFactor):
        pil_image = PIL.Image.fromarray((img).astype(np.uint8), "RGB").convert("YCbCr")
        image = np.ndarray((pil_image.size[1], pil_image.size[0], 3), 'u1', pil_image.tobytes()).astype(float)
        y, u, v = blockSplit(image[:,:,0], 8), blockSplit(image[:,:,1], 8), blockSplit(image[:,:,2], 8)
        y_quantize, u_quantize, v_quantize = dctQuantize(y, 'y', qualityFactor), dctQuantize(u, 'u', qualityFactor),\
                                             dctQuantize(v, 'v', qualityFactor)
        y_zrl, u_zrl, v_zrl = rleTiles(y_quantize), rleTiles(u_quantize), rleTiles(v_quantize)
        zrl_tiles = [[y_zrl[i], u_zrl[i], v_zrl[i]] for i in range(len(y_zrl))]    
        encoded = jpeg_huffman.JPEG_RL_TO_HUFF(zrl_tiles)
        row, col = image.shape[0], image.shape[1]
        encoded = str(bin(row))[2:].zfill(11) + str(bin(col))[2:].zfill(11) + str(bin(qualityFactor))[2:].zfill(7) + encoded
        encoded = encoded + '0' * ((8-len(encoded)%8)%8)
        return encoded

    def bitsToImg(self, bits, show=False, filename=None):
        row, col = int(bits[0:11], 2), int(bits[11:22], 2) 
        quality_factor = int(bits[22:29], 2)
        print 'size: ', row, col
        print 'quality factor: ', quality_factor
        zrl_tiles = jpeg_huffman.JPEG_HUFF_TO_RL(bits[29:])
        quantized_tiles = decodeFromZRLTiles(zrl_tiles)
        yuv_quantized = []         
        for i in range(3): 
            quantized = [] 
            for j in range(len(quantized_tiles)):
                quantized.append(quantized_tiles[j][i])
            yuv_quantized.append(quantized)        
        recon = reconstruct(yuv_quantized, (row, col), quality_factor)
        recon = np.where(recon > 255, 255, recon)
        recon = np.where(recon < 0, 0, recon)

        rgb_recon_PIL = PIL.Image.fromarray((recon).astype(np.uint8), "YCbCr").convert("RGB")
        rgb_recon = np.ndarray((row, col, 3), 'u1', rgb_recon_PIL.tobytes()).astype(np.uint8)
        if show: 
            plt.figure(figsize=(16,16))
            plt.imshow(rgb_recon)
        return rgb_recon