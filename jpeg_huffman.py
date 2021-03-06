
# coding: utf-8

# In[2]:

HUFFMAN_LUM_DC = {
    "00":0,
    "010":1,
    "011":2,
    "100":3,
    "101":4,
    "110":5,
    "1110":6,
    "11110":7,
    "111110":8,
    "1111110":9,
    "11111110":10,
    "111111110":11
}

HUFFMAN_LUM_AC = {
    "1010":"0/0",
    "00":"0/1",
    "01":"0/2",
    "100":"0/3",
    "1011":"0/4",
    "11010":"0/5",
    "1111000":"0/6",
    "11111000":"0/7",
    "1111110110":"0/8",
    "1111111110000010":"0/9",
    "1111111110000011":"0/A",
    "1100":"1/1",
    "11011":"1/2",
    "1111001":"1/3",
    "111110110":"1/4",
    "11111110110":"1/5",
    "1111111110000100":"1/6",
    "1111111110000101":"1/7",
    "1111111110000110":"1/8",
    "1111111110000111":"1/9",
    "1111111110001000":"1/A",
    "11100":"2/1",
    "11111001":"2/2",
    "1111110111":"2/3",
    "111111110100":"2/4",
    "1111111110001001":"2/5",
    "1111111110001010":"2/6",
    "1111111110001011":"2/7",
    "1111111110001100":"2/8",
    "1111111110001101":"2/9",
    "1111111110001110":"2/A",
    "111010":"3/1",
    "111110111":"3/2",
    "111111110101":"3/3",
    "1111111110001111":"3/4",
    "1111111110010000":"3/5",
    "1111111110010001":"3/6",
    "1111111110010010":"3/7",
    "1111111110010011":"3/8",
    "1111111110010100":"3/9",
    "1111111110010101":"3/A",
    "111011":"4/1",
    "1111111000":"4/2",
    "1111111110010110":"4/3",
    "1111111110010111":"4/4",
    "1111111110011000":"4/5",
    "1111111110011001":"4/6",
    "1111111110011010":"4/7",
    "1111111110011011":"4/8",
    "1111111110011100":"4/9",
    "1111111110011101":"4/A",
    "1111010":"5/1",
    "11111110111":"5/2",
    "1111111110011110":"5/3",
    "1111111110011111":"5/4",
    "1111111110100000":"5/5",
    "1111111110100001":"5/6",
    "1111111110100010":"5/7",
    "1111111110100011":"5/8",
    "1111111110100100":"5/9",
    "1111111110100101":"5/A",
    "1111011":"6/1",
    "111111110110":"6/2",
    "1111111110100110":"6/3",
    "1111111110100111":"6/4",
    "1111111110101000":"6/5",
    "1111111110101001":"6/6",
    "1111111110101010":"6/7",
    "1111111110101011":"6/8",
    "1111111110101100":"6/9",
    "1111111110101101":"6/A",
    "11111010":"7/1",
    "111111110111":"7/2",
    "1111111110101110":"7/3",
    "1111111110101111":"7/4",
    "1111111110110000":"7/5",
    "1111111110110001":"7/6",
    "1111111110110010":"7/7",
    "1111111110110011":"7/8",
    "1111111110110100":"7/9",
    "1111111110110101":"7/A",
    "111111000":"8/1",
    "111111111000000":"8/2",
    "1111111110110110":"8/3",
    "1111111110110111":"8/4",
    "1111111110111000":"8/5",
    "1111111110111001":"8/6",
    "1111111110111010":"8/7",
    "1111111110111011":"8/8",
    "1111111110111100":"8/9",
    "1111111110111101":"8/A",
    "111111001":"9/1",
    "1111111110111110":"9/2",
    "1111111110111111":"9/3",
    "1111111111000000":"9/4",
    "1111111111000001":"9/5",
    "1111111111000010":"9/6",
    "1111111111000011":"9/7",
    "1111111111000100":"9/8",
    "1111111111000101":"9/9",
    "1111111111000110":"9/A",
    "111111010":"A/1",
    "1111111111000111":"A/2",
    "1111111111001000":"A/3",
    "1111111111001001":"A/4",
    "1111111111001010":"A/5",
    "1111111111001011":"A/6",
    "1111111111001100":"A/7",
    "1111111111001101":"A/8",
    "1111111111001110":"A/9",
    "1111111111001111":"A/A",
    "1111111001":"B/1",
    "1111111111010000":"B/2",
    "1111111111010001":"B/3",
    "1111111111010010":"B/4",
    "1111111111010011":"B/5",
    "1111111111010100":"B/6",
    "1111111111010101":"B/7",
    "1111111111010110":"B/8",
    "1111111111010111":"B/9",
    "1111111111011000":"B/A",
    "1111111010":"C/1",
    "1111111111011001":"C/2",
    "1111111111011010":"C/3",
    "1111111111011011":"C/4",
    "1111111111011100":"C/5",
    "1111111111011101":"C/6",
    "1111111111011110":"C/7",
    "1111111111011111":"C/8",
    "1111111111100000":"C/9",
    "1111111111100001":"C/A",
    "11111111000":"D/1",
    "1111111111100010":"D/2",
    "1111111111100011":"D/3",
    "1111111111100100":"D/4",
    "1111111111100101":"D/5",
    "1111111111100110":"D/6",
    "1111111111100111":"D/7",
    "1111111111101000":"D/8",
    "1111111111101001":"D/9",
    "1111111111101010":"D/A",
    "1111111111101011":"E/1",
    "1111111111101100":"E/2",
    "1111111111101101":"E/3",
    "1111111111101110":"E/4",
    "1111111111101111":"E/5",
    "1111111111110000":"E/6",
    "1111111111110001":"E/7",
    "1111111111110010":"E/8",
    "1111111111110011":"E/9",
    "1111111111110100":"E/A",
    "11111111001":"F/0",
    "1111111111110101":"F/1",
    "1111111111110110":"F/2",
    "1111111111110111":"F/3",
    "1111111111111000":"F/4",
    "1111111111111001":"F/5",
    "1111111111111010":"F/6",
    "1111111111111011":"F/7",
    "1111111111111100":"F/8",
    "1111111111111101":"F/9",
    "1111111111111110":"F/A"
}

HUFFMAN_CHR_DC = {
    "00":0,
    "01":1,
    "10":2,
    "110":3,
    "1110":4,
    "11110":5,
    "111110":6,
    "1111110":7,
    "11111110":8,
    "111111110":9,
    "1111111110":10,
    "11111111110":11
}

HUFFMAN_CHR_AC = {
    "00":"0/0",
    "01":"0/1",
    "100":"0/2",
    "1010":"0/3",
    "11000":"0/4",
    "11001":"0/5",
    "111000":"0/6",
    "1111000":"0/7",
    "111110100":"0/8",
    "1111110110":"0/9",
    "111111110100":"0/A",
    "1011":"1/1",
    "111001":"1/2",
    "11110110":"1/3",
    "111110101":"1/4",
    "11111110110":"1/5",
    "111111110101":"1/6",
    "1111111110001000":"1/7",
    "1111111110001001":"1/8",
    "1111111110001010":"1/9",
    "1111111110001011":"1/A",
    "11010":"2/1",
    "11110111":"2/2",
    "1111110111":"2/3",
    "111111110110":"2/4",
    "111111111000010":"2/5",
    "1111111110001100":"2/6",
    "1111111110001101":"2/7",
    "1111111110001110":"2/8",
    "1111111110001111":"2/9",
    "1111111110010000":"2/A",
    "11011":"3/1",
    "11111000":"3/2",
    "1111111000":"3/3",
    "111111110111":"3/4",
    "1111111110010001":"3/5",
    "1111111110010010":"3/6",
    "1111111110010011":"3/7",
    "1111111110010100":"3/8",
    "1111111110010101":"3/9",
    "1111111110010110":"3/A",
    "111010":"4/1",
    "111110110":"4/2",
    "1111111110010111":"4/3",
    "1111111110011000":"4/4",
    "1111111110011001":"4/5",
    "1111111110011010":"4/6",
    "1111111110011011":"4/7",
    "1111111110011100":"4/8",
    "1111111110011101":"4/9",
    "1111111110011110":"4/A",
    "111011":"5/1",
    "1111111001":"5/2",
    "1111111110011111":"5/3",
    "1111111110100000":"5/4",
    "1111111110100001":"5/5",
    "1111111110100010":"5/6",
    "1111111110100011":"5/7",
    "1111111110100100":"5/8",
    "1111111110100101":"5/9",
    "1111111110100110":"5/A",
    "1111001":"6/1",
    "11111110111":"6/2",
    "1111111110100111":"6/3",
    "1111111110101000":"6/4",
    "1111111110101001":"6/5",
    "1111111110101010":"6/6",
    "1111111110101011":"6/7",
    "1111111110101100":"6/8",
    "1111111110101101":"6/9",
    "1111111110101110":"6/A",
    "1111010":"7/1",
    "11111111000":"7/2",
    "1111111110101111":"7/3",
    "1111111110110000":"7/4",
    "1111111110110001":"7/5",
    "1111111110110010":"7/6",
    "1111111110110011":"7/7",
    "1111111110110100":"7/8",
    "1111111110110101":"7/9",
    "1111111110110110":"7/A",
    "11111001":"8/1",
    "1111111110110111":"8/2",
    "1111111110111000":"8/3",
    "1111111110111001":"8/4",
    "1111111110111010":"8/5",
    "1111111110111011":"8/6",
    "1111111110111100":"8/7",
    "1111111110111101":"8/8",
    "1111111110111110":"8/9",
    "1111111110111111":"8/A",
    "111110111":"9/1",
    "1111111111000000":"9/2",
    "1111111111000001":"9/3",
    "1111111111000010":"9/4",
    "1111111111000011":"9/5",
    "1111111111000100":"9/6",
    "1111111111000101":"9/7",
    "1111111111000110":"9/8",
    "1111111111000111":"9/9",
    "1111111111001000":"9/A",
    "111111000":"A/1",
    "1111111111001001":"A/2",
    "1111111111001010":"A/3",
    "1111111111001011":"A/4",
    "1111111111001100":"A/5",
    "1111111111001101":"A/6",
    "1111111111001110":"A/7",
    "1111111111001111":"A/8",
    "1111111111010000":"A/9",
    "1111111111010001":"A/A",
    "111111001":"B/1",
    "1111111111010010":"B/2",
    "1111111111010011":"B/3",
    "1111111111010100":"B/4",
    "1111111111010101":"B/5",
    "1111111111010110":"B/6",
    "1111111111010111":"B/7",
    "1111111111011000":"B/8",
    "1111111111011001":"B/9",
    "1111111111011010":"B/A",
    "111111010":"C/1",
    "1111111111011011":"C/2",
    "1111111111011100":"C/3",
    "1111111111011101":"C/4",
    "1111111111011110":"C/5",
    "1111111111011111":"C/6",
    "1111111111100000":"C/7",
    "1111111111100001":"C/8",
    "1111111111100010":"C/9",
    "1111111111100011":"C/A",
    "11111111001":"D/1",
    "1111111111100100":"D/2",
    "1111111111100101":"D/3",
    "1111111111100110":"D/4",
    "1111111111100111":"D/5",
    "1111111111101000":"D/6",
    "1111111111101001":"D/7",
    "1111111111101010":"D/8",
    "1111111111101011":"D/9",
    "1111111111101100":"D/A",
    "11111111100000":"E/1",
    "1111111111101101":"E/2",
    "1111111111101110":"E/3",
    "1111111111101111":"E/4",
    "1111111111110000":"E/5",
    "1111111111110001":"E/6",
    "1111111111110010":"E/7",
    "1111111111110011":"E/8",
    "1111111111110100":"E/9",
    "1111111111110101":"E/A",
    "1111111010":"F/0",
    "111111111000011":"F/1",
    "1111111111110110":"F/2",
    "1111111111110111":"F/3",
    "1111111111111000":"F/4",
    "1111111111111001":"F/5",
    "1111111111111010":"F/6",
    "1111111111111011":"F/7",
    "1111111111111100":"F/8",
    "1111111111111101":"F/9",
    "1111111111111110":"F/A"
}


# In[3]:

HUFFMAN_LUM_DC_INV = {v:k for k,v in HUFFMAN_LUM_DC.items()}
HUFFMAN_LUM_AC_INV = {v:k for k,v in HUFFMAN_LUM_AC.items()}
HUFFMAN_CHR_DC_INV = {v:k for k,v in HUFFMAN_CHR_DC.items()}
HUFFMAN_CHR_AC_INV = {v:k for k,v in HUFFMAN_CHR_AC.items()}


# In[4]:
"""
s = '''

'''

for st in s.split('\n'):
    if len(st.strip()):
        cur = st.strip().split(' ')
        print '"' + cur[2] + '":"' + cur[0] + '",'

"""

# In[13]:

def JPEG_VAL_ENCODE(val):
    if val == 0:
        return ''
    neg = val < 0
    if neg:
        val = -val
    bits = str(bin(val))[2:]
    if neg:
        bits = "".join(['1' if bit == '0' else '0' for bit in bits])
    return bits

def JPEG_VAL_DECODE(bits):
    if not len(bits):
        return 0
    neg = bits[0] == '0'
    if neg:
        bits = "".join(['1' if bit == '0' else '0' for bit in bits])
    return (-1 if neg else 1) * int(bits, 2)

def JPEG_RL_TO_HUFF(tiles):
    bits = ''
    for tile in tiles:
        DC = True
        cnt = 0
        # for the first iteration, RL is actually irrelevant
        for rl, val in tile[0]: # LUM: Y in YCrCb
            if DC:
                val_bits = JPEG_VAL_ENCODE(val)
                size = len(val_bits)
                huff_code = HUFFMAN_LUM_DC_INV[size]
                bits += huff_code + val_bits
                DC = False
            else: #AC
                val_bits = JPEG_VAL_ENCODE(val)
                size = len(val_bits)
                key = hex(rl)[2:].upper() + "/" + hex(size)[2:].upper()
                #cnt += rl + 1
                huff_code = HUFFMAN_LUM_AC_INV[key]
                bits += huff_code + val_bits
                if cnt == 63 or key == '0/0': #EOB
                    break
        for chr_block in tile[1:3]: # CHR CHANNELS: Cr, Cb
            DC = True
            cnt = 0
            for rl, val in chr_block:
                if DC:
                    val_bits = JPEG_VAL_ENCODE(val)
                    size = len(val_bits)
                    huff_code = HUFFMAN_CHR_DC_INV[size]
                    bits += huff_code + val_bits
                    DC = False
                else: #AC
                    val_bits = JPEG_VAL_ENCODE(val)
                    size = len(val_bits)
                    key = hex(rl)[2:].upper() + "/" + hex(size)[2:].upper()
                    #cnt += rl + 1
                    huff_code = HUFFMAN_CHR_AC_INV[key]
                    bits += huff_code + val_bits
                    if cnt == 63 or key == '0/0': #EOB
                        break
    return bits

LUM_DC = 0
LUM_AC = 1
CHR0_DC = 2
CHR0_AC = 3
CHR1_DC = 4
CHR1_AC = 5

def JPEG_HUFF_TO_RL(bits):
    idx = 0
    cur = ''
    state = LUM_DC
    tiles = []
    tile = [[],[],[]]
    cnt = 0
    while idx < len(bits):
        cur += bits[idx]
        if state is LUM_DC:
            if cur in HUFFMAN_LUM_DC:
                val = HUFFMAN_LUM_DC[cur]
                dc_val = JPEG_VAL_DECODE(bits[idx+1:idx+1+val])
#                print "LUM_DC: ", cur, val, dc_val
                idx += val
                tile[0] += [(0, dc_val)]
                state = LUM_AC
                cur = ''
                cnt = 0
        if state is LUM_AC:
            if cur in HUFFMAN_LUM_AC:
                val = HUFFMAN_LUM_AC[cur]
                rl, siz = val.split('/')
                rl = int(rl, 16); siz = int(siz, 16)
                ac_val = JPEG_VAL_DECODE(bits[idx+1:idx+1+siz])
#                print "LUM_AC: ", cur, val, rl, siz, ac_val, cnt
                if rl is 0 and siz is 0: #EOB
                    tile[0] += [(0, 0)]
                    state = CHR0_DC
                    cur = ''
                    cnt = 0
                elif rl is 15 and siz is 0: #16 ZEROES:
                    tile[0] += [(15, 0)]
                    cur = ''
                    cnt += 16
                else:
                    idx += siz
                    cnt += rl + 1
                    tile[0] += [(rl, ac_val)]
                    cur = ''
                    if cnt == 63:
                        state = CHR0_DC
                        cnt = 0
                        cur = ''
        if state is CHR0_DC:
            if cur in HUFFMAN_CHR_DC:
                val = HUFFMAN_CHR_DC[cur]
                dc_val = JPEG_VAL_DECODE(bits[idx+1:idx+1+val])
#                print "CHR0_DC: ", cur, val, dc_val
                idx += val
                tile[1] += [(0, dc_val)]
                state = CHR0_AC
                cur = ''
                cnt = 0
        if state is CHR0_AC:
            if cur in HUFFMAN_CHR_AC:
                val = HUFFMAN_CHR_AC[cur]
                rl, siz = val.split('/')
                rl = int(rl, 16); siz = int(siz, 16)
                ac_val = JPEG_VAL_DECODE(bits[idx+1:idx+1+siz])
#                print "CHR0_AC: ", cur, val, rl, siz, ac_val
                if rl is 0 and siz is 0: #EOB
                    tile[1] += [(0, 0)]
                    state = CHR1_DC
                    cur = ''
                    cnt = 0
                elif rl is 15 and siz is 0: #16 ZEROES:
                    tile[1] += [(15, 0)]
                    cur = ''
                    cnt += 16
                else:
                    idx += siz
                    cnt += rl + 1
                    tile[1] += [(rl, ac_val)]
                    cur = ''
                    if cnt == 63:
                        state = CHR1_DC
                        cnt = 0
                        cur = ''
        if state is CHR1_DC:
            if cur in HUFFMAN_CHR_DC:
                val = HUFFMAN_CHR_DC[cur]
                dc_val = JPEG_VAL_DECODE(bits[idx+1:idx+1+val])
#                print "CHR1_DC: ", cur, val, dc_val
                idx += val
                tile[2] += [(0, dc_val)]
                state = CHR1_AC
                cur = ''
                cnt = 0
        if state is CHR1_AC:
            if cur in HUFFMAN_CHR_AC:
                val = HUFFMAN_CHR_AC[cur]
                rl, siz = val.split('/')
                rl = int(rl, 16); siz = int(siz, 16)
                ac_val = JPEG_VAL_DECODE(bits[idx+1:idx+1+siz])
#                print "CHR1_AC: ", cur, val, rl, siz, ac_val
                if rl is 0 and siz is 0: #EOB
                    tile[2] += [(0, 0)]
                    state = LUM_DC
                    cur = ''
                    cnt = 0
                    tiles += [tile]
                    tile = [[],[],[]]
                elif rl is 15 and siz is 0: #16 ZEROES:
                    tile[2] += [(15, 0)]
                    cur = ''
                    cnt += 16
                else:
                    idx += siz
                    cnt += rl + 1
                    tile[2] += [(rl, ac_val)]
                    cur = ''
                    if cnt == 63:
                        state = LUM_DC
                        cnt = 0
                        cur = ''
                        tiles += [tile]
                        tile = [[],[],[]]
        idx += 1
    return tiles

"""

# In[31]:

tiles = [[[(0, 105), (1, -2), (0, -1), (4, -1), (0, -1), (0, 0)], [(0, 6), (1, 1), (0, 0)], [(0, -3), (0, 0)]]] * 1000
encoded =  JPEG_RL_TO_HUFF(tiles)
print "LEN ENC: ", len(encoded) / 8.
decoded = JPEG_HUFF_TO_RL(encoded)
#print "DEC: ", decoded
assert str(tiles) == str(decoded)


# In[32]:

import data_transport
dt = data_transport.EE123_DT_LAYER()
dt.dry_run(encoded)


# In[10]:

int('0xA', 0)


# In[ ]:

"""
