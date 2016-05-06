import numpy as np
import util
import reedsolo
import zlib
import math

synch = "01" * 5
preamble = '011111110'
callsign = util.ASCIIToBin('KM6BJX')

# PACKET format
# preamble | header | payload | ecc | postamble
# header is 32 (crc) + 8 (packet no) + 8 (total packets) bits
# ecc is ecc_p * 16 bits
# preable / postamble are 9 bits
# packet efficiency is: payload_len / (payload_len + 57 + ecc_p * 16)

# with 1024 packet size
# at 2400 baud, we get ~2 packets per second
# that gives ~150 packets in 75s
# with ecc = ecc_p = 8, using the full time,
# we can transmit 130000 bits in 74s
# with channel efficiency ~74%

# message format:
# payload type (2 bits? 00: cartoon, 01: wavelet)
# payload size in bits (let's fix this to 24 bits)
# payload (variable, zlib compressed)

class packer(object):
    def __init__(self, packet = 1024, ecc = 24, ecc_p = 16):
        self.packet = packet
        self.ecc = ecc
        self.ecc_p = ecc_p
        self.codec = reedsolo.RSCodec(ecc * 2)
        self.codec_p = reedsolo.RSCodec(ecc_p * 2)


    def encode(self, bits, typ = 0):
        bits = np.array(list(bits), dtype=int)
        NUM_PACKETS = str(bin(int(math.ceil((2 + 24 + bits.shape[0]) / (0. + self.packet))) + 2 * self.ecc))[2:].zfill(8)
        PAYLOAD_TYPE = str(bin(typ))[2:].zfill(2)

        NUM_BITS = str(bin(bits.shape[0]))[2:].zfill(24)
        bits = np.concatenate((np.array(list(PAYLOAD_TYPE), dtype=int), np.array(list(NUM_BITS), dtype=int), bits))
        bits = np.array(list(bits), dtype=int)
        z_pad = (self.packet - (bits.shape[0] % self.packet)) % self.packet
        print "NUM BITS: ", bits.shape[0], "\t ZPAD: ", z_pad
        bits = np.concatenate((bits, np.zeros(z_pad))).astype(int)
        packets = np.reshape(bits, (-1, self.packet))

        packets_ascii = [list(util.binToASCII("".join(packet))) for packet in packets.astype(str)]
        packets_ecc = [self.codec.encode("".join(packet)) for packet in zip(*packets_ascii)]

        encoded_packets = []
        for packet_num in range(int(NUM_PACKETS, 2)):
            packet_payload = "".join([chr(packets_ecc[i][packet_num]) for i in range(self.packet / 8)])
            packet_nocrc = util.binToASCII(str(bin(packet_num))[2:].zfill(8)) + util.binToASCII(NUM_PACKETS) + packet_payload
            packet_withcrc = util.binToASCII(str(bin(zlib.crc32(packet_nocrc) % 2**32))[2:].zfill(32)) + packet_nocrc
            packet_ecc = str(self.codec_p.encode(packet_withcrc))
            encoded_packets += [packet_ecc]
        return encoded_packets

    def packets_to_bits(self, packets):
        packets_bits = [util.ASCIIToBin(packet) for packet in packets]
        packets_bits_stuffed = [list(util.bit_stuff(packet)) for packet in packets_bits]
        packets_bits_nrzi = [util.NRZ2NRZI(packet) for packet in packets_bits_stuffed]
        return callsign + synch + preamble + (preamble).join(packets_bits_nrzi) + preamble

    # give packets in (packet_num, packet_payload) format
    def decode_from_packets(self, packets, numpackets):
        if len(packets) == 0 or len(packets) < numpackets - 2 * self.ecc:
            raise Exception("Too many packets dropped!")

        packets_order = ['0' * self.packet for _ in range(numpackets)]
        for idx, payload in packets:
            packets_order[idx] = payload
        packets_order = np.array(packets_order)
        packets_ascii = [list(util.binToASCII("".join(packet))) for packet in packets_order.astype(str)]
        packets_corrected = [self.codec.decode("".join(packet)) for packet in zip(*packets_ascii)]

        payload = ''
        for packet_num in range(numpackets - 2 * self.ecc):
            packet_payload = "".join([chr(packets_corrected[i][packet_num]) for i in range(self.packet / 8)])
            payload += packet_payload
        payload_bits = util.ASCIIToBin(payload)
        payload_type = payload_bits[:2]
        payload_size = payload_bits[2:26]
        payload = payload_bits[26:26 + int(payload_size, 2)]
        return int(payload_type, 2), int(payload_size, 2), payload

    def decode(self, bits):
        bits = np.array(list(bits), dtype=int)
        packets = []
        flag = list(np.array(list(preamble), dtype=int))[-9:]
        cur_b = []
        num_packets = '0'
        for n in range(0, bits.shape[0] - 7):
            if list(bits[n:n+len(flag)-1]) == flag[1:]:
                if len(cur_b): # end packet; do CRC and add to packets
                    cur = "".join([str(c) for c in cur_b[len(flag)-2:-1]])
                    cur = util.NRZI2NRZ(cur)
                    cur = util.bit_unstuff(cur)
                    cur = "".join(['1' if b else '0' for b in cur])
                    if len(cur) < self.packet + 48:
                        cur_b = []
                        continue
                    cur += '0' * (self.packet + 48 + self.ecc_p * 16 - len(cur))
                    try:
                        if self.ecc_p:
                            cur = util.ASCIIToBin(str(self.codec_p.decode(util.binToASCII(cur))))
                    except:
                        cur_b = []
                        continue
                    crc = cur[0:32]
                    idx = cur[32:40]
                    tot = cur[40:48]
                    payload = cur[48:]
                    if (zlib.crc32(util.binToASCII(idx + tot + payload)) % 2**32) == int("".join(crc), 2):
                        packets += [(int(idx, 2), payload)]
                        num_packets = tot
                cur_b = []
            else:
                cur_b += [bits[n]]
        return packets, int(num_packets, 2)
