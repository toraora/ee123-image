import util
import packet
import modem
import zlib
import numpy as np
import pyaudio
import serial
import sys
import Queue
import time
import threading

class EE123_DT_LAYER(object):
    def __init__(self, baud = 2400, fc = 2400, fs = 48000, packet_siz = 1536, ecc = 6, ecc_p = 6, pll_a = 0.7):
        self.packer = packet.packer(packet_siz, ecc, ecc_p)
        self.modem = modem.FSK_modem(fs, baud, fc)
        self.pll_a = pll_a
        self.ready = False
        p = pyaudio.PyAudio()
        util.printDevNumbers(p)
        p.terminate()

    def dry_run(self, bits, typ = 0):
        bits = np.array(list(bits), dtype=str)
        if bits.shape[0] % 8 != 0:
            raise Exception("Number of bits should be a multiple of 8")

        bits_compressed = util.ASCIIToBin(min([zlib.compress(util.binToASCII("".join(bits)), i) for i in range(1,10)], key = lambda s: len(s)))
        ENT_RATIO = len(bits_compressed) / float(len(bits))
        bits_packets = self.packer.packets_to_bits(self.packer.encode(bits_compressed, typ))
        PACKED_BYTES = len(bits_packets) / 8.
        print "\nEntropy compressed ratio: ", ENT_RATIO
        print "Final size (bytes): ", PACKED_BYTES
        print "Channel efficiency ratio: ", len(bits_compressed) / 8. / PACKED_BYTES

        self.audio_sig = self.modem.modulate(np.array(list(bits_packets), dtype=int))
        print "Audio time is: ", len(self.audio_sig) / 48000.
        if len(self.audio_sig) < 75 * 48000:
            self.ready = True
            print "ready to go! call dt.transmit()"

    def transmit(self, out_port, s):
        if not self.ready:
            print "Dry run first!"
            return
        util.loopback(self.audio_sig, s, out_port)
        self.ready = False

    def loopback(self, out_port, in_port, s):
        if not self.ready:
            print "Dry run first!"
            return
        dat = util.loopback(self.audio_sig, s, out_port, in_port)
        self.ready = False
        return dat

    def receive(self, in_port, t = 75):
        p = pyaudio.PyAudio()
        Qin = Queue.Queue()
        cQin = Queue.Queue()
        t_rec = threading.Thread(target = util.record_audio,   args = (Qin, cQin, p, 48000, in_port))
        t_rec.start()

        print "STARTED RECORDING..."
        sys.stdout.flush()
        time.sleep(t)
        cQin.put('EOT')
        p.terminate()
        print "END RECORDING..."

        data = np.array([])
        while not Qin.empty():
            data = np.concatenate((data, Qin.get()))

        np.save("RECEIVED_LAST.npy", data)

        print "START DECODE..."
        rcv_bits = self.modem.demodulate_bits(data, pll_a = self.pll_a).astype(int)
        print "received bits:", len(rcv_bits)
        rcv_packets, n_p = self.packer.decode(rcv_bits)
        print "received packets: ", len(rcv_packets), "\t total packets: ", n_p
        payload = self.packer.decode_from_packets(rcv_packets, n_p)
        decomp = zlib.decompress(util.binToASCII(payload[2]))

        return util.ASCIIToBin(decomp)

    def receive_saved(self, data):
        print "START DECODE..."
        rcv_bits = self.modem.demodulate_bits(data, pll_a = self.pll_a).astype(int)
        print "received bits:", len(rcv_bits)
        rcv_packets, n_p = self.packer.decode(rcv_bits)
        print "received packets: ", len(rcv_packets), "\t total packets: ", n_p
        payload = self.packer.decode_from_packets(rcv_packets, n_p)
        decomp = zlib.decompress(util.binToASCII(payload[2]))

        return util.ASCIIToBin(decomp)
