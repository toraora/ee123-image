import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from fractions import gcd

def PLL(NRZa, a, fs, baud):
    print "fs: ", fs, " baud: ", baud
    counter = 0
    increment = int(1 + 2**32 / (float(fs) / baud))
    i = 0
    idx = []
    prev = 0
    for sample in NRZa:
        do_sample = counter > 2**31 - increment
        counter += increment

        if i != 0 and np.sign(prev) != np.sign(sample):
            counter = int(counter * a)

        if do_sample:
            counter -= int(2**32)
            idx += [i - 1]

        i += 1
        prev = sample
    return np.array(idx)

class FSK_modem(object):
    def __init__(self, fs = 48000, baud = 2400, fc = 1800):
        self.fs = fs
        self.fc = fc
        self.baud = baud
        self.f_mark = self.fc - baud / 4
        self.f_space = self.fc + baud / 4
        self.delta_f = (self.f_space - self.f_mark) / 2
        self.spb = self.fs / baud

        taps = 2 * int(fs * 2 / baud) + 1
        h = signal.firwin(taps, self.baud / 2., nyq = fs / 2., window = 'hanning')
        self.bp_mark = np.exp(2j * np.pi * self.f_mark * np.r_[0:taps] / float(fs)) * h
        self.bp_space = np.exp(2j * np.pi * self.f_space * np.r_[0:taps] / float(fs)) * h
        self.h_nrz = signal.firwin(taps, self.baud * 1.2, nyq = fs / 2., window = 'hanning')

    def modulate(self, bits):
        bits_repeated = 2 * (np.repeat(bits, self.spb) - 0.5)

        t = np.r_[0:bits_repeated.shape[0]] / float(self.fs)
        sig = np.cos(2 * np.pi * self.fc * t[:-1] - \
                     2 * np.pi * self.delta_f * integrate.cumtrapz(bits_repeated, t, dx = 1./self.fs))

        return sig

    def demodulate(self, sig):
        sig_mark = signal.fftconvolve(sig, self.bp_mark, mode='same')
        sig_space = signal.fftconvolve(sig, self.bp_space, mode='same')
        sig_nrz = np.abs(sig_mark) - np.abs(sig_space)
        sig_nrz_filt = signal.fftconvolve(sig_nrz, self.h_nrz, mode='same')
        return sig_nrz

    def demodulate_bits(self, sig, pll_a = 0.75):
        sig_nrz = self.demodulate(sig)
        idx = PLL(sig_nrz, pll_a, self.fs, self.baud)
        return (np.sign(sig_nrz[idx])>0)

modem = FSK_modem(baud = 2400, fc = 3000)
