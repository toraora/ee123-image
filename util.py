from numpy import *
from matplotlib.pyplot import *
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import Queue
import threading,time
import sys

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from numpy import ones
from scipy import signal
from scipy import integrate

import threading,time
import multiprocessing

from numpy import mean
from numpy import power
from numpy.fft import fft
from numpy.fft import fftshift
from numpy.fft import ifft
from numpy.fft import ifftshift
import bitarray
from  scipy.io.wavfile import read as wavread
import serial
from fractions import gcd

def PSNR(i1, i2):
    i1 = i1.astype(np.int64)
    i2 = i2.astype(np.int64)
    mse = np.mean((i1 - i2)**2)
    return 10 * np.log10(255**2 / mse)

def PSNR_official(im_truth, im_test, maxval=255.):
    mse = np.linalg.norm(im_truth.astype(np.float64) - im_test.astype(np.float64))**2 / np.prod(np.shape(im_truth))
    return 10 * np.log10(maxval**2 / mse)

def NRZ2NRZI(NRZ):
    NRZ = np.array(list(NRZ)).astype(int).astype(bool)
    NRZI = NRZ.copy()
    current = True
    for n in range(0,len(NRZ)):
        if NRZ[n] :
            NRZI[n] = current
        else:
            NRZI[n] = not(current)
        current = NRZI[n]
    return "".join(NRZI.astype(int).astype(str))

def NRZI2NRZ(NRZI, current = True):
    NRZI = np.array(list(NRZI)).astype(int).astype(bool)
    NRZ = NRZI.copy()

    for n in range(0,len(NRZI)):
        NRZ[n] = NRZI[n] == current
        current = NRZI[n]

    return NRZ


def binToASCII(b):
    return "".join([chr(int(b[8*i:8*i+8], 2)) \
        for i in range(len(b) / 8)])
def ASCIIToBin(a):
    return "".join([bin(ord(c))[2:].zfill(8) \
        for c in a])
def bytearray_ASCIIToBin(a):
    return "".join([bin(c)[2:].zfill(8) \
        for c in a])

def bit_stuff(data):
    count = 0
    for bit in data:
        bit = int(bit)
        if bit:
            count += 1
        else:
            count = 0
        yield bit
        if count == 5:
            yield 0
            count = 0

def bit_unstuff(data):
    count = 0
    skip = False
    ret_bits = []
    for bit in data:
        if not(skip):
            if bit:
                count += 1
            else:
                count = 0
            ret_bits += [int(bit)]

            if count == 5:
                skip = True;
                count = 0
        else:
            skip = False
    return np.array(ret_bits)


# function to compute least common multipler
def lcm(numbers):
    return reduce(lambda x, y: (x*y)/gcd(x,y), numbers, 1)


# function to compute average power spectrum
def avgPS( x, N=256, fs=1):
    M = floor(len(x)/N)
    x_ = reshape(x[:M*N],(M,N)) * np.hamming(N)[None,:]
    X = np.fft.fftshift(np.fft.fft(x_,axis=1),axes=1)
    return r_[-N/2.0:N/2.0]/N*fs, mean(abs(X)**2,axis=0)


# Plot an image of the spectrogram y, with the axis labeled with time tl,
# and frequency fl
#
# t_range -- time axis label, nt samples
# f_range -- frequency axis label, nf samples
# y -- spectrogram, nf by nt array
# dbf -- Dynamic range of the spect

def sg_plot( t_range, f_range, y, dbf = 60, fig = None) :
    eps = 10.0**(-dbf/20.0)  # minimum signal

    # find maximum
    y_max = abs(y).max()

    # compute 20*log magnitude, scaled to the max
    y_log = 20.0 * np.log10( (abs( y ) / y_max)*(1-eps) + eps )

    # rescale image intensity to 256
    img = 256*(y_log + dbf)/dbf - 1

    fig=figure(figsize=(16,6))

    plt.imshow( np.flipud( 64.0*(y_log + dbf)/dbf ), extent= t_range  + f_range ,cmap=plt.cm.gray, aspect='auto')
    plt.xlabel('Time, s')
    plt.ylabel('Frequency, Hz')
    plt.tight_layout()

    return fig


def myspectrogram_hann_ovlp(x, m, fs, fc,dbf = 60):
    # Plot the spectrogram of x.
    # First take the original signal x and split it into blocks of length m
    # This corresponds to using a rectangular window %


    isreal_bool = np.isreal(x).all()

    # pad x up to a multiple of m
    lx = len(x);
    nt = (lx + m - 1) // m
    x = append(x,zeros(-lx+nt*m))
    x = x.reshape((m/2,nt*2), order='F')
    x = concatenate((x,x),axis=0)
    x = x.reshape((m*nt*2,1),order='F')
    x = x[r_[m//2:len(x),ones(m//2)*(len(x)-1)].astype(int)].reshape((m,nt*2),order='F')


    xmw = x * hanning(m)[:,None];


    # frequency index
    t_range = [0.0, lx / fs]

    if isreal_bool:
        f_range = [ fc, fs / 2.0 + fc]
        xmf = np.fft.fft(xmw,len(xmw),axis=0)
        sg_plot(t_range, f_range, xmf[0:m/2,:],dbf=dbf)
        print 1
    else:
        f_range = [-fs / 2.0 + fc, fs / 2.0 + fc]
        xmf = np.fft.fftshift( np.fft.fft( xmw ,len(xmw),axis=0), axes=0 )
        sg_plot(t_range, f_range, xmf,dbf = dbf)

    return t_range, f_range, xmf

def play_audio( Q,ctrlQ ,p, fs , dev, ser="", keydelay=0.1):
    # play_audio plays audio with sampling rate = fs
    # Q - A queue object from which to play
    # ctrlQ - A queue object for ending the thread
    # p   - pyAudio object
    # fs  - sampling rate
    # dev - device number
    # ser - pyserial device to key the radio
    # keydelay - delay after keying the radio
    #
    #
    # There are two ways to end the thread:
    #    1 - send "EOT" through  the control queue. This is used to terminate the thread on demand
    #    2 - send "EOT" through the data queue. This is used to terminate the thread when data is done.
    #
    # You can also key the radio either through the data queu and the control queue


    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),output=True,output_device_index=dev)
    # play audio
    while (1):
        if not ctrlQ.empty():

            # control queue
            ctrlmd = ctrlQ.get()
            if ctrlmd is "EOT"  :
                    ostream.stop_stream()
                    ostream.close()
                    print("Closed  play thread")
                    return;
            elif (ctrlmd is "KEYOFF"  and ser!=""):
                ser.setDTR(0)
                #print("keyoff\n")
            elif (ctrlmd is "KEYON" and ser!=""):
                ser.setDTR(1)  # key PTT
                #print("keyon\n")
                time.sleep(keydelay) # wait 200ms (default) to let the power amp to ramp up


        data = Q.get()

        if (data is "EOT") :
            ostream.stop_stream()
            ostream.close()
            print("Closed  play thread")
            return;
        elif (data is "KEYOFF"  and ser!=""):
            ser.setDTR(0)
            #print("keyoff\n")
        elif (data is "KEYON" and ser!=""):
            ser.setDTR(1)  # key PTT
            #print("keyon\n")
            time.sleep(keydelay) # wait 200ms (default) to let the power amp to ramp up

        else:
            try:
                ostream.write( data.astype(np.float32).tostring() )
            except:
                print("Exception")
                break

def record_audio( queue,ctrlQ, p, fs ,dev,chunk=1024):
    # record_audio records audio with sampling rate = fs
    # queue - output data queue
    # p     - pyAudio object
    # fs    - sampling rate
    # dev   - device number
    # chunk - chunks of samples at a time default 1024
    #
    # Example:
    # fs = 44100
    # Q = Queue.queue()
    # p = pyaudio.PyAudio() #instantiate PyAudio
    # record_audio( Q, p, fs, 1) #
    # p.terminate() # terminate pyAudio


    istream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),input=True,input_device_index=dev,frames_per_buffer=chunk)

    # record audio in chunks and append to frames
    frames = [];
    while (1):
        if not ctrlQ.empty():
            ctrlmd = ctrlQ.get()
            if ctrlmd is "EOT"  :
                istream.stop_stream()
                istream.close()
                print("Closed  record thread")
                return;
        try:  # when the pyaudio object is distroyed stops
            data_str = istream.read(chunk) # read a chunk of data
        except:
            break
        data_flt = np.fromstring( data_str, 'float32' ) # convert string to float
        queue.put( data_flt ) # append to list




def text2Morse(text,fc,fs,dt):
    CODE = {'A': '.-',     'B': '-...',   'C': '-.-.',
        'D': '-..',    'E': '.',      'F': '..-.',
        'G': '--.',    'H': '....',   'I': '..',
        'J': '.---',   'K': '-.-',    'L': '.-..',
        'M': '--',     'N': '-.',     'O': '---',
        'P': '.--.',   'Q': '--.-',   'R': '.-.',
     	'S': '...',    'T': '-',      'U': '..-',
        'V': '...-',   'W': '.--',    'X': '-..-',
        'Y': '-.--',   'Z': '--..',

        '0': '-----',  '1': '.----',  '2': '..---',
        '3': '...--',  '4': '....-',  '5': '.....',
        '6': '-....',  '7': '--...',  '8': '---..',
        '9': '----.',

        ' ': ' ', "'": '.----.', '(': '-.--.-',  ')': '-.--.-',
        ',': '--..--', '-': '-....-', '.': '.-.-.-',
        '/': '-..-.',   ':': '---...', ';': '-.-.-.',
        '?': '..--..', '_': '..--.-'
        }

    Ndot= 1.0*fs*dt
    Ndah = 3*Ndot

    sdot = sin(2*pi*fc*r_[0.0:Ndot]/fs)
    sdah = sin(2*pi*fc*r_[0.0:Ndah]/fs)

    # convert to dit dah
    mrs = ""
    for char in text:
        mrs = mrs + CODE[char.upper()] + "*"

    sig = zeros(1)
    for char in mrs:
        if char == " ":
            sig = concatenate((sig,zeros(Ndot*7)))
        if char == "*":
            sig = concatenate((sig,zeros(Ndot*3)))
        if char == ".":
            sig = concatenate((sig,sdot,zeros(Ndot)))
        if char == "-":
            sig = concatenate((sig,sdah,zeros(Ndot)))
    return sig



def printDevNumbers(p):
    N = p.get_device_count()
    for n in range(0,N):
        name = p.get_device_info_by_index(n).get('name')
        print n, name


def loopback(signal, s, out, inp = None):
    p = pyaudio.PyAudio()
    Q = Queue.Queue()
    ctrlQ = Queue.Queue()  # dummy, but necessary!

    Qin = Queue.Queue()
    cQin = Queue.Queue()
    fs_usb = p.get_device_info_by_index(2)['defaultSampleRate']


    t_rec = threading.Thread(target = record_audio,   args = (Qin, cQin, p, 48000, inp))
    if inp:
        t_rec.start()

    Q.put(np.zeros(48000 * 0.5))
    Q.put(signal*0.2)
    Q.put(np.zeros(48000 * 0.5))
    Q.put('EOT')

    s.setDTR(1)
    play_audio(Q, ctrlQ, p, 48000, out)
    s.setDTR(0)

    ctrlQ.put('EOT')
    cQin.put('EOT')
    time.sleep(0.5) # give time for the thread to get killed

    p.terminate()

    if inp:
        data = np.array([])
        while not Qin.empty():
            data = np.concatenate((data, Qin.get()))

        return data

def grayToRGB(grayscale):
    img = np.zeros((grayscale.shape[0], grayscale.shape[1], 3), dtype=np.uint8)
    img[:,:,0] = grayscale
    img[:,:,1] = grayscale
    img[:,:,2] = grayscale
    return img
