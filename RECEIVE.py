import data_transport
import decimate
import matplotlib.pyplot as plt
import color_decimate
import Queue
import threading
import time
import sys
import jpeg
import numpy as np

de = decimate.DecimateEncoder()
cde = color_decimate.ColorDecimateEncoder()
jp = jpeg.JpegEncoder()
dt = data_transport.EE123_DT_LAYER(packet_siz = 1536, ecc = 4, ecc_p = 4)


def DT_RECEIVE_ASYNC(dt, p, t, Q):
    try:
        Q.put(dt.receive(p, t))
    except:
        Q.put(False)
    return

def get_image(filename, encoder, t = 75, p = 2):
    print "FILENAME: ", filename
    print "ENCODER: ", encoder
    print "time: ", t, "\t audio: ", p
    sys.stdout.flush()
    try:
        Q = Queue.Queue()
        th = threading.Thread(target = DT_RECEIVE_ASYNC, args = (dt, p, t, Q))
        th.daemon = True
        th.start()

        i = 0
        while Q.empty():
            time.sleep(0.5)
            sys.stdout.flush()

        bits = Q.get()
        np.save("RECEIVED_LAST_BITS.npy", bits)

        if not bits:
            raise Exception("failed to decode")

        img = encoder.bitsToImg(bits)
        plt.imsave(filename, img.astype(np.uint8))
        print "done!!"
    except Exception as e:
        print "failed:", e

    return

test = lambda: get_image("asdf.png", de, 5, 2)
