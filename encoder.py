import numpy as np
import data_transport, util

class EncoderBase(object):
    def imgToBits(self, img):
        pass
    def bitsToImg(self, bits):
        pass
    def self_test(self, img, encArgs = [], decArgs = []):
        dt = data_transport.EE123_DT_LAYER()
        dt.dry_run(self.imgToBits(img, *encArgs))
        print "PSNR: ", util.PSNR_official(img, self.bitsToImg(self.imgToBits(img, *encArgs), *decArgs))
