from PIL import Image
import numpy as np
import sys
from utils import *
# top left corner of the image is (0,0)
# the pixels are aranged in the following order
#(0,0) -> (1,0) -> (2,0)
#|        |         |
#v        v         v
#(0,1) -> (1,1) -> (2,1)
#topleftx : the x coordinates of the original image that will become the top left corner of the cropped image
#toplefty : the y coordinates of the original image that will become the top left corner of the cropped image
def crop640by480(inf,outf,topleftx,toplefty):
    topleftx = int(topleftx)
    toplefty = int(toplefty)
    imArr = load_image(inf)
    imOut = imArr_compressed = np.zeros((480,640,3),dtype = imArr.dtype)
    for ch in range(3):
        try:
            imOut[:,:,ch] = imArr[toplefty:toplefty+480,topleftx:topleftx+640,ch]
        except:
            print("!!!Error, size too small or specified coordinates too large to crop to 640 * 480!!!")
            sys.exit()
    save_image(imArr_compressed, outf)
if __name__ == '__main__':
    crop640by480(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
