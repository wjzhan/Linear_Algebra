from utils import *
import os

def main():
    print("images in folder")
    for imgf in os.listdir("imgs"):
        img_path = os.path.join("imgs",imgf)
        print("\t",img_path)
    img_path = 'imgs/Figure1.png'
    print("Loading",img_path)
    imArr = load_image(img_path)
    print("imArr size",imArr.shape)
    #ks = [5]
    ks = [1,5, 50, 150, 400, 600, 800, 1050, 1200]
    err = []
    for k in ks:
        print("Perform SVD for k=%d ..." % k)
        imArr_compressed = svd_compress(imArr, K=k)
        err += [approx_error(imArr, imArr_compressed)]
        save_image(imArr_compressed, 'imgs/result_{}.png'.format(k))
        print("err",err[-1])
    plot_curve(ks, err, show=False)

if __name__ == '__main__':
    main()
