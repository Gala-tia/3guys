import cv2 as cv
import os
testA = 'D:\\JIT\\pycharm\\CODE\\CycleGAN-PyTorch\\datasets\\ukiyoe2photopregauss\\testA\\'
testB = 'D:\\JIT\\pycharm\\CODE\\CycleGAN-PyTorch\\datasets\\ukiyoe2photopregauss\\testB\\'
trainA = 'D:\\JIT\\pycharm\\CODE\\CycleGAN-PyTorch\\datasets\\ukiyoe2photopregauss\\trainA\\'
trainB = 'D:\\JIT\\pycharm\\CODE\\CycleGAN-PyTorch\\datasets\\ukiyoe2photopregauss\\trainB\\'

saveA = 'D:\\JIT\\pycharm\\CODE\\CycleGAN-PyTorch\\datasets\\ukiyoe2photogauss\\testA\\'
saveB = 'D:\\JIT\\pycharm\\CODE\\CycleGAN-PyTorch\\datasets\\ukiyoe2photogauss\\testB\\'
savetA = 'D:\\JIT\\pycharm\\CODE\\CycleGAN-PyTorch\\datasets\\ukiyoe2photogauss\\trainA\\'
savetB = 'D:\\JIT\\pycharm\\CODE\\CycleGAN-PyTorch\\datasets\\ukiyoe2photogauss\\trainB\\'
# for i, j, k in os.walk(testA):
#     # print(i)
#     # print(j)
#     # print(k)

from skimage import util
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

tA = os.listdir(trainA)
for i in tA:
    img1 = Image.open(trainA + i)
    img = np.array(img1)

    # noisy1 = util.random_noise(img, mode='gaussian', mean=0, var=0.01)
    # noisy2 = util.random_noise(img, mode='gaussian', mean=0.1, var=0.01)
    # noisy3 = util.random_noise(img, mode='gaussian', mean=0, var=0.2)
    noisy3 = util.random_noise(img, mode='gaussian', mean=0, var=1)
    plt.imsave(savetA + i, noisy3)

