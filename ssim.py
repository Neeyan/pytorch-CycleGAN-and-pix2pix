#conding utf-8
# -*- codingen: utf-8 -*-
#文章参考的博客
# https://www.runoob.com/python/python-func-open.html
# https://blog.csdn.net/weixin_42630613/article/details/106808632
import argparse
import glob
import os
import cv2
from skimage.measure import compare_psnr, compare_ssim

txt_file = open(r'/content/inference','a')

def calc_measures(hr_path, calc_psnr=True, calc_ssim=True):
	HR_files = glob.glob(hr_path + '/*')
	mean_psnr = 0
	mean_ssim = 0

    for file in HR_files:
        hr_img = cv2.imread(file)
        filename = file.rsplit('/', 1)[-1]
        path = os.path.join(args.inference_result, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError('')

        inf_img = cv2.imread(path)

        print('-' * 10)
        if calc_psnr:
            psnr = compare_psnr(hr_img, inf_img)
            print('{0} : PSNR {1:.3f} dB'.format(filename, psnr))
            mean_psnr += psnr
        if calc_ssim:
            ssim = compare_ssim(hr_img, inf_img, multichannel=True)  # 单个SSIM比较值
            print('{0} : SSIM {1:.3f}'.format(filename, ssim))
            mean_ssim += ssim
        txt_file.write('PSNR,{:.3f} , SSIM, {:.3f}'.format(psnr , ssim))
        txt_file.write('\n')

    print('-' * 10)
    if calc_psnr:
        M_psnr = mean_psnr / len(HR_files)
        print('mean-PSNR {:.3f} dB'.format(M_psnr))
    if calc_ssim:
        M_ssim = mean_ssim / len(HR_files)
        print('mean-SSIM {:.3f}'.format(M_ssim))
    txt_file.write('mean-PSNR, {:.3f} , mean-SSIM, {:.3f}'.format(M_psnr,M_ssim))
    txt_file.write('\n'*2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--HR_data_dir', default=r'/content/test_latest/images-real', type=str)  #原始图像路径
    parser.add_argument('--inference_result', default=r'/content/test_latest/image-fake', type=str)  #生成图像路径

    args = parser.parse_args()
    calc_measures(args.HR_data_dir, calc_psnr=True, calc_ssim=True)
