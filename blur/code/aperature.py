import os
import glob
import skimage.io as skio
import numpy as np


directory_path = "final_project/blur/images/rectified"

image_files = glob.glob(os.path.join(directory_path, '*.png'))  

avg_img = np.zeros_like(skio.imread(image_files[0]), dtype=np.float64)


pow2 = [256, 128, 64, 32, 16, 8, 4, 2, 1]




for n in pow2:
    avg_img = np.zeros_like(skio.imread(image_files[0]), dtype=np.float64)
    for i in range(0, n):

        

        image_file_list = image_files[i].split("_")
        y_coord = int(float(image_file_list[2]))
        x_coord = int(float(image_file_list[3]))
        
        y_coord = -(y_coord)
        x_coord = (x_coord)


        

        im = skio.imread(image_files[i]).astype(np.uint8)
        im = np.roll(im, shift=int(y_coord), axis=0)
        im = np.roll(im, shift=int(x_coord), axis=1)


        avg_img += np.array(im) / (len(image_files) / n)


    skio.imsave(f"ap_for_{int(n)}.png", avg_img)