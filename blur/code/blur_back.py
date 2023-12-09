import os
import glob
import skimage.io as skio
import numpy as np


directory_path = "final_project/blur/images/rectified"

image_files = glob.glob(os.path.join(directory_path, '*.png'))  

avg_img = np.zeros_like(skio.imread(image_files[0]), dtype=np.float64)


range_0_3 = np.array(list(range(0, 31))) / 10

for n in range_0_3: 
    avg_img = np.zeros_like(skio.imread(image_files[0]), dtype=np.float64)
    for image_file in image_files:



        image_file_list = image_file.split("_")
        y_coord = int(float(image_file_list[2]))
        x_coord = int(float(image_file_list[3]))
        
        y_coord = -(y_coord) * n
        x_coord = (x_coord) * n


        

        im = skio.imread(image_file).astype(np.uint8)
        im = np.roll(im, shift=int(y_coord), axis=0)
        im = np.roll(im, shift=int(x_coord), axis=1)


        avg_img += np.array(im) / len(image_files)


    skio.imsave(f"blur_for_{n}.png", avg_img)