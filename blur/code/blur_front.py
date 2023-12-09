import os
import glob
import skimage.io as skio
import numpy as np


directory_path = "final_project/blur/images/rectified"

image_files = glob.glob(os.path.join(directory_path, '*.png'))  

avg_img = np.array(skio.imread(image_files[0])).astype(np.uint8) / len(image_files)


for image_file in image_files[1:]:
    im = skio.imread(image_file).astype(np.uint8)
    avg_img += np.array(im) / len(image_files)


skio.imsave("blur_front.png", avg_img)