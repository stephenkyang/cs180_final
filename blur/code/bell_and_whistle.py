import os
import glob
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt


directory_path = "final_project/blur/images/rectified"

image_files = glob.glob(os.path.join(directory_path, '*.png'))  

beginning = skio.imread(image_files[0])

avg_img = np.zeros_like(beginning, dtype=np.float64)

im_shape = np.array(beginning).shape
print(im_shape)
y = im_shape[0]
x = im_shape[1]


dot_location = (np.random.randint(0, x + 1), np.random.randint(0, y + 1))
# dot_location = (1330, 25)
rand_x, rand_y = dot_location

# for beginning
plt.imshow(beginning)
plt.scatter(*dot_location, color='red', s=10) 
plt.savefig(f"{dot_location}_beginning.png")

for image_file in image_files:

    image_file_list = image_file.split("_")
    y_coord = int(float(image_file_list[2]))
    x_coord = int(float(image_file_list[3]))
    
    y_coord = -(y_coord) * (rand_y / y) * 3
    x_coord = (x_coord) * (rand_y / y) * 3

    im = skio.imread(image_file).astype(np.uint8)
    im = np.roll(im, shift=int(y_coord), axis=0)
    im = np.roll(im, shift=int(x_coord), axis=1)


    avg_img += np.array(im) / len(image_files)

avg_img = avg_img / avg_img.max()
plt.imshow(avg_img)
plt.scatter(*dot_location, color='red', s=10) 
plt.savefig(f"{dot_location}_blur.png")