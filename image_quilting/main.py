import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from random import random
import time
import utils
from utils import cut

def quilt_random(sample, out_size, patch_size):
    """
    Randomly samples square patches of size patchsize from sample in order to create an output image of size outsize.

    :param sample: numpy.ndarray   The image you read from sample directory
    :param out_size: int            The width of the square output image
    :param patch_size: int          The width of the square sample patch
    :return: numpy.ndarray
    """
    height, width, channels = sample.shape
    
    new_texture = np.zeros((height, width, channels), dtype = np.uint8)
    
    for y in range(0, out_size, patch_size):
        for x in range(0, out_size, patch_size):
            random_y = np.random.randint(0, height - patch_size + 1)
            random_x = np.random.randint(0, width - patch_size + 1)
            
            sampled_image = sample[random_y: random_y + patch_size, random_x: random_x + patch_size]
            new_texture[y: y + patch_size, x: x + patch_size] = sampled_image
        
    return new_texture
    
def ssd_patch(template, mask, input_image):
    template = template/1.0
    mask = mask/1.0
    input_image = input_image/1.0
    ssd_cost = np.zeros((input_image.shape[0], input_image.shape[1]))
    
    for i in range(3):
        mask_temp = mask[:, :, i]**2
        template_temp = template[:, :, i]**2
        input_temp = input_image[:, :, i]**2
        ssd_cost += np.sum(template_temp* mask_temp) - 2*cv2.filter2D(input_temp, ddepth = -1, kernel = mask_temp*template_temp) + cv2.filter2D(input_temp**2, ddepth = -1, kernel = mask_temp)
    return ssd_cost
    
def choose_sample(cost_image, tol):
    flat_costs = cost_image.flatten()
    
    sorted_indices = np.argsort(flat_costs)
    chosen_index = np.random.choice(sorted_indices[:tol])
    chosen_y, chosen_x = np.unravel_index(chosen_index, cost_image.shape)
    return chosen_y, chosen_x
    
def quilt_simple(sample, out_size, patch_size, overlap, tol):
    """
    Randomly samples square patches of size patchsize from sample in order to create an output image of size outsize.
    Feel free to add function parameters
    :param sample: numpy.ndarray
    :param out_size: int
    :param patch_size: int
    :param overlap: int
    :param tol: float)
    :return: numpy.ndarray
    """
    
    height, width, channel = sample.shape
    
    step_size = patch_size - overlap
    
    new_texture = np.zeros((out_size, out_size, channel), dtype = np.uint8)
    
    template = np.zeros((patch_size, patch_size, channel), dtype = np.uint8)
    
    for y in range(0, out_size, step_size):
        if(y + patch_size >= out_size):
            break
        for x in range(0, out_size, step_size):
            if(x == 0 and y == 0):
                rand_y = np.random.randint(0, height - patch_size + 1)
                rand_x = np.random.randint(0, width - patch_size + 1)
                new_texture[y:y+patch_size, x:x+patch_size] = sample[rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
            else:
                if(x + patch_size >= out_size):
                    continue
                template = new_texture[y:y+patch_size, x:x+patch_size]
                mask = np.zeros_like(template)
                if(x != 0):
                    mask[:, :overlap] = 1
                if(y != 0):
                    mask[:overlap, :] = 1
                
                cost_image = ssd_patch(template, mask, sample)
                
                chosen_y, chosen_x = choose_sample(cost_image[patch_size//2 + 1:width - patch_size //2 - 1, patch_size//2 + 1:height - patch_size //2 - 1], tol)
            
                new_texture[y:y+patch_size, x:x+patch_size] = sample[chosen_y: chosen_y + patch_size, chosen_x : chosen_x + patch_size]
        
        
    return new_texture

def BND(patch, overlap):
    BND = np.sum((overlap - patch)**2, axis = 2)
    return BND


def quilt_cut(sample, out_size, patch_size, overlap, tol):
    """
    Samples square patches of size patchsize from sample using seam finding in order to create an output image of size outsize.
    Feel free to add function parameters
    :param sample: numpy.ndarray
    :param out_size: int
    :param patch_size: int
    :param overlap: int
    :param tol: float
    :return: numpy.ndarray
    """
    height, width, channel = sample.shape
    
    step_size = patch_size - overlap
    
    new_texture = np.zeros((height, width, channel), dtype = np.uint8)
    
    template = np.zeros((patch_size, patch_size, channel), dtype = np.uint8)
    
    for y in range(0, out_size, step_size):
        if(y + patch_size >= height):
            break
        for x in range(0, out_size, step_size):
            if(x == 0 and y == 0):
                rand_y = np.random.randint(0, height - patch_size + 1)
                rand_x = np.random.randint(0, width - patch_size + 1)
                new_texture[y:y+patch_size, x:x+patch_size] = sample[rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
            else:
                if(x + patch_size >= width):
                    continue
                template = new_texture[y:y+patch_size, x:x+patch_size]
                mask = np.zeros_like(template)
                
                if(x != 0):
                    mask[:, :overlap] = 1
                if(y != 0):
                    mask[:overlap, :] = 1
                
                cost_image = ssd_patch(template, mask, sample)
                
                chosen_y, chosen_x = choose_sample(cost_image[patch_size//2 + 1:width - patch_size //2 - 1, patch_size//2 + 1:height - patch_size //2 - 1], tol)
            
                temp_texture = sample[chosen_y: chosen_y + patch_size, chosen_x : chosen_x + patch_size]
                
                mask_seam_horizontal = np.ones((patch_size, patch_size), dtype = bool)
                mask_seam_vertical = np.ones((patch_size, patch_size), dtype = bool)
                
                
                horizontal_overlap = temp_texture[:overlap, :]
                horizontal_patch = new_texture[y:y + overlap, x: x + patch_size]
                if(horizontal_overlap.shape != horizontal_patch.shape):
                    horizontal_overlap = horizontal_patch[:horizontal_overlap.shape[0], :horizontal_overlap.shape[1]]
                bnd_horizontal = BND(horizontal_overlap, horizontal_patch)
                horizontal_mask = cut(bnd_horizontal.T).T
                vertical_overlap = temp_texture[:, :overlap]
                vertical_patch = new_texture[y:y + patch_size, x: x + overlap]
                if(vertical_overlap.shape != vertical_patch.shape):
                    vertical_overlap = vertical_patch[:vertical_overlap.shape[0], :vertical_overlap.shape[1]]
                bnd_vertical = BND(vertical_overlap, vertical_patch)
                vertical_mask = cut(bnd_vertical)
                
                mask_seam_horizontal[:horizontal_mask.shape[0], :horizontal_mask.shape[1]] = horizontal_mask
                mask_seam_vertical[:vertical_mask.shape[0], :vertical_mask.shape[1]] = vertical_mask
                
                mask_seam = np.logical_and(mask_seam_horizontal, mask_seam_vertical)
                
                new_texture[y:y+patch_size, x:x+patch_size] = temp_texture[:mask_seam.shape[0], :mask_seam.shape[1]]
        
    return new_texture

def texture_transfer(sample, patch_size, overlap, tol, guidance_im, alpha):
    """
    Samples square patches of size patchsize from sample using seam finding in order to create an output image of size outsize.
    Feel free to modify function parameters
    :param sample: numpy.ndarray
    :param patch_size: int
    :param overlap: int
    :param tol: float
    :param guidance_im: target overall appearance for the output
    :param alpha: float 0-1 for strength of target
    :return: numpy.ndarray
    """
    height, width, channel = sample.shape
    
    step_size = patch_size - overlap
    
    new_texture = np.zeros((height, width, channel), dtype = np.uint8)
    
    template = np.zeros((patch_size, patch_size, channel), dtype = np.uint8)
    
    for y in range(0, out_size, step_size):
        if(y + patch_size >= height or y + patch_size >= guidance_im.shape[0]):
            break
        for x in range(0, out_size, step_size):
            if(x == 0 and y == 0):
                rand_y = np.random.randint(0, height - patch_size + 1)
                rand_x = np.random.randint(0, width - patch_size + 1)
                new_texture[y:y+patch_size, x:x+patch_size] = sample[rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
            else:
                if(x + patch_size >= width or x + patch_size >= guidance_im.shape[1]):
                    continue
                template = new_texture[y:y+patch_size, x:x+patch_size]
                target = guidance_im[y: y + patch_size, x: x + patch_size]
                mask = np.zeros_like(template)
                
                if(x != 0):
                    mask[:, :overlap] = 1
                if(y != 0):
                    mask[:overlap, :] = 1
                
                cost_image1 = ssd_patch(template, mask, sample)
                cost_image2 = ssd_patch(target, mask, sample)
                
                cost_image = cost_image1*alpha + cost_image2*(1 - alpha)
                
                chosen_y, chosen_x = choose_sample(cost_image[patch_size//2 + 1:width - patch_size //2 - 1, patch_size//2 + 1:height - patch_size //2 - 1], tol)
            
                temp_texture = sample[chosen_y: chosen_y + patch_size, chosen_x : chosen_x + patch_size]
                
                mask_seam_horizontal = np.ones((patch_size, patch_size), dtype = bool)
                mask_seam_vertical = np.ones((patch_size, patch_size), dtype = bool)
                
                
                horizontal_overlap = temp_texture[:overlap, :]
                horizontal_patch = new_texture[y:y + overlap, x: x + patch_size]
                if(horizontal_overlap.shape != horizontal_patch.shape):
                    horizontal_overlap = horizontal_patch[:horizontal_overlap.shape[0], :horizontal_overlap.shape[1]]
                bnd_horizontal = BND(horizontal_overlap, horizontal_patch)
                horizontal_mask = cut(bnd_horizontal.T).T
                vertical_overlap = temp_texture[:, :overlap]
                vertical_patch = new_texture[y:y + patch_size, x: x + overlap]
                if(vertical_overlap.shape != vertical_patch.shape):
                    vertical_patch = vertical_overlap[:vertical_patch.shape[0], :vertical_patch.shape[1]]
                bnd_vertical = BND(vertical_overlap, vertical_patch)
                vertical_mask = cut(bnd_vertical)
                
                mask_seam_horizontal[:horizontal_mask.shape[0], :horizontal_mask.shape[1]] = horizontal_mask
                mask_seam_vertical[:vertical_mask.shape[0], :vertical_mask.shape[1]] = vertical_mask
                
                mask_seam = np.logical_and(mask_seam_horizontal, mask_seam_vertical)
                if(temp_texture.shape == (25, 25, 3)):
                    new_texture[y:y+patch_size, x:x+patch_size] = temp_texture[:mask_seam.shape[0], :mask_seam.shape[1]]
        
    return new_texture
    

# sample_img_fn = 'samples/bricks_small.jpg' # feel free to change
# sample_img = cv2.cvtColor(cv2.imread(sample_img_fn), cv2.COLOR_BGR2RGB)

# out_size = 200  # change these parameters as needed
# patch_size = 15 
# res = quilt_random(sample_img, out_size, patch_size)
# if res is not None:
#     plt.imshow(res)
#     plt.show()

# out_size = 300  # change these parameters as needed
# patch_size = 25
# overlap = 11
# tol = 5
# res = quilt_simple(sample_img, out_size, patch_size, overlap, tol)
# if res is not None:
#     plt.figure(figsize=(10,10))
#     plt.imshow(res)
#     plt.show()

# out_size = 300  # change these parameters as needed
# patch_size = 25
# overlap = 11
# tol = 5
# res = quilt_cut(sample_img, out_size, patch_size, overlap, tol)
# if res is not None:
#     plt.figure(figsize=(15,15))
#     plt.imshow(res)
#     plt.show()

sample_img_fn = 'samples/feynman.tiff' # feel free to change
texture_img = cv2.cvtColor(cv2.imread(sample_img_fn), cv2.COLOR_BGR2RGB)
sample_img_fn = 'samples/toast.jpg' # feel free to change
guidance_img = cv2.cvtColor(cv2.imread(sample_img_fn), cv2.COLOR_BGR2RGB)
patch_size = 25
overlap = 11
tol = 3
alpha = 0.5
out_size = 300
res = texture_transfer(texture_img, patch_size, overlap, tol, guidance_img, alpha)

plt.figure(figsize=(15,15))
plt.imshow(res)
plt.show()
