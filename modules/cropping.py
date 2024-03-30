import cv2
import numpy as np


###
# Purpose: Determines the row/column index to crop the image at
# Inputs: image - the image array (2D only - grayscale)
#        idx - the current index to check 
#        delta - the change in index for the next image
#        direction - -1 for starting index, +1 for ending index
#        axis - 0 for x (horizontal/column), 1 for y (vertical/row)
# Return: the index to crop at
###
def findCropIdx(image, idx, delta, direction, axis):
    # Base Case - Change from last iteration was 1 or 0
    if delta == 1:
        return idx
    
    outsideImage = True
    # Iterate over rows/columns at index, look for any non-zero (non-black) 
    # values, which indicate image pixels in that row/column 
    if axis == 0:
        # for column in range(0,image.shape[1]):
        #     if image[idx, column] != 0:
        #         outsideImage = False
        #         break
        if np.any(image[idx, :]):
            outsideImage = False
    else:
        # for row in range(0,image.shape[0]):
        #     if image[row, idx] != 0:
        #         outsideImage = False
        #         break
        if np.any(image[:,idx]):
            outsideImage = False
        
    # If we are in image, keep moving in direction by delta to find outside img
    if not outsideImage:
        new_idx = idx + (direction * delta)
    # Otherwise, move in opposite direction to find image again
    else:
        new_idx = idx + (-direction * delta)
    new_delta = delta//2
    return (findCropIdx(image, new_idx, new_delta, direction, axis))
    
###
# Purpose: Automatically crops a panorama image to cut off the black areas 
#          outside the image
# Inputs: image - a image in numpy array form (RGB or grayscale)
# Returns: the cropped image
###         
def autoCropper(image):
    # Convert to grayscale if color (autocropper only works with grayscale)
    # Creates a copy so that final image can be in color still 
    if image.ndim == 3:
        crop_img = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2GRAY)
    else:
        crop_img = image 

    center_col = crop_img.shape[1]//2
    center_row = crop_img.shape[0]//2

    start_row = None
    end_row = None    
    start_col = None
    end_col = None

    # Test if initial and final rows/columns have image pixels
    # for column in range(0, crop_img.shape[1]):
    #     if crop_img[0, column] != 0:
    #         start_row = 0
    #     if crop_img[crop_img.shape[0]-1, column] != 0:
    #         end_row = crop_img.shape[0]-1
    # for row in range(0, crop_img.shape[0]):
    #     if crop_img[row, 0] != 0:
    #         start_col = 0
    #     if crop_img[row, crop_img.shape[1]-1] != 0:
    #         end_col = crop_img.shape[1]-1
    if np.any(crop_img[0, :]):
        start_row = 0
    if np.any(crop_img[crop_img.shape[0]-1, :]):
        end_row = crop_img.shape[0]-1   
    
    if np.any(crop_img[:, 0]):
        start_col = 0
    if np.any(crop_img[:, crop_img.shape[1]-1]):
        end_col = crop_img.shape[1]-1
    
    # Calculate first and last image rows/columns (if not initial and final)
    if start_row == None:   
        start_row = findCropIdx(image = crop_img, idx = center_row, 
                                delta = center_row//2, direction = -1, axis = 0)
    if end_row == None:
        end_row = findCropIdx(image = crop_img, idx = center_row, 
                              delta = center_row//2, direction = 1, axis = 0)
    
    if start_col == None:
        start_col = findCropIdx(image = crop_img, idx = center_col, 
                                delta = center_col//2, direction = -1, axis=1)
    if end_col == None: 
        end_col = findCropIdx(image = crop_img, idx = center_col, 
                              delta = center_col//2, direction = 1, axis=1)

    # Create cropped cropped image
    if image.ndim == 3:
        result = np.zeros((end_row-start_row+1, end_col-start_col+1, image.shape[2]))
        result[:,:,:] = image[start_row:end_row+1, start_col:end_col+1, :]
    else:
        result = np.zeros((end_row-start_row+1, end_col-start_col+1))
        result[:,:] = image[start_row:end_row+1, start_col:end_col+1]
    
    return result