import numpy as np

###
# Purpose: Determine the details of a panorama layout
# Inputs: Layout - the layout graph, list of ordered pair tuples (row, column)
# Returns: width - number of images horizontally in the layout
#          height - number of images vertically in the layout
#          centerPoint - ordered pair representing index (row, column) of center image
###
def getLayoutDetails(layout):
    # Find max row and column index, use result to find centermost point
    height = 0
    width = 0
    for point in layout:
        if point[0] > height:
            height = point[0]
        if point[1] > width:
            width = point[1]
    centerPoint = (height // 2, width // 2)

    return (width, height, centerPoint)


###
# Purpose: Initializes the final panorama result canvas (a numpy array)
# Inputs: layout_h - the number of images vertically
#         layout_w - the number of images horizontally
#         image - An image in the panorama (used to determine overall height/width)
#         colour_type - 'rgb' for RGB, 'gray' for grayscale
# Returns: The result canvas, all entries initalized to zeroes
###
def initResult(layout_h, layout_w, image, colour_type):
    # Find height and width of one image, use to find overall h, w and initialize
    base_height = image.shape[0]
    base_width = image.shape[1]
    result_width = base_width * (layout_w + 1)
    result_height = base_height * (layout_h + 1)
    if (colour_type == 'rgb'):
        result = np.zeros((result_height, result_width, 3))
    else:
        result = np.zeros((result_height, result_width))
    return result


###
# Purpose: Place center image on the final panorama image canvas
# Inputs: result - The canvas
#         img_center - the center image
#         layout_put_c - the top-left corner point on the canvas of the center image 
#         colour_type - RGB or grayscale
# Returns: The canvas with the center image placed
###
def placeCenterImage(result, img_center, layout_pt_c, colour_type):
    start_row = img_center.shape[0] * layout_pt_c[0]
    end_row = img_center.shape[0] + start_row
    start_col = img_center.shape[1] * layout_pt_c[1]
    end_col = img_center.shape[1] + start_col
    if colour_type == "rgb":
        result[start_row:end_row, start_col:end_col, 0:3] = img_center
    else:
        result[start_row:end_row, start_col:end_col] = img_center
    return result