import cv2
import numpy as np
from modules.matching import detectAndDescribe, matchKeypoints

###
# Purpose: Stiches an image onto the final panorama image (grayscale images). 
#          Does not blend the images (takes pixel value from result where the 
#          images) overlap
# Inputs: result - The final panorama image
#         image - The image to stitch
#         matchType - feature matching type, 0 = brute force 1 = k-nearest neighbours
# Returns: The result with the image stitched onto it
###
def __stitchGray(result, image, match_type):
    kpA, fA = detectAndDescribe(np.uint8(result))
    kpB, fB = detectAndDescribe(image)
    _, H, _, _ = matchKeypoints(kpA, kpB, fA, fB, match_type)

    imageWarped = cv2.warpPerspective(image, H, (result.shape[1], result.shape[0]))

    print(imageWarped.shape)
    print(result.shape)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if (imageWarped[i][j] != 0 and result[i][j] == 0):
                result[i][j] = imageWarped[i][j]
    
    return result
                

###
# Purpose: Stiches an image onto the final panorama image (colour images). Does
#          not blend the images (takes pixel value from result where the images)
#          overlap
# Inputs: result - The final panorama image
#         image - The image to stitch
#         match_type - feature matching type, 0 = brute force 1 = k-nearest neighbours
#         Returns: The result with the image stitched onto it
###
def __stitchColour(result, image, matchType):
    kpA, fA = detectAndDescribe(cv2.cvtColor(np.uint8(result), cv2.COLOR_RGB2GRAY))
    kpB, fB = detectAndDescribe(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    _, H, _, _ = matchKeypoints(kpA, kpB, fA, fB, matchType)

    imageWarped = cv2.warpPerspective(image, H, (result.shape[1], result.shape[0]))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                if (imageWarped[i][j][k] != 0 and result[i][j][k] == 0):
                    result[i][j][k] = imageWarped[i][j][k]     

    return result


def __stitchGrayAvg(result, image, match_type):
    kpA, fA = detectAndDescribe(np.uint8(result))
    kpB, fB = detectAndDescribe(image)
    _, H, _, _ = matchKeypoints(kpA, kpB, fA, fB, match_type)

    imageWarped = cv2.warpPerspective(image, H, (result.shape[1], result.shape[0]))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if (imageWarped[i][j] != 0):
                if (result[i][j] != 0):
                    result[i][j] = (imageWarped[i][j] + result[i][j]) // 2
                else:
                    result[i][j] = imageWarped[i][j]
            # Otherwise imageWarped[i][j] == 0 and result doesn't change

    return result


def __stitchColourAvg(result, image, matchType):
    kpA, fA = detectAndDescribe(cv2.cvtColor(np.uint8(result), cv2.COLOR_RGB2GRAY))
    kpB, fB = detectAndDescribe(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    _, H, _, _ = matchKeypoints(kpA, kpB, fA, fB, matchType)

    imageWarped = cv2.warpPerspective(image, H, (result.shape[1], result.shape[0]))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                if (imageWarped[i][j][k] != 0):
                    if (result[i][j][k] != 0):
                        result[i][j][k] = (imageWarped[i][j][k] + result[i][j][k]) // 2
                    else:
                        result[i][j][k] = imageWarped[i][j][k]  

    return result


def __leftEdges(result, imageWarped):
    edgePixels = {}

    n_rows = result.shape[0]
    n_cols = result.shape[1]

    for i in range(n_rows):
        for j in range(n_cols):
            # Find first pixel where images overlap
            edgePixels[f'{i}'] = [0, None]
            if(result[i][j] != 0 and imageWarped[i][j] != 0):
                if(j != 0):
                    # Pixel to left is all image
                    if(result[i][j-1] == 0):
                        edgePixels[f'{i}'] = [j, 'image']
                    # Pixel to left is all result
                    elif(imageWarped[i][j-1] == 0):
                        edgePixels[f'{i}'] = [j, 'result']
                break

    return edgePixels


def __rightEdges(result, imageWarped):
    edgePixels = {}

    n_rows = result.shape[0]
    n_cols = result.shape[1]

    for i in range(n_rows):
        for j in range(n_cols-1, -1, -1):
            # Find first pixel where images overlap
            edgePixels[f'{i}'] = [n_cols-1, None]
            if(result[i][j] != 0 and imageWarped[i][j] != 0):
                if(j != n_cols-1):
                    # Pixel to right is all image
                    if(result[i][j+1] == 0):
                        edgePixels[f'{i}'] = [j, 'image']
                    # Pixel to left is all result
                    elif(imageWarped[i][j+1] == 0):
                        edgePixels[f'{i}'] = [j, 'result']
                break

    return edgePixels


def __topEdges(result, imageWarped):
    edgePixels = {}

    n_rows = result.shape[0]
    n_cols = result.shape[1]

    for j in range(n_cols):
        for i in range(n_rows):
            # Find first pixel where images overlap
            edgePixels[f'{j}'] = [0, None]
            if(result[i][j] != 0 and imageWarped[i][j] != 0):
                if(i != 0):
                    # Pixel to right is all image
                    if(result[i-1][j] == 0):
                        edgePixels[f'{j}'] = [i, 'image']
                    # Pixel to left is all result
                    elif(imageWarped[i-1][j] == 0):
                        edgePixels[f'{j}'] = [i, 'result']
                break

    return edgePixels


def __bottomEdges(result, imageWarped):
    edgePixels = {}

    n_rows = result.shape[0]
    n_cols = result.shape[1]

    for j in range(n_cols):
        for i in range(n_rows-1, -1, -1):
            # Find first pixel where images overlap
            edgePixels[f'{j}'] = [n_rows-1, None]
            if(result[i][1] != 0 and imageWarped[i][j] != 0):
                if(i != n_rows-1):
                    # Pixel to right is all image
                    if(result[i+1][j] == 0):
                        edgePixels[f'{j}'] = [i, 'image']
                    # Pixel to left is all result
                    elif(imageWarped[i+1][j] == 0):
                        edgePixels[f'{j}'] = [i, 'result']
                break

    return edgePixels

def __weightPixel(resultPixel, imgPixel, startInfo, endInfo, location):
    start = startInfo[0]
    distance = endInfo[0] - startInfo[0]
    weight = (location - start) / distance

    # 8 cases - start is (result, image, or None) 
    #           end is (result, image, or None)
    #           start is None and end is None covered by checks in blendPixel

    # Case 1: start is result, end is image - begins all result, ends all image
    if (startInfo[1] == 'result' and endInfo[1] == 'image'):
        weightedPixel = ((1 - weight) * resultPixel) + (weight * imgPixel)

    # Case 2: start is image, end is result - opposite of case 1 
    elif (startInfo[1] == 'image' and endInfo[1] == 'result'):
        weightedPixel = ((1 - weight) * imgPixel) + (weight * resultPixel)

    # Case 3: start and end is result - all result at edges, 50/50 blend in middle
    elif (startInfo[1] == 'result' and endInfo[1] == 'result'):
        if weight >= 0.5:
            weightedPixel = (weight * resultPixel) + ((1 - weight) * imgPixel)
        else: 
            weightedPixel = ((1 - weight) * resultPixel) + (weight * imgPixel)

    # Case 4: start and end is image - opposite of case 3
    elif (startInfo[1] == 'image' and endInfo[1] == 'image'):
        if weight >= 0.5:
            weightedPixel = (weight * resultPixel) + ((1 - weight) * imgPixel)
        else: 
            weightedPixel = ((1 - weight) * resultPixel) + (weight * imgPixel)

    # Case 5: start is result and end is no image - all result at start, 50/50 blend at end
    elif (startInfo[1] == 'result' and endInfo[1] == None):
        weightedPixel = ((1 - (weight / 2)) *  resultPixel) + (weight / 2) * imgPixel

    elif (startInfo[1] == 'image' and endInfo[1] == None):
        weightedPixel = ((1 - (weight / 2)) *  imgPixel) + (weight / 2) * resultPixel   

    elif (startInfo[1] == None and endInfo[1] == 'result'):
        weightedPixel = ((1 - (weight / 2)) *  imgPixel) + (weight / 2) * resultPixel 

    elif (startInfo[1] == None and endInfo[1] == 'image'):
        weightedPixel = ((1 - (weight / 2)) *  resultPixel) + (weight / 2) * imgPixel    

    return int(weightedPixel)



def __blendPixel(resultPixel, imgPixel, leftInfo, rightInfo, topInfo, bottomInfo, row, col):
    if (topInfo[1] == None and bottomInfo[1] == None and leftInfo[1] == None and rightInfo[1] == None):
        blendedPixel == resultPixel + imgPixel // 2
    elif (topInfo[1] == None and bottomInfo[1] == None):
        blendedPixel = __weightPixel(resultPixel, imgPixel, leftInfo, rightInfo, col)
    elif (leftInfo[1] == None and rightInfo[1] == None):
        blendedPixel = __weightPixel(resultPixel, imgPixel, topInfo, bottomInfo, row)
    else:
        horizontalIntensity = __weightPixel(resultPixel, imgPixel, leftInfo, rightInfo, col)
        verticalIntensity = __weightPixel(resultPixel, imgPixel, topInfo, bottomInfo, row)
        blendedPixel = (horizontalIntensity + verticalIntensity) // 2
    return blendedPixel


def __stitchGrayWeighted(result, image, match_type):
    kpA, fA = detectAndDescribe(np.uint8(result))
    kpB, fB = detectAndDescribe(image)
    _, H, _, _ = matchKeypoints(kpA, kpB, fA, fB, match_type)

    imageWarped = cv2.warpPerspective(image, H, (result.shape[1], result.shape[0]))

    edgesLeft = __leftEdges(result, imageWarped)
    edgesRight = __rightEdges(result, imageWarped)
    edgesTop = __topEdges(result, imageWarped)
    edgesBottom = __bottomEdges(result, imageWarped)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if (imageWarped[i][j]):
                if (result[i][j] != 0):
                    leftEdgePixel = edgesLeft[f'{i}']
                    rightEdgePixel = edgesRight[f'{i}']
                    topEdgePixel = edgesTop[f'{j}']
                    bottomEdgePixel = edgesBottom[f'{j}']
                    result[i][j] = __blendPixel(result[i][j], imageWarped[i,j], 
                                                leftEdgePixel, rightEdgePixel,
                                                topEdgePixel, bottomEdgePixel,
                                                i, j)
                else:
                    result[i][j] = imageWarped[i][j]

    return result



###
# Purpose: Main function from this module. Stitches two images together. Calls 
#          one of the other functions to perform the stitching, depending on the 
#          colour type of the images and the blending method used
# Inputs: TODO 
# Returns: The stitched together result.
###
def stitch(result, image, colour_type, match_type=0, blend_type=0):
    if colour_type == 'rgb':
        if blend_type == 2:
            result = __stitchColour(result, image, match_type)
        elif blend_type == 1:
            result = __stitchColourAvg(result, image, match_type)
        else: 
            result = __stitchColour(result, image, match_type)
    else: 
        if blend_type == 2:
            result == __stitchGrayWeighted(result, image, match_type)
        elif blend_type == 1:
            result = __stitchGrayAvg(result, image, match_type)
        else: 
            result = __stitchGray(result, image, match_type)

    return result


if __name__ == "__main__":
    M1 = np.array(([[1,1,1,0,0],[0,0,0,1,1],[0,0,1,1,1]]))
    #M1 = [1 1 1 0 0
    #      0 0 0 1 1
    #      0 0 1 1 1]
    M2 = np.array(([[0,0,0,1,1],[0,0,1,1,1],[0,0,0,0,1]]))
    #M2 = [0 0 0 1 1 
    #      0 0 1 1 1
    #      0 0 0 0 1 ]
    
    #Expected: {0:[None,None], 1:[3,image], 2:[4,result]}
    print("Left Edges")
    print(__leftEdges(M1, M2))

    #Should be {0:[1, result], 1:[None, None], 2:[1, result], 3:[0, image], 4:[None,None]}
    