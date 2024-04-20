# The Panorammer  
An automatic Panorama stitching tool that orders inspired by the openCV Stitcher API.  

## Features:  
<ul>
  <li>Generate panorama from any number and orientation of images</li>
  <li>Makes grayscale or colour panoramas</li>
  <li>Automatically generates the layout of images in the panorama.</li>
  <li>Crops images</li>
  <li>Image blending</li>
</ul>  

## How To Use 
Import the panoram function from panorammer.py.  This function can be called to generate a panorama, it will return the panorama image in openCV numpy array format. This functions takes as parameters:
<ol>
  <li>images - a list of images (in openCV numpy array format)</li>
  <li>layout - a paired list to images, with image locations in the panorama as array index (row, column) starting from (0,0). Eg (0,1) is second image in first row, (3,2) is third image in fourth row. If no layout is passed, a layout will automatically be generated</li>
  <li>match_type - integer for match type to use in feature matching. 0 = brute force matcher, 1 = k-nearest-neighbours matcher</li>
  <li>blend_type - blending method to use. 0 = no blending, 1 = average pixel intensities in area of overlap, 2 = linear/bilinear blending</li>
</ol>

## Limitations
<ul>
  <li>All images must be roughty the same size (in pixels)</li>
  <li>All images must be same colour space</li>
  <li>Images must be roughly aligned horizontally and vertically for automatic layout generating to work. There also must be a top left image</li>
  <li>Runs very slow once more than 3 or 4 images are being stitched together</li>
</ul>

  
By Neal Hamacher and Conlan Myers
