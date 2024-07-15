import numpy as np
import random
import math
import matplotlib.pyplot as plt

import _heapq as heapq
import surface_crns.readers as readers
from surface_crns.simulators.event import Event
from surface_crns.options.option_processor import SurfaceCRNOptionParser
from surface_crns import ModSurfaceCRNQueueSimulator
from surface_crns import SurfaceCRNQueueSimulator
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter

def circlebox(l, w, r):
    gx = w - 2*r
    gy = l - 2*r
    im = Image.new('RGB', (w, l), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.ellipse(xy = (-r, -r, r, r),
                fill = (255, 255, 255, 255),
                outline = (255, 255, 255),
                width = 0)
    draw.ellipse(xy = (r + gx, -r, w + r, r),
                fill = (255, 255, 255, 255),
                outline = (255, 255, 255),
                width = 0)
    draw.ellipse(xy = (-r, r + gy, r, l + r),
                fill = (255, 255, 255, 255),
                outline = (255, 255, 255),
                width = 0)
    draw.ellipse(xy = (r + gx, r + gy, w + r, l + r),
                fill = (255, 255, 255, 255),
                outline = (255, 255, 255),
                width = 0)
    return im

test = 'grain contact right'

# mask = np.array([[1, 1, 1, 0, 0, 0],
#                [1, 1, 1, 0, 0, 0],
#                [1, 1, 1, 0, 0, 0],
#                [1, 1, 1, 1, 1, 1],
#                [1, 1, 1, 1, 1, 1],
#                [1, 1, 1, 1, 1, 1]])

# mask = np.transpose(mask)

# manifest_filename = "C:\\Users\\wlind\\Documents\\microSurfaceCRN\\surface_crns\\GH_test_manifest.txt"

if test == 'random':
    mask = np.ones((64, 64))

    mask[0:32, 32:64] = 0

    mask[0:32, 0:32] = 0.33

    mask[32:64, 0:32] = 0.66

    plt.subplots(1, 1, figsize=(10, 8))
    plt.axis('off')
    plt.imshow(mask, cmap='coolwarm')
    plt.colorbar()
    plt.show()

    mask = np.transpose(mask)

    manifest_filename = "GH_mask_test_manifest.txt"

    ModSurfaceCRNQueueSimulator.simulate_surface_crn(manifest_filename, maskarray = mask)

elif test == "dissolution":
    mask = np.ones((100, 100))

    mask[0:50, 0:50] = 0.1

    mask[50:100, 0:50] = 0.2

    mask[50:100, 50:100] = 0.3

    plt.subplots(1, 1, figsize=(10, 8))
    plt.axis('off')
    plt.imshow(mask, cmap='coolwarm')
    plt.colorbar()
    plt.show()

    mask = np.transpose(mask)

    manifest_filename = "GR_mask_unitcell_test_manifest.txt"

    ModSurfaceCRNQueueSimulator.simulate_surface_crn(manifest_filename, maskarray = mask)

elif test == "smart mask":
    im = circlebox(100, 100, 30)
    imgray = im.convert('L')
    imarray = np.array(imgray)
    imarray[imarray == 0] = 1
    imarray[imarray == 255] = 0
    imarray = imarray.astype(np.float32)
    imarray[0:50,0:50][imarray[0:50,0:50] == 0] = 0
    imarray[50:100,0:50][imarray[50:100,0:50] == 0] = 0.25
    imarray[50:100,50:100][imarray[50:100,50:100] == 0] = 0.5
    imarray[0:50,50:100][imarray[0:50,50:100] == 0] = 1.0

    # plt.subplots(1, 1, figsize=(10, 8))
    # plt.axis('off')
    # plt.imshow(imarray, cmap='coolwarm')
    # plt.colorbar()
    # plt.show()
    
    mask = np.transpose(imarray)

    manifest_filename = "GR_mask_unitcell_test_manifest.txt"

    ModSurfaceCRNQueueSimulator.simulate_surface_crn(manifest_filename, maskarray = mask)

elif test == "grain contact":
    im = Image.new('RGB', (200, 100), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.ellipse(xy = (24, 10, 109, 90),
                fill = (255, 255, 255, 255),
                outline = (255, 255, 255),
                width = 0)
    
    draw.ellipse(xy = (91, 10, 176, 90),
            fill = (255, 255, 255, 255),
            outline = (255, 255, 255),
            width = 0)
    
    draw.ellipse(xy = (96, 21, 115, 79),
        fill = (200, 200, 200),
        outline = (200, 200, 200),
        width = 0)
    
    draw.ellipse(xy = (85, 21, 104, 79),
        fill = (200, 200, 200),
        outline = (200, 200, 200),
        width = 0)
    
    draw.ellipse(xy = (91, 24, 109, 76),
            fill = (150, 150, 150),
            outline = (150, 150, 150),
            width = 0)
    
    draw.ellipse(xy = (96, 24, 104, 76),
        fill = (100, 100, 100),
        outline = (100, 100, 100),
        width = 0)
    
    draw.ellipse(xy = (99, 24, 101, 76),
    fill = (0, 0, 0),
    outline = (0, 0, 0),
    width = 0)

    imgray = im.convert('L')
    imarray = 1 - np.array(imgray)/255

    print(np.unique(imarray))

    #contactregion = imarray[21:79, 80:120]
    #blurred = gaussian_filter(contactregion, sigma=1.25)

    #imarray[21:79, 80:120] = blurred

    # plt.subplots(1, 1, figsize=(10, 8))
    # plt.axis('off')
    # plt.imshow(im, cmap='coolwarm')
    # plt.colorbar()
    # plt.show()

    mask = np.transpose(imarray)

    manifest_filename = "GR_graincontact_manifest.txt"

    ModSurfaceCRNQueueSimulator.simulate_surface_crn(manifest_filename, maskarray = mask)

elif test == "grain contact amp":
    im = Image.new('RGB', (200, 100), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.ellipse(xy = (24, 10, 109, 90),
                fill = (255, 255, 255, 255),
                outline = (255, 255, 255),
                width = 0)
    
    draw.ellipse(xy = (91, 10, 176, 90),
            fill = (255, 255, 255, 255),
            outline = (255, 255, 255),
            width = 0)
    
    draw.ellipse(xy = (96, 21, 115, 79),
        fill = (200, 200, 200),
        outline = (200, 200, 200),
        width = 0)
    
    draw.ellipse(xy = (85, 21, 104, 79),
        fill = (200, 200, 200),
        outline = (200, 200, 200),
        width = 0)
    
    draw.ellipse(xy = (91, 24, 109, 76),
            fill = (150, 150, 150),
            outline = (150, 150, 150),
            width = 0)
    
    draw.ellipse(xy = (96, 24, 104, 76),
        fill = (100, 100, 100),
        outline = (100, 100, 100),
        width = 0)
    
    draw.ellipse(xy = (99, 24, 101, 76),
    fill = (0, 0, 0),
    outline = (0, 0, 0),
    width = 0)

    imgray = im.convert('L')
    imarray = np.array(imgray)

    scaledarray = (imarray * 99 / 255) + 1

    print(np.unique(imarray))

    #contactregion = imarray[21:79, 80:120]
    #blurred = gaussian_filter(contactregion, sigma=1.25)

    #imarray[21:79, 80:120] = blurred

    plt.subplots(1, 1, figsize=(10, 8))
    plt.axis('off')
    plt.imshow(scaledarray, cmap='coolwarm')
    plt.colorbar()
    plt.show()

    mask = np.transpose(scaledarray)

    manifest_filename = "GR_graincontact_manifest.txt"

    #ModSurfaceCRNQueueSimulator.simulate_surface_crn(manifest_filename, maskarray = mask)

elif test == "grain contact left":
    im = Image.new('RGB', (200, 100), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.ellipse(xy = (24, 10, 109, 90),
                fill = (255, 255, 255, 255),
                outline = (255, 255, 255),
                width = 0)
    
    draw.ellipse(xy = (96, 21, 115, 79),
        fill = (200, 200, 200),
        outline = (200, 200, 200),
        width = 0)
    
    draw.ellipse(xy = (85, 21, 104, 79),
        fill = (200, 200, 200),
        outline = (200, 200, 200),
        width = 0)
    
    draw.ellipse(xy = (91, 24, 109, 76),
            fill = (150, 150, 150),
            outline = (150, 150, 150),
            width = 0)
    
    draw.ellipse(xy = (96, 24, 104, 76),
        fill = (100, 100, 100),
        outline = (100, 100, 100),
        width = 0)
    
    draw.ellipse(xy = (99, 24, 101, 76),
    fill = (0, 0, 0),
    outline = (0, 0, 0),
    width = 0)

    draw.rectangle(xy = (100, 24, 110, 76),
    fill = (0, 0, 0),
    outline = (0, 0, 0),
    width = 0)

    imgray = im.convert('L')
    imarray = 1 - np.array(imgray)/255

    print(np.unique(imarray))

    #contactregion = imarray[21:79, 80:120]
    #blurred = gaussian_filter(contactregion, sigma=1.25)

    #imarray[21:79, 80:120] = blurred

    # plt.subplots(1, 1, figsize=(10, 8))
    # plt.axis('off')
    # plt.imshow(im, cmap='coolwarm')
    # plt.colorbar()
    # plt.show()

    mask = np.transpose(imarray)

    manifest_filename = "GR_graincontact_left_manifest.txt"

    ModSurfaceCRNQueueSimulator.simulate_surface_crn(manifest_filename, maskarray = mask)

elif test == "grain contact right":
    im = Image.new('RGB', (200, 100), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.ellipse(xy = (91, 10, 176, 90),
            fill = (255, 255, 255, 255),
            outline = (255, 255, 255),
            width = 0)
    
    draw.ellipse(xy = (96, 21, 115, 79),
        fill = (200, 200, 200),
        outline = (200, 200, 200),
        width = 0)
    
    draw.ellipse(xy = (85, 21, 104, 79),
        fill = (200, 200, 200),
        outline = (200, 200, 200),
        width = 0)
    
    draw.ellipse(xy = (91, 24, 109, 76),
            fill = (150, 150, 150),
            outline = (150, 150, 150),
            width = 0)
    
    draw.ellipse(xy = (96, 24, 104, 76),
        fill = (100, 100, 100),
        outline = (100, 100, 100),
        width = 0)
    
    draw.ellipse(xy = (99, 24, 101, 76),
    fill = (0, 0, 0),
    outline = (0, 0, 0),
    width = 0)

    draw.rectangle(xy = (90, 24, 100, 76),
    fill = (0, 0, 0),
    outline = (0, 0, 0),
    width = 0)

    imgray = im.convert('L')
    imarray = 1 - np.array(imgray)/255

    print(np.unique(imarray))

    #contactregion = imarray[21:79, 80:120]
    #blurred = gaussian_filter(contactregion, sigma=1.25)

    #imarray[21:79, 80:120] = blurred

    # plt.subplots(1, 1, figsize=(10, 8))
    # plt.axis('off')
    # plt.imshow(im, cmap='coolwarm')
    # plt.colorbar()
    # plt.show()

    mask = np.transpose(imarray)

    manifest_filename = "GR_graincontact_right_manifest.txt"

    ModSurfaceCRNQueueSimulator.simulate_surface_crn(manifest_filename, maskarray = mask)

else:
    print("No test case selected")



#SurfaceCRNQueueSimulator.simulate_surface_crn(manifest_filename)

# print("Reading information from manifest file " + manifest_filename + "...")

# manifest_options = \
#                readers.manifest_readers.read_manifest(manifest_filename)

# print(np.shape(manifest_options['init_state']))

#manifest_options['mask_vals'] = mask

#opts = SurfaceCRNOptionParser(manifest_options)

#grid = opts.grid
#print(opts.__dir__())
#print(opts.mask)
#print(grid.__dir__())
#print(grid.grid[0][0].__dir__())
#print(grid.grid[0][0].position)