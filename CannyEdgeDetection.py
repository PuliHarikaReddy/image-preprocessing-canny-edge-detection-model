import cv2
import sys
import scipy.ndimage as ndi
import scipy
import numpy as np
import math
from math import pi
import PIL
from PIL import Image
import scipy.misc
import imageio

# 1. Gaussian Blur
variance = float(input("Enter variance for smoothing"))
src = sys.argv[1]

img = Image.open(src).convert('L')                                          
img_array = np.array(img, dtype = float)                                 
gaussian_blur = ndi.filters.gaussian_filter(img_array, variance)        
imageio.imwrite('smooth.png', gaussian_blur)
 
# 2. Finding Gradient Intensity 
gradient = Image.new('L', img.size)                                      
grad_x = np.array(gradient, dtype = float)                        
grad_y = np.array(gradient, dtype = float)

sk_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
sk_y = [[-1,-2,-1],[0,0,0],[1,2,1]]

w = img.size[1]
h = img.size[0]

# Calculating Gradients Horizontally and Vertically

for x in range(1, w-1):
    for y in range(1, h-1):
        p_x = (sk_x[0][0] * gaussian_blur[x-1][y-1]) + (sk_x[0][1] * gaussian_blur[x][y-1]) + \
             (sk_x[0][2] * gaussian_blur[x+1][y-1]) + (sk_x[1][0] * gaussian_blur[x-1][y]) + \
             (sk_x[1][1] * gaussian_blur[x][y]) + (sk_x[1][2] * gaussian_blur[x+1][y]) + \
             (sk_x[2][0] * gaussian_blur[x-1][y+1]) + (sk_x[2][1] * gaussian_blur[x][y+1]) + \
             (sk_x[2][2] * gaussian_blur[x+1][y+1])

        p_y = (sk_y[0][0] * gaussian_blur[x-1][y-1]) + (sk_y[0][1] * gaussian_blur[x][y-1]) + \
             (sk_y[0][2] * gaussian_blur[x+1][y-1]) + (sk_y[1][0] * gaussian_blur[x-1][y]) + \
             (sk_y[1][1] * gaussian_blur[x][y]) + (sk_y[1][2] * gaussian_blur[x+1][y]) + \
             (sk_y[2][0] * gaussian_blur[x-1][y+1]) + (sk_y[2][1] * gaussian_blur[x][y+1]) + \
             (sk_y[2][2] * gaussian_blur[x+1][y+1])
        grad_x[x][y] = p_x
        grad_y[x][y] = p_y

s_h = np.hypot(grad_x, grad_y)
s_v = np.arctan2(grad_y, grad_x)

imageio.imwrite('gradx.png', s_h)
imageio.imwrite('grady.png', s_v)

# 3. Thinning the Edges
final = s_h.copy()
for x in range(1, w-1):
    for y in range(1, h-1):
        if s_v[x][y]==0:
            if (s_h[x][y]<=s_h[x][y+1]) or \
               (s_h[x][y]<=s_h[x][y-1]):
                final[x][y]=0
        elif s_v[x][y]==45:
            if (s_h[x][y]<=s_h[x-1][y+1]) or \
               (s_h[x][y]<=s_h[x+1][y-1]):
                final[x][y]=0
        elif s_v[x][y]==90:
            if (s_h[x][y]<=s_h[x+1][y]) or \
               (s_h[x][y]<=s_h[x-1][y]):
                final[x][y]=0
        else:
            if (s_h[x][y]<=s_h[x+1][y+1]) or \
               (s_h[x][y]<=s_h[x-1][y-1]):
                final[x][y]=0

imageio.imwrite('thin.png', final)

# 4. Finding Potential Edges
max_val = np.max(final)
#print(m)
min_threshold = float(input("enter minimum threshold")) * max_val
max_threshold = float(input("enter maximum threshold")) * max_val

weak = np.zeros((w, h))
strong = np.zeros((w, h))

for x in range(w):
    for y in range(h):
        if final[x][y]>=max_threshold:
            strong[x][y]=final[x][y]
        if final[x][y]>=min_threshold:
            weak[x][y]=final[x][y]
imageio.imwrite('weak.png', weak)
imageio.imwrite('strong.png', strong)

# 5. Final Detection
def final_detection(i, j):
    x = [-1, 0, 1, -1, 1, -1, 0, 1]
    y = [-1, -1, -1, 0, 0, 1, 1, 1]
    for k in range(8):
        if strong[i+x[k]][j+y[k]]==0 and weak[i+x[k]][j+y[k]]!=0:
            strong[i+x[k]][j+y[k]]=1
            final_detection(i+x[k], j+y[k])

for i in range(1, w-1):
    for j in range(1, h-1):
        if strong[i][j]:
            strong[i][j]=1
            final_detection(i, j)


imageio.imwrite('canny.png', strong)
