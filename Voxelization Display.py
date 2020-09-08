import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D


rows = 80
columns = rows
layers = columns

voxel_result_path = r'1.mat'
origin_mat_with_missing_region_path = r'bed_0141_with_missing_region.mat'

mat = loadmat(voxel_result_path)['VoxelMat']
mat_origin = loadmat(origin_mat_with_missing_region_path)['VoxelMat']

# set Mask
mask_cropped = np.zeros((layers,rows,columns))
# print(origin_mask.shape)

count = 0
for row in range(rows//2,rows):
	for column in range(columns):
		for layer in range(layers//2,layers):
			mask_cropped[row][column][layer]=1
			count += 1
			
mat_inpainted = np.multiply(mat,mask_cropped)
mat_origin = np.multiply(mat_origin,1-mask_cropped)


# Set voxel space
x, y, z = np.indices((rows, columns, layers))


# Initial
voxel_mat = (x < 0) & (y < 0) & (z < 0)
voxel_inpainted = (x < 0) & (y < 0) & (z < 0)

MSE_Square = 0
for row in range(0,rows):
	for col in range(0,columns):
		for layer in range(0,layers):
			if mat_inpainted[layer][row][col]==1:	
				pixel_mat = (x>=layer) & (x<(layer+1)) & (y>=row) & (y<(row+1)) & (z>=col) & (z<(col+1))
				voxel_inpainted = voxel_inpainted | pixel_mat

			if mat_origin[layer][row][col]==1:
				pixel_mat = (x>=layer) & (x<(layer+1)) & (y>=row) & (y<(row+1)) & (z>=col) & (z<(col+1))
				voxel_mat = voxel_mat | pixel_mat		

all_pixels = voxel_mat | voxel_inpainted

# set the colors of each object
colors = np.empty(voxel_mat.shape, dtype=object)
colors[voxel_mat] = '#77889960'
colors[voxel_inpainted] = '#D3CCFFD0'

# Set canvas size
plt.figure('Voxelization',figsize=(12, 12))
ax = plt.gca(projection='3d')

# Hide coordinate axis
plt.axis('off')

ax.voxels(voxel_mat, facecolors=colors, edgecolor='#C9E9FFE0', linewidth=0.3)
ax.voxels(voxel_inpainted, facecolors=colors, edgecolor='#483D8BE0',linewidth=0.3)
ax.view_init(elev=-30, azim=-60)

print("Saving picture ……")
plt.savefig("3D-DCGAN.png")
print("Picture saved!")

# plt.show()
