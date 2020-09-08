import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D


rows = 80
columns = rows
layers = columns


mat = loadmat(r'/Users/tiky.xu/Desktop/实验结果/Bed/BiGAN/1.mat')['VoxelMat']

# mat_origin = loadmat(r'/Users/tiky.xu/Desktop/实验结果/Airplane/airplane_0115.mat')['VoxelMat']
mat_origin = loadmat(r'/Users/tiky.xu/Desktop/实验结果/Bed/bed_0141.mat')['VoxelMat']
# mat_origin = loadmat(r'/Users/tiky.xu/Desktop/实验结果/Bottle/bottle_0085.mat')['VoxelMat']
# mat_origin = loadmat(r'/Users/tiky.xu/Desktop/实验结果/Cone/cone_0114.mat')['VoxelMat']


# 遮罩Mask
mask_cropped = np.zeros((layers,rows,columns))
# print(origin_mask.shape)


print("开始对Mask进行计数…………")
count = 0  #添加计数标记
for row in range(rows//2,rows):
	for column in range(columns):
		for layer in range(layers//2,layers):
			mask_cropped[row][column][layer]=1
			count += 1
print("Mask计数完毕！共计：", count, "个像素。")
mat_inpainted = np.multiply(mat,mask_cropped)
mat_origin = np.multiply(mat_origin,1-mask_cropped)


# 设置体素化空间
x, y, z = np.indices((rows, columns, layers))


# 初始化显示体素化的内容为空
voxel_mat = (x < 0) & (y < 0) & (z < 0)
voxel_inpainted = (x < 0) & (y < 0) & (z < 0)


print("开始添加体素化方块…………")
# 逐个添加体素化方块
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
print("体素化方块添加完毕！")
# all_pixels = voxel_mat | voxel_inpainted
# all_pixels = voxel_mat


# set the colors of each object
colors = np.empty(voxel_mat.shape, dtype=object)
colors[voxel_mat] = '#77889960'
colors[voxel_inpainted] = '#D3CCFFD0'


# 设置画板大小 1200x1200 Pixels
plt.figure('Voxelization',figsize=(12, 12))
ax = plt.gca(projection='3d')
# 设置坐标轴标题
# ax.set_xlabel("X:Rows")
# ax.set_ylabel("Y:Columns")
# ax.set_zlabel("Z:Layers")


# 隐藏坐标轴
plt.axis('off')

ax.voxels(voxel_mat, facecolors=colors, edgecolor='#C9E9FFE0', linewidth=0.3)
ax.voxels(voxel_inpainted, facecolors=colors, edgecolor='#483D8BE0',linewidth=0.3)
ax.view_init(elev=-30, azim=-60)
# ax.view_init(elev=-30, azim=-90)
# plt.savefig("Airplane_0115_DCGAN_Inpainted.png")

print("Saving picture ……")
# plt.savefig("Airplane_0115_BiGAN.png")
plt.savefig("Bed_0141_BiGAN_Brighter.png")
# plt.savefig("Bottle_0085_GAN.png")
# plt.savefig("Cone_0114_GAN.png")
print("Picture saved!")

# plt.show()