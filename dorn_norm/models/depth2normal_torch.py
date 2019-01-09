#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np
import time
##input depth(array H*W)，intrnsics(list[fx,fy,cx,cy])  output: norm(3*h*w)
fx = 5.1885790117450188e+02
fy = 5.1946961112127485e+02
cx = 3.2558244941119034e+02
cy = 2.5373616633400465e+02
def depth2normal_layer(depth_map, intrinsics, inverse, nei=5):
    '''
    Args:
    depth_map: tensor of size (HxW)
    intrinsic: list of parameters [fx, fy, cx, cy]
    inverse: flag
    nei: stride of neighborhood, default=5
    '''
    ## mask is used to filter the background with infinite depth
    mask = depth_map > 0 

    if inverse:
        mask_clip = 1e-8 * (1.0-mask.float()) ## Add black pixels (depth = infinite) with delta
        depth_map += mask_clip
        depth_map = 1.0/depth_map ## inverse depth map
    kitti_shape = depth_map.shape
    pts_3d_map = compute_3dpts(depth_map, intrinsics)
    
    ## shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[nei:-nei, nei:-nei, :]
    pts_3d_map_x0 = pts_3d_map[nei:-nei, 0:-(2*nei), :]
    pts_3d_map_y0 = pts_3d_map[0:-(2*nei), nei:-nei, :]
    pts_3d_map_x1 = pts_3d_map[nei:-nei, 2*nei:, :]
    pts_3d_map_y1 = pts_3d_map[2*nei:, nei:-nei, :]
    pts_3d_map_x0y0 = pts_3d_map[0:-(2*nei), 0:-(2*nei), :]
    pts_3d_map_x0y1 = pts_3d_map[2*nei:, 0:-(2*nei), :]
    pts_3d_map_x1y0 = pts_3d_map[0:-(2*nei), 2*nei:, :]
    pts_3d_map_x1y1 = pts_3d_map[2*nei:, 2*nei:, :]

    ## generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    ## flatten the diff to a #pixle by 3 matrix
    pix_num = (kitti_shape[0]-2*nei) * (kitti_shape[1]-2*nei)
    diff_x0 = diff_x0.view( [pix_num, 3])
    diff_y0 = diff_y0.view( [pix_num, 3])
    diff_x1 = diff_x1.view( [pix_num, 3])
    diff_y1 = diff_y1.view( [pix_num, 3])
    diff_x0y0 = diff_x0y0.view( [pix_num, 3])
    diff_x0y1 = diff_x0y1.view( [pix_num, 3])
    diff_x1y0 = diff_x1y0.view( [pix_num, 3])
    diff_x1y1 = diff_x1y1.view( [pix_num, 3])
    
    normals0 = normalize_l2(torch.cross(diff_y1, diff_x1))
    normals1 = normalize_l2(torch.cross(diff_y0, diff_x0))
    normals2 = normalize_l2(torch.cross(diff_x0y0, diff_x0y1))
    normals3 = normalize_l2(torch.cross(diff_x1y1, diff_x1y0))
    
    normal_vector = torch.sum(torch.stack([normals0, normals1, normals2, normals3], 0),0)
    normal_vector = normalize_l2(normal_vector)
    normal_map = torch.squeeze(normal_vector).view( [kitti_shape[0]-2*nei,kitti_shape[1]-2*nei,3])
    
    normal_map *= torch.unsqueeze(mask[nei:-nei, nei:-nei].float(), 2)
    normal_map = torch.nn.functional.pad(normal_map, [0,0,nei,nei,nei,nei] ,"constant",0)

    return normal_map

def depth2normal_layer_batch(depth_map, intrinsics=[fx,fy,cx,cy], scale=16,inverse=False, nei=3):
    '''
    Args:
    depth_map: tensor of size (HxW)
    intrinsic: list of parameters [fx, fy, cx, cy]
    inverse: flag
    nei: stride of neighborhood, default=5
    '''
    intrinsics[0] = fx/scale
    intrinsics[1] = fy/scale
    intrinsics[2] = cx/scale
    intrinsics[3] = cy/scale
    depth_map=torch.squeeze(depth_map)
    #intrinsics=[fx,fy,cx,cy]
    ## mask is used to filter the background with infinite depth
    mask = depth_map > 0 

    if inverse:
        mask_clip = 1e-8 * (1.0-mask.float()) ## Add black pixels (depth = infinite) with delta
        depth_map += mask_clip
        depth_map = 1.0/depth_map ## inverse depth map
    kitti_shape = depth_map.shape
    pts_3d_map = compute_3dpts_batch(depth_map, intrinsics)


    ## shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[:,nei:-nei, nei:-nei, :]
    pts_3d_map_x0 = pts_3d_map[:,nei:-nei, 0:-(2*nei), :]
    pts_3d_map_y0 = pts_3d_map[:,0:-(2*nei), nei:-nei, :]
    pts_3d_map_x1 = pts_3d_map[:,nei:-nei, 2*nei:, :]
    pts_3d_map_y1 = pts_3d_map[:,2*nei:, nei:-nei, :]
    pts_3d_map_x0y0 = pts_3d_map[:,0:-(2*nei), 0:-(2*nei), :]
    pts_3d_map_x0y1 = pts_3d_map[:,2*nei:, 0:-(2*nei), :]
    pts_3d_map_x1y0 = pts_3d_map[:,0:-(2*nei), 2*nei:, :]
    pts_3d_map_x1y1 = pts_3d_map[:,2*nei:, 2*nei:, :]

    ## generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    ## flatten the diff to a #pixle by 3 matrix
    pix_num = kitti_shape[0] * (kitti_shape[1]-2*nei) * (kitti_shape[2]-2*nei)
    diff_x0 =diff_x0.view( [pix_num, 3])
    diff_y0 =diff_y0.view( [pix_num, 3])
    diff_x1 =diff_x1.view( [pix_num, 3])
    diff_y1 =diff_y1.view( [pix_num, 3])
    diff_x0y0 =diff_x0y0.view( [pix_num, 3])
    diff_x0y1 =diff_x0y1.view( [pix_num, 3])
    diff_x1y0 =diff_x1y0.view( [pix_num, 3])
    diff_x1y1 =diff_x1y1.view( [pix_num, 3])

    ## calculate normal by cross product of two vectors
    normals0 = normalize_l2(torch.cross(diff_y1, diff_x1)) #* tf.tile(normals0_mask[:, None], [1,3])
    normals1 = normalize_l2(torch.cross(diff_y0, diff_x0)) #* tf.tile(normals1_mask[:, None], [1,3])
    normals2 = normalize_l2(torch.cross(diff_x0y0, diff_x0y1)) #* tf.tile(normals2_mask[:, None], [1,3])
    normals3 = normalize_l2(torch.cross(diff_x1y1, diff_x1y0)) #* tf.tile(normals3_mask[:, None], [1,3])
    
    normal_vector = torch.sum(torch.stack([normals0, normals1, normals2, normals3], 0),0)
    normal_vector = normalize_l2(normal_vector)
    normal_map = torch.squeeze(normal_vector).view([kitti_shape[0],kitti_shape[1]-2*nei,kitti_shape[2]-2*nei,3])

    normal_map *= torch.unsqueeze(mask[:,nei:-nei, nei:-nei].float(), 3)
    normal_map = torch.nn.functional.pad(normal_map, [0,0,nei,nei,nei,nei] ,"constant",0)
    normal_map = torch.transpose(normal_map,1,3)
    normal_map = torch.transpose(normal_map,2,3)
    return normal_map 

def compute_3dpts(pts, intrinsics):

    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    
    shape = list(pts.shape)
    pts_z = pts
    x = torch.arange(0, shape[1]).float()
    y = torch.arange(0, shape[0]).float()
    
    pts_x = (torch.meshgrid([ y, x])[0] - cx) / fx * pts
    pts_y = (torch.meshgrid([ y, x])[1] - cy) / fy * pts
    pts_3d = torch.stack([pts_x, pts_y, pts_z], 2)

    return pts_3d

def compute_3dpts_batch(pts, intrinsics):
    
    ## pts is the depth map of rank3 [batch, h, w], intrinsics is in [batch, 4]
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3] 
    shape = list(pts.shape)
    pts_z = pts
    x = torch.arange(0, shape[2]).float().cuda()
    y = torch.arange(0, shape[1]).float().cuda()
    pts_x = (torch.meshgrid([ y, x])[0] - cx) / fx * pts
    pts_y = (torch.meshgrid([ y, x])[1] - cy) / fy * pts
    #pts_x = pts_x.expand(shape[0], pts_x.shape[0], pts_x.shape[1])
    #pts_y = pts_y.expand(shape[0], pts_y.shape[0], pts_y.shape[1])
    pts_x = pts_x.expand(shape[0], pts_x.shape[1], pts_x.shape[2])
    pts_y = pts_y.expand(shape[0], pts_y.shape[1], pts_y.shape[2])
    pts_3d = torch.stack([pts_x, pts_y, pts_z], 3)


    return pts_3d

def normalize_l2(vector):
    return torch.nn.functional.normalize(vector, 1)
