import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import time 

import torch
import cv2
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
#from models import Res_CE2P
#from dataset.randomcrop import LIPDataValSet
from tensorboardX import SummaryWriter
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage

import torch.nn as nn 
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
  
def compute_errors(gt, pred):
    
    min_depth = 1e-3
    max_depth = 80
    
    # print(np.min(gt))

    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth
    gt[gt < min_depth] = min_depth
    gt[gt > max_depth] = max_depth

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, 
                        help="Path to the directory containing the PASCAL VOC dataset.") 
    parser.add_argument("--data-list", type=str, 
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=(473,473),
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num-classes", type=int, default=20,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--restore-from", type=str, 
                        help="Where restore model parameters from.")
    parser.add_argument("--is-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--save-dir", type=str, 
                        help="Path to the output results.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    return parser.parse_args()

 
def scale_image(image, scale):  
    image = image[0,:,:,:]
    image = image.transpose((1, 2, 0)) 
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR) 
    image = image.transpose((2, 0, 1)) 
    return image 
  
def predict(net, image, output_size, is_mirror=True, scales=[1]): 
    if is_mirror:
        image_rev = image[:,:,:,::-1]

    interp = nn.Upsample(size=output_size, mode='nearest')

    outputs = []
    if is_mirror:
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
                image_rev_scale = scale_image(image=image_rev, scale=scale)
            else:
                image_scale = image[0,:,:,:]
                image_rev_scale = image_rev[0,:,:,:]

            image_scale = np.stack((image_scale,image_rev_scale))

            prediction,_ = net(Variable(torch.from_numpy(image_scale), volatile=True).cuda())
            prediction = interp(prediction).cpu().data.numpy()

            prediction_rev = prediction[1,:,:,:].copy()
            prediction_rev = prediction_rev[:,:,::-1]
            prediction = prediction[0,:,:,:]
            prediction = np.mean([prediction, prediction_rev], axis=0)

            outputs.append(prediction)

        outputs = np.mean(outputs, axis=0)
        outputs = outputs.transpose(1,2,0)  
    else:
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
            else:
                image_scale = image[0,:,:,:]

            prediction = net(Variable(torch.from_numpy(image_scale).unsqueeze(0), volatile=True).cuda())
            prediction = interp(prediction[1]).cpu().data.numpy()
            outputs.append(prediction[0,:,:,:])

        outputs = np.mean(outputs, axis=0)
        outputs = outputs.transpose(1,2,0)  

    return outputs
  

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def getConfusionMatrixPlot(conf_arr):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/max(1.0,float(a)))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width, height = conf_arr.shape
 

    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix.png', format='png')
    

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w) 
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    model = Res_CE2P(num_classes=args.num_classes)
       
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 
        
    restore_from  = args.restore_from 
    saved_state_dict = torch.load(restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    testloader = data.DataLoader(LIPDataValSet(args.data_dir,  args.data_list, crop_size=input_size, mean=IMG_MEAN), 
                                    batch_size=1, shuffle=False, pin_memory=True)
      
    errors = np.zeros(7)
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d images have been proceeded'%(index))
        image, label, ori_size, name = batch 
          
        ori_size = ori_size[0].numpy()
         
        output = predict(model, image.numpy(), (np.asscalar(ori_size[0]), np.asscalar(ori_size[1])), is_mirror=args.is_mirror, scales=[1])
        depth_pred = np.asarray(output.squeeze(), dtype=np.float32)

        scipy.misc.imsave(args.save_dir+name[0]+'.png',depth_pred)
        
        depth_gt = np.asarray(label[0].numpy(), dtype=np.float32)  
        ignore_index = depth_gt != 0  
        depth_gt = depth_gt[ignore_index]
        depth_pred = depth_pred[ignore_index]   
        errors  += compute_errors(depth_gt, depth_pred)
           

    errors = errors/len(testloader)
    print('abs_rel: %f \n'% errors[0])
    print('sq_rel: %f \n'% errors[1])
    print('rmse: %f \n'% errors[2]) 
    print('rmse_log: %f \n'% errors[3]) 
    print('a1: %f \n'% errors[4]) 
    print('a2: %f \n'% errors[5]) 
    print('a3: %f \n'% errors[6]) 
                
     
if __name__ == '__main__':
    main()


