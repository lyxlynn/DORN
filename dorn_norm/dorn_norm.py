import argparse 
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import math 
from dataset import randomcrop
import random
import timeit
import criteria
import DORN_nyu
import logging
from tensorboardX import SummaryWriter
from models import get_segmentation_model
import models.depth2normal_torch as d2n
#from models.models_base import resnet101_base
#from dataset.datasets import LIPParsingEdgeDataSet,LIPDataValSet
from dataset.normdatasets import LIPParsingEdgeDataSet,LIPDataValSet
from utils.utils import decode_labels, inv_preprocess, decode_predictions,decode_norms
from utils.criterion import CriterionL1Loss,CriterionL2Loss 
##from utils.encoding import SelfDataParallel, ModelDataParallel, CriterionDataParallel
from numpy.distutils.tests.test_exec_command import emulate_nonposix
from evaluate import * 
start = timeit.default_timer()
import matplotlib.pyplot as plt
cmap = plt.cm.jet
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 40
TEST_DIRECTORY = '/data/liuyx/nyuv2/nyu_depth_v2_labeled'
TEST_LIST_PATH = './dataset/list/nyuv2/test.txt'
DATA_DIRECTORY = '/home/amax/LIP/LIP_data_model/data/resize473/split'
DATA_LIST_PATH = './dataset/list/lip/trainList_split.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '473, 473'
LEARNING_RATE = 7e-4
MOMENTUM = 0.9
NUM_CLASSES = 20
NUM_STEPS = 300000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_init.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
GPU_DEVICES = '0,1,2,3,4' 
SAVE_DIR = 'outputs_val/'
max_a1=0
def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--network", type=str,
                        help="choose which model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--test-dir", type=str, default=TEST_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--test-list", type=str, default=TEST_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save results of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--norm-weight", type=float, default=0.1,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default=GPU_DEVICES,
                        help="choose gpu device.")   
    return parser.parse_args()

args = get_arguments()
def colored_depthmap(depth):
    d_min=0
    d_max=10
    depth_relative =(depth-d_min)/(d_max-d_min)
    return 255*cmap(depth_relative)[:,:,:3]
def eval(model, testloader):
    global max_a1
    """Create the model and start the evaluation process."""
    args = get_arguments()
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w) 
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu[0]

    #model = Res_CE2P(num_classes=args.num_classes)
 #   model = resnet101_base(num_classes=args.num_classes)
       
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 
        
    
    #saved_state_dict = torch.load(restore_from)
#    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    #testloader = data.DataLoader(LIPDataValSet(args.test_dir,  args.test_list, crop_size=input_size, mean=IMG_MEAN), 
     #                               batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
      
    errors = np.zeros(7)
    print(len(testloader))
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d images have been proceeded'%(index))
        image, label, ori_size, name = batch 
          
        ori_size = ori_size[0].numpy()
         
        depth_gt = np.asarray(label[0].numpy(), dtype=np.float32)  
        output = predict(model, image.numpy(), (depth_gt.shape[0], depth_gt.shape[1]),  scales=[1])
        
        output = randomcrop.get_depth_sid(torch.from_numpy(output).cuda())
        depth_pred = np.asarray(output.squeeze(), dtype=np.float32)

        scipy.misc.imsave(args.save_dir+name[0]+'.png',depth_pred)
        #depth_gt = np.asarray(label[0].numpy(), dtype=np.float32)  
        ignore_index = depth_gt != 0 
        depth_gt = depth_gt[ignore_index]
        depth_pred = depth_pred[ignore_index] 
        ##log
        #depth_pred = np.exp(depth_pred)-1
        errors  += compute_errors(depth_gt, depth_pred)
           

    errors = errors/len(testloader)
    print('abs_rel: %f \n'% errors[0])
    print('sq_rel: %f \n'% errors[1])
    print('rmse: %f \n'% errors[2]) 
    print('rmse_log: %f \n'% errors[3]) 
    print('a1: %f \n'% errors[4]) 
    print('a2: %f \n'% errors[5]) 
    print('a3: %f \n'% errors[6]) 
    if errors[4]>max_a1:
        max_a1=errors[4]
    print('max_a1:',max_a1)
    return errors
                
          
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
 
def lr_step(base_lr, iter, step):
    return base_lr*((0.5)**(float(iter)//step))
def adjust_learning_rate(optimizer, i_iter): 
     
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)     
#    lr = lr_step(args.learning_rate, i_iter, 5000)     
        
    optimizer.param_groups[0]['lr'] = lr
    return lr

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.evar()

def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003

def main():
    """Create the model and start the training."""
    writer = SummaryWriter(args.snapshot_dir)
    global max_a1
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
 
    #deeplab = Res_CE2P(num_classes=args.num_classes) 
   # deeplab = resnet101_base(num_classes=args.num_classes)
    deeplab = get_segmentation_model(args.network,NoLabels=args.num_classes) 

   # deeplab = DORN_nyu.DORN()
    saved_state_dict = torch.load(args.restore_from)
    new_params = deeplab.state_dict().copy()
    #for i in saved_state_dict:
    #        i_parts = i.split('.')
    #        if not i_parts[0]=='fc' and not  i_parts[0]=='last_linear' and not  i_parts[0]=='classifier':
    #          new_params['.'.join(i_parts[0:])] = saved_state_dict[i]  
    if args.start_iters > 0:
        deeplab.load_state_dict(saved_state_dict)
    else:
        for i in saved_state_dict:
         #Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            #if i_parts[1]=='layer5':
            #    saved_state_dict[i] = deeplab.state_dict()[i,] 
        deeplab.load_state_dict(saved_state_dict, strict = False)
    model = nn.DataParallel(deeplab) 
    model.train()
    model.float() 
    model.cuda()    
 
    criterion = criteria.ordLoss_norm()
   # criterion = CriterionL2Loss()
   # criterion = CriterionDataParallel(criterion)
    criterion.cuda()
    
    cudnn.benchmark = True

    interp = nn.Upsample(size=(input_size[0],input_size[1]),mode='bilinear')
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    trainloader = data.DataLoader(LIPParsingEdgeDataSet(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, crop_size=input_size, 
                     mirror=args.random_mirror, mean=IMG_MEAN), 
                    batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    conv1_params = list(map(id,deeplab.Scale.conv1.parameters()))
    conv2_params = list(map(id,deeplab.Scale.layer1.parameters()))
    aspp_params = list(map(id, deeplab.Scale.layer5.parameters()))
    aspp_linear_params = list(map(id, deeplab.Scale.layer5.encoder.global_fc.parameters()))
    aspp_wo_linear_params = list(set(aspp_params) - set(aspp_linear_params))
    

    print(len(aspp_linear_params))
    print(len(aspp_wo_linear_params))
    print(len(aspp_params))
    ignore_params = conv1_params+conv2_params+aspp_wo_linear_params
   # print(ignore_params)
   # print(conv1_params.name)
   # print(conv2_params)
    base_params = filter(lambda p: id(p) not in ignore_params, deeplab.parameters())
    optimizer = optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, base_params), 'lr': args.learning_rate },
             {'params': filter(lambda p: id(p) not in aspp_linear_params, deeplab.Scale.layer5.parameters()), 'lr': args.learning_rate*10},], 
             lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()
  
   # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[10,15,18],0.1)          
    testloader = data.DataLoader(LIPDataValSet(args.test_dir,  args.test_list, crop_size=input_size, mean=IMG_MEAN), 
                                    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
      
    last_epoch = 0 
    iter_per_epoch = 23737/(args.batch_size*len(args.gpu.split(',')))
    for i_iter, batch in enumerate(trainloader):
        epoch = i_iter // iter_per_epoch
      #  if epoch > last_epoch:
      #      last_epoch += 1
      #      lr_scheduler.step()
        i_iter += args.start_iters
        images, labels,norms, _, _ = batch
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        pred_d, pred_ord = model(images)
        labels_ = labels.unsqueeze(1)
        preds = randomcrop.get_depth_sid(pred_d)
        pred_norm=d2n.depth2normal_layer_batch(preds,scale=2) 
        loss,loss1, loss2 = criterion(pred_ord, labels,pred_norm.cuda(), norms, weight = args.norm_weight)
        loss.backward()
        optimizer.step()

        labels = randomcrop.get_depth_sid(labels.float())
        if i_iter % 100 == 0:
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], i_iter)
            writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)
            writer.add_scalar('loss_depth', loss1.data.cpu().numpy(), i_iter)
            writer.add_scalar('loss_norm', loss2.data.cpu().numpy(), i_iter)

        if i_iter % args.save_pred_every == 0:
            images_inv = inv_preprocess(images, args.save_num_images, IMG_MEAN)
            labels_colors = decode_labels(labels, args.save_num_images, args.num_classes)
            if isinstance(pred_norm, list):
                pred_norm = pred_norm[0] 
            pred_norms = decode_norms(pred_norm,args.save_num_images)
            norms = decode_norms(norms,args.save_num_images)
            if isinstance(preds, list):
                preds = preds[0] 
            preds_colors = decode_predictions(preds, args.save_num_images, args.num_classes)
            for index, (img, lab,pred_norm,norm) in enumerate(zip(images_inv, labels_colors, pred_norms,norms)):
                writer.add_image('Images/'+str(index), img.transpose(2,0,1), i_iter)
                writer.add_image('Labels/'+str(index), colored_depthmap(lab).astype(np.uint8).transpose(2,0,1), i_iter)
                writer.add_image('preds/'+str(index),colored_depthmap(preds_colors[index]).astype(np.uint8).transpose(2,0,1), i_iter)
                writer.add_image('norm/'+str(index), (norm+1.0)/2.0, i_iter)
                writer.add_image('pred_norm/'+str(index), (pred_norm+1.0)/2.0, i_iter)
        if i_iter %50 ==0:
            print('iter = {} of {} completed, loss = {}, learning_rate = {:e}'.format(i_iter, args.num_steps, loss.data.cpu().numpy(), optimizer.param_groups[0]['lr']))

        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(deeplab.state_dict(),osp.join(args.snapshot_dir, 'nyu_'+str(args.num_steps)+'.pth'))                                                                         
            break

        if i_iter % (args.save_pred_every*4) == 0:
            print('taking snapshot ...')
            if i_iter % 20000 ==0:
                torch.save(deeplab.state_dict(),osp.join(args.snapshot_dir, 'nyu_'+str(i_iter)+'.pth'))     
            error= eval(model,testloader)
            if error[4]>0.8 or error[4]>max_a1:
                torch.save(deeplab.state_dict(),osp.join(args.snapshot_dir, 'nyu_best.pth'))     
            model.train()
            writer.add_scalar('abs_rel', error[1], i_iter)
            writer.add_scalar('sq_rel',error[1] , i_iter)
            writer.add_scalar('rmse', error[2], i_iter)
            writer.add_scalar('rmse_log', error[3], i_iter)
            writer.add_scalar('a1',error[4] , i_iter)
            writer.add_scalar('a2',error[5] , i_iter)
            writer.add_scalar('a3',error[6] , i_iter)
	    
    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
