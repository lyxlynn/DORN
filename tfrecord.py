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
import random
import timeit
import logging
from tensorboardX import SummaryWriter
from models import get_segmentation_model
#from models.models_base import resnet101_base
from dataset.datasets import LIPParsingEdgeDataSet
from utils.utils import decode_labels, inv_preprocess, decode_predictions
from utils.criterion import CriterionL1Loss 
from utils.encoding import SelfDataParallel, ModelDataParallel, CriterionDataParallel
from numpy.distutils.tests.test_exec_command import emulate_nonposix
import tensorflow as tf
from evaluate import * 
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 40
TEST_DIRECTORY = '/home/sdc/nyu_depth_v2_labeled'
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


def decode_from_tfrecords( data_queue, img_height,img_width,is_batch):
    batch_size=2
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(data_queue)
    features = tf.parse_single_example(serialized_example,
                                    features={
                                        'rgb': tf.FixedLenFeature([], tf.string),
                                        'depth': tf.FixedLenFeature([img_height*img_width], tf.float32),
                                    })
    rgb = tf.decode_raw(features['rgb'], tf.uint8)
    rgb = tf.reshape(rgb, [img_height, img_width, 3])
    depth = features['depth']
    depth = tf.reshape(depth, [img_height, img_width, 1])

    if is_batch:
        min_after_dequeue = 100*batch_size
        capacity = min_after_dequeue + 100*batch_size
        rgb_batch, depth_batch = tf.train.shuffle_batch([rgb, depth],
                                                        batch_size=batch_size,
                                                        num_threads=3,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    return rgb_batch, depth_batch

def data_augmentation( im_rgb, im_d, out_h, out_w):
    # Random scaling
    def random_scaling(im, im_d):
        batch_size, in_h, in_w, _ = im.get_shape().as_list()
        scaling = tf.random_uniform([2], 1, 1.15)
        x_scaling = scaling[0]
        y_scaling = scaling[1]
        out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
        out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
        im = tf.image.resize_area(im, [out_h, out_w])
        im_d = tf.image.resize_area(im_d, [out_h, out_w])
        return im, im_d

    def random_rotating(im, im_d):
        rot_ang = tf.random_uniform([1], -10.0/180.0, 10.0/180.0)
        im = tf.contrib.image.rotate(im, rot_ang, interpolation='BILINEAR')
        im_d = tf.contrib.image.rotate(im_d, rot_ang, interpolation='BILINEAR')
        return im, im_d

    # Random cropping
    def random_cropping(im, im_d, out_h, out_w):
        # batch_size, in_h, in_w, _ = im.get_shape().as_list()
        batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
        offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
        offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
        im = tf.image.crop_to_bounding_box(
            im, offset_y, offset_x, out_h, out_w)
        im_d = tf.image.crop_to_bounding_box(
            im_d, offset_y, offset_x, out_h, out_w)
        return im, im_d

    # Random coloring
    def random_coloring(im):
        batch_size, in_h, in_w, in_c = im.get_shape().as_list()
        im_f = tf.image.convert_image_dtype(im, tf.float32)

        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        im_aug = im_f ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        im_aug = im_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([in_c], 0.8, 1.2)
        white = tf.ones([batch_size, in_h, in_w])
        color_image = tf.stack([white * random_colors[i] for i in range(in_c)], axis=3)
        im_aug *= color_image

        # saturate
        im_aug = tf.clip_by_value(im_aug,  0, 1)

        im_aug = tf.image.convert_image_dtype(im_aug, tf.uint8)

        return im_aug
    im_rgb, im_d = random_scaling(im_rgb, im_d)
    im_rgb, im_d = random_rotating(im_rgb, im_d)
    im_rgb, im_d = random_cropping(im_rgb, im_d, out_h, out_w)

    im_rgb = tf.cast(im_rgb, dtype=tf.uint8)
    do_augment = tf.random_uniform([], 0, 1)
    im_rgb = tf.cond(do_augment > 0.5, lambda: random_coloring(im_rgb), lambda: im_rgb)

    do_flip_lr = tf.random_uniform([], 0, 1)
    im_rgb = tf.cond(do_flip_lr > 0.5,
                     lambda:  tf.map_fn(lambda img: tf.image.flip_left_right(img), im_rgb),
                     lambda: im_rgb)
    im_d = tf.cond(do_flip_lr > 0.5,
                   lambda: tf.map_fn(lambda img: tf.image.flip_left_right(img), im_d),
                   lambda: im_d)

    return im_rgb, im_d



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
    parser.add_argument("--gpu", type=str, default=GPU_DEVICES,
                        help="choose gpu device.")   
    return parser.parse_args()

args = get_arguments()
 
def eval(model, testloader):
    """Create the model and start the evaluation process."""
    args = get_arguments()
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w) 
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

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
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d images have been proceeded'%(index))
        image, label, ori_size, name = batch 
          
        ori_size = ori_size[0].numpy()
         
        output = predict(model, image.numpy(), (np.asscalar(ori_size[0]), np.asscalar(ori_size[1])),  scales=[1])
        depth_pred = np.asarray(output.squeeze(), dtype=np.float32)

       # scipy.misc.imsave(args.save_dir+name[0]+'.png',depth_pred)
        
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
    return errors
                
          
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
 
def adjust_learning_rate(optimizer, i_iter): 
     
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)     
        
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
#   """Create the model and start the training."""
#    writer = SummaryWriter(args.snapshot_dir)
#    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
#    h, w = map(int, args.input_size.split(','))
#    input_size = (h, w)
#
#    cudnn.enabled = True
# 
#    #deeplab = Res_CE2P(num_classes=args.num_classes) 
#   # deeplab = resnet101_base(num_classes=args.num_classes) 
#    deeplab = get_segmentation_model(args.network,num_classes=args.num_classes) 
#
#    saved_state_dict = torch.load(args.restore_from)
#    new_params = deeplab.state_dict().copy()
#   # for i in saved_state_dict:
#   #         i_parts = i.split('.')
#   #         if not i_parts[0]=='fc' and not  i_parts[0]=='last_linear' and not  i_parts[0]=='classifier':
#   #             new_params['.'.join(i_parts[0:])] = saved_state_dict[i]  
#    for i in saved_state_dict: 
#        i_parts = i.split('.') 
#        if not i_parts[1]=='layer5':
#            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
#    if args.start_iters > 0:
#        deeplab.load_state_dict(saved_state_dict)
#    else:
#        deeplab.load_state_dict(new_params)
#    model = ModelDataParallel(deeplab) 
#    model.train()
#    model.float() 
#    model.cuda()    
# 
#    criterion = CriterionL1Loss()
#    criterion = CriterionDataParallel(criterion)
#    criterion.cuda()
#    
#    cudnn.benchmark = True
#
#    if not os.path.exists(args.snapshot_dir):
#        os.makedirs(args.snapshot_dir)
#
#
#    trainloader = data.DataLoader(LIPParsingEdgeDataSet(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, crop_size=input_size, 
#                     mirror=args.random_mirror, mean=IMG_MEAN), 
#                    batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
#
#    optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, deeplab.parameters()), 'lr': args.learning_rate }], 
#                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
#    optimizer.zero_grad()
#  
#    testloader = data.DataLoader(LIPDataValSet(args.test_dir,  args.test_list, crop_size=input_size, mean=IMG_MEAN), 
#                                    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
#      
#    data_queue = tf.train.string_input_producer(['/home/sdb/Projects_wyw/deeper_depth/nyu_depth_train.tfrecords'])
#    rgb_input, depth_gt = decode_from_tfrecords(data_queue,224,224, is_batch=True)
# #   for i_iter, batch in enumerate(trainloader):
#   # for i_iter in range(5):
#   #  images_, labels_,  _, _ = batch
    data_queue = tf.train.string_input_producer(['/home/sdb/Projects_wyw/deeper_depth/nyu_depth_train_rmneg_rawdep_every10_1.tfrecords'])
    rgb_input, depth_gt = decode_from_tfrecords(data_queue,228,304, is_batch=True)
    print(type(rgb_input))
  #  images, labels = data_augmentation(rgb_input, depth_gt, 224, 224)
  #  print(labels.get_shape())
  #  print(type(images))
    print('2222')
    with tf.Session() as s:
        img,label= s.run([rgb_input, depth_gt])
    print('1111111111111')
        #sess.run(tf.initialize_all_variables())
        #sess.run(tf.global_variables_initializer())

    for i in range(1):     
        print(type(im),'im')
        images = tf.image.convert_image_dtype(rgb_input, dtype=tf.float32)
        print(type(images))
        print(type(images_))
       # images = Variable(images.cuda())
       # labels = Variable(labels.cuda())

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        preds = model(images)
        print (preds.shape)
        print(labels.shape)    
        loss = criterion(preds,labels, ignore_label=0)
        loss.backward()
        optimizer.step()

        if i_iter % 100 == 0:
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

        if i_iter % args.save_pred_every == 0:
            images_inv = inv_preprocess(images, args.save_num_images, IMG_MEAN)
            labels_colors = decode_labels(labels, args.save_num_images, args.num_classes)
            if isinstance(preds, list):
                preds = preds[0] 
            preds_colors = decode_predictions(preds, args.save_num_images, args.num_classes)
            for index, (img, lab) in enumerate(zip(images_inv, labels_colors)):
                writer.add_image('Images/'+str(index), img.transpose(2,0,1), i_iter)
                writer.add_image('Labels/'+str(index), lab, i_iter)
                #scipy.misc.imsave('{}.png'.format(i_iter),lab)
                writer.add_image('preds/'+str(index), preds_colors[index], i_iter)
        print(args.save_pred_every,'save_pred_every')          
        print('iter = {} of {} completed, loss = {}, learning_rate = {:e}'.format(i_iter, args.num_steps, loss.data.cpu().numpy(), lr))

        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(deeplab.state_dict(),osp.join(args.snapshot_dir, 'nyu_'+str(args.num_steps)+'.pth'))                                                                         
            break

        if i_iter % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save(deeplab.state_dict(),osp.join(args.snapshot_dir, 'nyu_'+str(i_iter)+'.pth'))     
            error= eval(model,testloader)
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
