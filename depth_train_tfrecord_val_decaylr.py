import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import models
from PIL import Image
import random
import scipy.io as sio
from scipy import misc
import cv2
from tensorflow.python import pywrap_tensorflow
import time
import pprint
import scipy.misc
# import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]=""

flags = tf.app.flags
flags.DEFINE_string("mode",                         "",    "(train_rigid, train_flow) or (test_depth, test_pose, test_flow)")
flags.DEFINE_string("dataset_tf_dir",               "",    "Dataset directory")
flags.DEFINE_string("init_ckpt_file",               "",    "Specific checkpoint file to initialize from")
flags.DEFINE_string("init_weight_file",             "",    "Specific weight file to initialize from")
flags.DEFINE_string("checkpoint_dir",               "",    "Directory name to save the checkpoints")
flags.DEFINE_string("eval_data",                    "",    "Directory name to eval data")
flags.DEFINE_string("loss_type",                    "",    "loss_type")
flags.DEFINE_string("op_type",                      "",    "op_type")
flags.DEFINE_string("net_type",       "ResNet50UpProj",    "net_type")
flags.DEFINE_string("fixbn",                   "false",    "fix_bn")
flags.DEFINE_string("withda",                   "true",    "fix_bn")
flags.DEFINE_string("rawdep",                   "true",    "rawdep")
flags.DEFINE_integer("batch_size",                  16,    "The size of of a sample batch")
flags.DEFINE_integer("depth_height",               228,    "The height of depth")
flags.DEFINE_integer("depth_width",                304,    "The withd of depth")
flags.DEFINE_integer("tfrecord_size",             1000,    "The size of tfrecords")
flags.DEFINE_integer("max_steps",              3000000,    "Maximum number of training iterations")
flags.DEFINE_integer("start_step",                  -1,    "the starting step")
flags.DEFINE_float("learning_rate",               1e-5,    "Learning rate for adam")
flags.DEFINE_integer("save_ckpt_freq",           10000,    "Save the checkpoint model every save_ckpt_freq iterations")
flags.DEFINE_integer("eval_freq",                  200,    "eval every eval_freq iterations")
flags.DEFINE_integer("summary_freq",               100,    "summary frequence")
flags.DEFINE_float("alpha_ssim",                  0.85,    "Alpha weight between SSIM and L1 in reconstruction loss")
flags.DEFINE_integer("max_to_keep",                 20,    "Maximum number of checkpoints to save")



opt = flags.FLAGS


class FCRN(object):
    def __init__(self):

        # self.ckpt_file ='./checkpoints/raw/model'
        self.checkpoint_dir = opt.checkpoint_dir
        # self.ckpt_init_file = './checkpoints/init_0/model-0'
        self.ckpt_init_file = opt.init_ckpt_file
        self.init_weight_dir = opt.init_weight_file
        self.tfrecord_train_file = opt.dataset_tf_dir
        self.batch_size = opt.batch_size
        self.img_height = 228
        self.img_width = 304
        self.learning_rate = opt.learning_rate
        self.max_steps = opt.max_steps
        # self.training_epochs = 20
        self.alpha = opt.alpha_ssim

    

    def load_weight(self, data_path, sess):
        
        # initial all the variable with random values
        # with tf.variable_scope("", reuse=True):
        #     for var in tf.trainable_variables():
        #         # print(var.get_shape())
        #         # print(var.name)
        #         data=0.1*np.random.randn(*(var.get_shape()))
        #         sess.run(var.assign(data))
                
        #         # var = tf.get_variable(param_name)

        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            # print(op_name)
            # print('#################################')
            with tf.variable_scope('dp', reuse=tf.AUTO_REUSE):
                with tf.variable_scope(op_name, reuse=True):
                    for param_name, data in data_dict[op_name].iteritems():
                        try:
                            print('load: ' + op_name + '/' + param_name)
                            var = tf.get_variable(param_name)
                            sess.run(var.assign(data))
                        except:
                            print('not found: '+ op_name + '/'+param_name)
        
    def build_inference(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                                                self.img_height, self.img_width, 3], name='raw_input')
        rgb_input = tf.image.convert_image_dtype(input_uint8, dtype=tf.float32)

        with tf.variable_scope('dp', reuse=True):
            if opt.net_type == 'ResNet50UpProj':
                net = models.ResNet50UpProj({'data': rgb_input}, self.batch_size, 1, False)
            elif opt.net_type == 'ResNet50UpProj_aspp':
                net = models.ResNet50UpProj_aspp({'data': rgb_input}, self.batch_size, 1, False)
        

        self.eval_inputs = rgb_input
        self.eval_pred_depth = net.get_output()

        self.abs_rel_ph = tf.placeholder(tf.float32, None, name='abs_rel_ph')
        self.sq_rel_ph = tf.placeholder(tf.float32, None, name='sq_rel_ph')
        self.rms_ph = tf.placeholder(tf.float32, None, name='rms_ph')
        self.log_rms_ph = tf.placeholder(tf.float32, None, name='log_rms_ph')
        self.a1_ph = tf.placeholder(tf.float32, None, name='a1_ph')
        self.a2_ph = tf.placeholder(tf.float32, None, name='a2_ph')
        self.a3_ph = tf.placeholder(tf.float32, None, name='a3_ph')


        abs_rel_smry = tf.summary.scalar("abs_rel", self.abs_rel_ph)
        sq_rel_smry = tf.summary.scalar("sq_rel", self.sq_rel_ph)
        rms_smry = tf.summary.scalar("rms", self.rms_ph)
        log_rms_smry = tf.summary.scalar("log_rms", self.log_rms_ph)
        a1_smry = tf.summary.scalar("a1", self.a1_ph)
        a2_smry = tf.summary.scalar("a2", self.a2_ph)
        a3_smry = tf.summary.scalar("a3", self.a3_ph)
        self.merged_summary_test = tf.summary.merge([abs_rel_smry, sq_rel_smry, rms_smry, log_rms_smry, 
                                                        a1_smry, a2_smry, a3_smry])

        #load eval data
        self.eval_data = np.load(opt.eval_data).item()
        self.eval_data['rgb'] =  self.eval_data['rgb'] *255.0
        # print(self.eval_data['depth'].shape)
        # # print(self.eval_data['rgb'].shape)
        # for i in range((eval_data['rgb'].shape)[0]):
        #     input_rgb_path = opt.checkpoint_dir + '/eval_rgb/' + str(i) +'_rgb.png'                   
        #     scipy.misc.imsave(input_rgb_path, self.eval_data['rgb'][i,:,:,:])


    def inference(self, sess):
        


        eval_data = self.eval_data

        num_eval = (eval_data['rgb'].shape)[0]

        rms=np.zeros(num_eval,np.float32)
        log_rms=np.zeros(num_eval,np.float32)
        abs_rel=np.zeros(num_eval,np.float32)
        sq_rel=np.zeros(num_eval,np.float32)
        
        a1=np.zeros(num_eval,np.float32)
        a2=np.zeros(num_eval,np.float32)
        a3=np.zeros(num_eval,np.float32)

        
        cnt = 0
        res = np.zeros((128*26,160*26), np.float32)
        res_err = np.zeros((128*26,160*26), np.float32)

        for t in range(0,num_eval, opt.batch_size):
            cur_batch = np.zeros((self.batch_size, self.img_height, self.img_width, 3), np.float32)
            if t+opt.batch_size < num_eval:
                cur_batch = eval_data['rgb'][t:t+opt.batch_size,:,:,:]
            else:
                cur_batch[:num_eval-t] = eval_data['rgb'][t:,:,:,:]

            # print(cur_batch.shape)

            

            pred_depth = sess.run([self.eval_pred_depth], feed_dict={self.eval_inputs: cur_batch})
            pred_depth = pred_depth[0]
            pred_depth = np.array(pred_depth).astype('float32')
            # print(pred_depth.shape)
            # pred_depth = pred_depth[0]  

            

            for i in range(opt.batch_size):
                if cnt < num_eval:
                    # print(eval_data['depth'][cnt,:,:].shape, pred_depth[i,:,:,0].shape)
                    abs_rel[cnt], sq_rel[cnt], rms[cnt], log_rms[cnt], a1[cnt], a2[cnt], a3[cnt] = self.compute_errors(eval_data['depth'][cnt,:,:], pred_depth[i,:,:,0])
                    
                    #save eval res
                    cur_pred_depth = pred_depth[i,:,:,0]
                    
                    row = cnt//26
                    col = cnt%26

                    # print(cnt, row,col)

                    res[row*128:(row+1)*128, col*160:(col+1)*160] = cur_pred_depth/10.0*255.0
                    res_err[row*128:(row+1)*128, col*160:(col+1)*160] = abs(cur_pred_depth - eval_data['depth'][cnt,:,:])/10.0*255.0
                    cnt +=1
                    
                    # if cnt == 1:
                    #     cur_pred_depth = pred_depth[i,:,:,0]
                    #     cur_pred_depth=cur_pred_depth/10.0*255.0
                    #     pred_depth_path = opt.checkpoint_dir + '/' + str(self.step).zfill(9) + '_' + str(cnt) +'d.png'
                    #     scipy.misc.imsave(pred_depth_path, cur_pred_depth)
                    #     if self.step == 0:
                    #         input_rgb_path = opt.checkpoint_dir + '/' + str(self.step).zfill(9) + '_' + str(cnt) +'rgb.png'                   
                    #         scipy.misc.imsave(input_rgb_path, cur_batch[i,:,:,:])

                    # if not os.path.isdir(opt.checkpoint_dir + '/' + str(self.step)):
                    #     os.makedirs(opt.checkpoint_dir + '/' + str(self.step))
                    # cur_pred_depth = pred_depth[i,:,:,0]
                    # cur_pred_depth=(cur_pred_depth - np.min(cur_pred_depth))/(np.max(cur_pred_depth)-np.min(cur_pred_depth))
                    # pred_depth_path = opt.checkpoint_dir + '/' + str(self.step) + '/' + str(cnt) +'d.png'
                    # scipy.misc.imsave(pred_depth_path, cur_pred_depth)
                    # input_rgb_path = opt.checkpoint_dir + '/' + str(self.step) + '/' + str(cnt) +'rgb.png'                   
                    # scipy.misc.imsave(input_rgb_path, cur_batch[i,:,:,:])

                    # index = depth_files[i].find('test_depth')
                    # # pred_depth_path = FLAGS.pred_dir + depth_files[i][index+11:-4]+'.mat'
                    # pred_depth_path = opt.pred_dir +'/' + depth_files[i][index + 11:-4] + '.png'
                    # # sio.savemat(pred_depth_path, {"imgDepthPred": pred_depth})
                    # print(pred_depth_path)
                    
            
    
        pred_depth_path = opt.checkpoint_dir + '/' + str(self.step).zfill(9) +'d.png'
        scipy.misc.imsave(pred_depth_path, res)
        pred_depth_path = opt.checkpoint_dir + '/' + str(self.step).zfill(9) +'e.png'
        scipy.misc.imsave(pred_depth_path, res_err)

        self.abs_rel = abs_rel.mean()
        self.sq_rel = sq_rel.mean()
        self.rms = rms.mean()
        self.log_rms = log_rms.mean()
        self.a1 = a1.mean()
        self.a2 = a2.mean()
        self.a3 = a3.mean()
        print('eval res: abs_rel %f   sq_rel %f  rms %f log_rms %f a1 %f a2 %f a3 %f' %(self.abs_rel, self.sq_rel, self.rms, self.log_rms, self.a1, self.a2, self.a3))

        # print(self.abs_rel)

    def train(self):
        
        seed = 8964
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print('net_type:',opt.net_type)
        print('dataset_tf_dir:',opt.dataset_tf_dir)
        print('init_ckpt_file:',opt.init_ckpt_file)
        print('checkpoint_dir:',opt.checkpoint_dir)
        print('loss_type:',opt.loss_type)
        print('op_type:',opt.op_type)
        print('learning_rate:',opt.learning_rate)
        print('alpha_ssim:',opt.alpha_ssim)
        print('fixbn:',opt.fixbn)
        print('withda:',opt.withda)


        print('build tf reader for %s... '%(opt.dataset_tf_dir))
        

        # read from tfrecord file
        data_queue = tf.train.string_input_producer([self.tfrecord_train_file])
        rgb_input, depth_gt = self.decode_from_tfrecords(data_queue, is_batch=True)
        
        if opt.withda == 'true':
            rgb_input = tf.image.resize_images(rgb_input, [480, 640])
            rgb_input, depth_gt = self.data_augmentation(rgb_input, depth_gt, 480, 640)
            rgb_input = tf.image.resize_images(rgb_input, [self.img_height, self.img_width])
            rgb_input = tf.cast(rgb_input, dtype=tf.uint8)
        
        # print(depth_gt.get_shape())

        rgb_input = tf.image.convert_image_dtype(rgb_input, dtype=tf.float32)*255.0
        # construct network

        with tf.variable_scope('dp', reuse=tf.AUTO_REUSE):
            if opt.net_type == 'ResNet50UpProj':
                if opt.fixbn == 'false':
                    net = models.ResNet50UpProj({'data': rgb_input}, self.batch_size, 0.5, True)
                elif opt.fixbn == 'true':
                    net = models.ResNet50UpProj({'data': rgb_input}, self.batch_size, 0.5, False)
            elif opt.net_type == 'ResNet50UpProj_aspp':
                if opt.fixbn == 'false':
                    net = models.ResNet50UpProj_aspp({'data': rgb_input}, self.batch_size, 0.5, True)
                elif opt.fixbn == 'true':
                    net = models.ResNet50UpProj_aspp({'data': rgb_input}, self.batch_size, 0.5, False)

        print('drop out:', net.keep_prob)
        # prediction
        if opt.rawdep == 'false':
            pred_depth = net.get_output()
            depth_gt_resize = tf.image.resize_images(depth_gt, [128, 160])
        elif opt.rawdep == 'true':
            pred_depth = tf.image.resize_images(net.get_output(), [480, 640])
            depth_gt_resize = depth_gt
        # 

        # compute loss
        
        # depth_gt_resize = depth_gt

        depth_mask = tf.stop_gradient(tf.to_float(tf.greater(depth_gt_resize, tf.constant(1e-1))))

        pixel_loss = 0.0
        if opt.loss_type == 'berhu':
            pixel_loss = self.berHu_loss(depth_gt_resize, pred_depth, depth_mask)
        elif opt.loss_type == 'l1':
            pixel_loss = tf.reduce_mean(tf.abs(depth_gt_resize - pred_depth)*depth_mask)
        elif opt.loss_type == 'l2':
            pixel_loss = tf.reduce_mean((depth_gt_resize - pred_depth)*(depth_gt_resize - pred_depth)*depth_mask)
        elif opt.loss_type == 'berhu_fix':
            pixel_loss = self.berHu_loss_fix(depth_gt_resize, pred_depth, depth_mask)
        
        # SSIM loss
        ssim_loss = 0.0
        if opt.alpha_ssim > 0:
            pred_depth_n = (pred_depth - tf.reduce_min(pred_depth)) / (tf.reduce_max(pred_depth) - tf.reduce_min(pred_depth )) * 255.0
            depth_gt_resize_n = (depth_gt_resize - tf.reduce_min(depth_gt_resize)) / (tf.reduce_max(depth_gt_resize) - tf.reduce_min(depth_gt_resize)) * 255.0
            ssim = self.SSIM(pred_depth_n, depth_gt_resize_n)
            ssim_loss = tf.reduce_mean(ssim)
            total_loss = (1 - self.alpha) * pixel_loss + self.alpha * ssim_loss
        else:
            total_loss = pixel_loss   


        pred_min_smry = tf.summary.scalar("predicted min", tf.reduce_min(pred_depth))
        pred_max_smry = tf.summary.scalar("predicted max", tf.reduce_max(pred_depth))
        pred_mean_smry = tf.summary.scalar("predicted mean", tf.reduce_mean(pred_depth))
        gt_min_smry = tf.summary.scalar("gt min", tf.reduce_min(depth_gt_resize))
        gt_max_smry = tf.summary.scalar("gt max", tf.reduce_max(depth_gt_resize))
        gt_mean_smry = tf.summary.scalar("gt mean", tf.reduce_mean(depth_gt_resize))
        pixel_loss_smry = tf.summary.scalar("pixel_loss", pixel_loss)
        ssim_loss_smry = tf.summary.scalar("ssim_loss", ssim_loss)
        tot_loss_smry = tf.summary.scalar("total_loss", total_loss)
        rgb_input_smry = tf.summary.image("rgb", rgb_input)
        # tf.summary.image("ground-truth depth", depth_gt)
        gt_resize_smry = tf.summary.image("gt", depth_gt_resize)
        mask_smry = tf.summary.image("mask", depth_mask)
        pred_depth_smry = tf.summary.image("predicted depth", pred_depth)
        

        # learing rate setting
        global_step = tf.Variable(0, name='global_step', trainable=False)
        incr_global_step = tf.assign(global_step, global_step+1)
        
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 200000, 0.5, staircase=True)
        # boundaries = [1000000, 2000000]
        # values = [1.0, 0.5, 0.1]
        # for v in range(len(values)):
        #     values[v] *= self.learning_rate
        # # print(values)
        # learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        lrrate_smry = tf.summary.scalar('learning_rate', learning_rate)
        self.merged_summary_train = tf.summary.merge([pred_min_smry, pred_max_smry, pred_mean_smry, gt_min_smry,gt_max_smry, gt_mean_smry, 
                                                        pixel_loss_smry, ssim_loss_smry, tot_loss_smry, rgb_input_smry, gt_resize_smry, mask_smry, pred_depth_smry, lrrate_smry])
        
        if opt.op_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
        elif opt.op_type == 'grad':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

        saver = tf.train.Saver(max_to_keep=opt.max_to_keep)

        self.build_inference()
        
        # launch graph
        with tf.Session() as sess:
            
            # dbg load resnet weight
            sess.run(tf.global_variables_initializer())
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)


            # # initial from pre-trained value
            # print('load pre-trained weights from %s ...'%(self.ckpt_init_file))
            # saver.restore(sess, self.ckpt_init_file)


            #try to load the existed checkpoint first
            checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoints')
            last_ck_path = tf.train.latest_checkpoint(checkpoint_path)
            
            if last_ck_path != None:
                print('load pre-trained weights from ...')
                print(last_ck_path)
                saver.restore(sess,last_ck_path)
            else:
                print('load pre-trained weights from %s ...'%(self.ckpt_init_file))
                saver.restore(sess, self.ckpt_init_file)

            # with tf.variable_scope("dp", reuse=tf.AUTO_REUSE):
            #     reader = tf.train.NewCheckpointReader(self.ckpt_init_file)

            #     vars_in_ckpt = reader.get_variable_to_shape_map()
            #     for r_var in vars_in_ckpt:
            #         print(r_var)
            #         cur_var = tf.get_variable(r_var)
            #         sess.run(cur_var.assign(reader.get_tensor(r_var)))

                #
                # for cur_var in variables_to_restore_C:
                #     cur_var_name = tf.contrib.framework.get_variable_full_name(cur_var)
                #     saved_var_name = cur_var_name.split('unflow/')[1]
                #     saved_var = reader.get_tensor(saved_var_name)
                #     assign_placeholder = tf.placeholder(tf.float32, shape=saved_var.shape)
                #
                #     assign_op = cur_var.assign(assign_placeholder)
                #     cur_var._assign_placeholder = assign_placeholder
                #     cur_var._assign_op = assign_op
                #     assign_op_list.append(assign_op)
                #     assign_placeholder_list.append(assign_placeholder)
                #     saved_var_list.append(saved_var)
                #     # sess.run(assign_op, feed_dict={assign_placeholder: saved_var})


            # --------------------------------------
            # self.load_weight(self.init_weight_dir, sess)
            # checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoints')
            # self.save(saver, sess, checkpoint_path, 0)
            # asdf
            # ========================================

            # with tf.variable_scope("bn_conv1", reuse=True):
            #     print(sess.run(tf.get_variable('mean')))
            #     print(sess.run(tf.get_variable('variance')))
            # for var in tf.trainable_variables():
            #     print(var.name, var)
            # asdf
            # summary
            
            writer = tf.summary.FileWriter(self.checkpoint_dir, sess.graph)

            # Collect tensors that are useful later (e.g. tf summary)
           
            # rgb_image_normalize = preprocess_rgb_image(rgb_image)
            # depth_image_normalize = preprocess_depth_image(depth_image)

            step = 0

            
            if opt.start_step >= 0:               
                sess.run(global_step.assign(0))      

            t_0 = time.time()
            while step < self.max_steps:
                
                # t_1 = time.time()
                
                step = sess.run([global_step])[0]
                self.step = step


                if step % opt.eval_freq == 0:
                    
                    self.inference(sess)

                    summary_test = sess.run(self.merged_summary_test, feed_dict={self.abs_rel_ph: self.abs_rel, self.sq_rel_ph: self.sq_rel, 
                                                            self.rms_ph:self.rms, self.log_rms_ph:self.log_rms, self.a1_ph:self.a1,
                                                            self.a2_ph:self.a2, self.a3_ph:self.a3})
                    writer.add_summary(summary_test, global_step=step)
                if step % opt.save_ckpt_freq == 0 or (step + 1) == self.max_steps:
                    checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoints')
                    self.save(saver, sess, checkpoint_path, step)

                                   
                # parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
                if step % opt.summary_freq == 0:
                    
                    loss, _, summary = sess.run([pixel_loss, optimizer, self.merged_summary_train])
                    writer.add_summary(summary, global_step=step)
                    print('Step: %6d' % (step), 'loss: %6f' %(loss), 'time: %f' %((time.time() - t_0)) )
                    
                else:
                    loss,_ = sess.run([pixel_loss, optimizer])
                
                

                

                

            print('Step finished')
            coord.request_stop()
            coord.join(threads)
            writer.close()


    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        # return SSIM
        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)
    def data_augmentation(self, im_rgb, im_d, out_h, out_w):
            # Random scaling
        def random_scaling(im, im_d):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([1], 1, 1.5)
            x_scaling = scaling[0]
            y_scaling = scaling[0]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            im_d = tf.image.resize_area(im_d, [out_h, out_w])/scaling
            return im, im_d

        def random_rotating(im, im_d):
            rot_ang = tf.random_uniform([1], -5.0/180.0*3.14, 5.0/180.0*3.14)
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

    def berHu_loss(self, gt, pred, mask):
 
        abs_error = tf.abs(pred - gt)
        c = 0.2 * tf.reduce_max(abs_error, [1,2,3], keep_dims=True)
        berHu_loss = tf.where(abs_error <= c,abs_error, (tf.square(abs_error) + tf.square(c)) / (2 * c))
        loss = tf.reduce_mean(berHu_loss*mask)

        return loss

    def berHu_loss_fix(self, gt, pred, mask):
     
        abs_error = tf.abs(pred - gt)
        c = 0.4
        berHu_loss = tf.where(abs_error <= c,abs_error, (tf.square(abs_error) + tf.square(c)) / (2 * c))
        loss = tf.reduce_mean(berHu_loss*mask)

        return loss

    # def decode_from_tfrecords(self, data_queue, is_batch):
    #     reader = tf.TFRecordReader()
    #     _, serialized_example = reader.read(data_queue)
    #     features = tf.parse_single_example(serialized_example,
    #                                     features={
    #                                         'rgb': tf.FixedLenFeature([], tf.string),
    #                                         'depth': tf.FixedLenFeature([480*640], tf.float32),
    #                                     })
    #     rgb = tf.decode_raw(features['rgb'], tf.uint8)
    #     rgb = tf.reshape(rgb, [self.img_height, self.img_width, 3])
    #     depth = features['depth']
    #     depth = tf.reshape(depth, [480, 640, 1])

    #     if is_batch:
    #         min_after_dequeue = 1000*self.batch_size
    #         capacity = min_after_dequeue + 1000*self.batch_size
    #         rgb_batch, depth_batch = tf.train.shuffle_batch([rgb, depth],
    #                                                         batch_size=self.batch_size,
    #                                                         num_threads=3,
    #                                                         capacity=capacity,
    #                                                         min_after_dequeue=min_after_dequeue)
    #     return rgb_batch, depth_batch

    def decode_from_tfrecords(self, data_queue, is_batch):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(data_queue)
        features = tf.parse_single_example(serialized_example,
                                        features={
                                            'rgb': tf.FixedLenFeature([], tf.string),
                                            'depth': tf.FixedLenFeature([opt.depth_height*opt.depth_width], tf.float32),
                                        })
        rgb = tf.decode_raw(features['rgb'], tf.uint8)
        rgb = tf.reshape(rgb, [self.img_height, self.img_width, 3])
        depth = features['depth']
        depth = tf.reshape(depth, [opt.depth_height, opt.depth_width, 1])

        if is_batch:
            min_after_dequeue = opt.tfrecord_size*self.batch_size
            capacity = min_after_dequeue + opt.tfrecord_size*self.batch_size
            rgb_batch, depth_batch = tf.train.shuffle_batch([rgb, depth],
                                                            batch_size=self.batch_size,
                                                            num_threads=3,
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        return rgb_batch, depth_batch

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")
        if shuffle:
            random.shuffle(idx_list)
        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
            minibatch_start += minibatch_size
        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])
        return zip(range(len(minibatches)), minibatches)

    def get_train_batch(self, train_rgb, train_depth, batch_size, batchidx, image_height, image_width):
        if len(batchidx) == batch_size:
            rgb_batch = np.zeros((batch_size, 480, 640, 3), dtype='float32')
            depth_batch = np.zeros((batch_size, 427, 561, 1), dtype='float32')
            for i in range(len(batchidx)):
                rgb_name = train_rgb[batchidx[i]]
                
                # rgb_img = Image.open(rgb_name)
                # rgb_img = rgb_img.resize([self.img_width, self.img_height], Image.ANTIALIAS)
                # print(rgb_name)
                # rgb_img = cv2.imread(rgb_name)
                # rgb_img = cv2.resize(rgb_img, (self.img_height, self.img_width))
                # print(rgb_name)
                # t_0 = time.time()
                try:
                    rgb_img = cv2.imread(rgb_name)
                    # rgb_img = cv2.resize(rgb_img, (self.img_width, self.img_height))
                except:
                    print(rgb_name)
                    return rgb_batch, depth_batch

                # plt.imshow(img)
                # plt.show()
                rgb_img = np.array(rgb_img).astype('float32')
                rgb_img = np.expand_dims(np.asarray(rgb_img), axis=0)
                rgb_batch[i,:,:,:]=rgb_img
                
                # t_1=time.time()

                depth_name = train_depth[batchidx[i]]
                data = sio.loadmat(depth_name)
                depth_img = data['imgDepthFilled']
                depth_img = Image.fromarray(depth_img)
                # depth_img = depth_img.resize([self.img_width, self.img_height], Image.ANTIALIAS)

                # plt.imshow(img)
                # plt.show()
                depth_img = np.array(depth_img).astype('float32')
                depth_img = np.expand_dims(np.asarray(depth_img), axis=0)
                depth_img = np.expand_dims(np.asarray(depth_img), axis=-1)
                depth_batch[i,:,:,:]=depth_img
                # t_2= time.time()
                # print(1000*(t_1 - t_0), 1000*(t_2 - t_1))
            return rgb_batch, depth_batch
    def read_data(self):
        # training data
        train_rgb = []
        with open('../datasets/nyudepth/train_rgb.txt', 'r') as f:
            for images in f:
                images = images.rstrip('\n')
                train_rgb.append(images)
        train_depth = []
        with open('../datasets/nyudepth/train_depth.txt', 'r') as f:
            for images in f:
                images = images.rstrip('\n')
                train_depth.append(images)
        return train_rgb, train_depth

    def save(self, saver, sess, checkpoint_dir, step):
        model_name = 'model'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)
    def compute_errors(self, gt, pred):
        
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

def main(_):
    model = FCRN()
    model.train()
   # model.test()


if __name__ == '__main__':
    tf.app.run()
