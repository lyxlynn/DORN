from datetime import datetime
import shutil
import socket
import time
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from dataloaders import nyu_dataloader
from metrics import Result#AverageMeter, Result
import utils
import criteria
import os
import torch.nn as nn
from dataset.datasets import LIPParsingEdgeDataSet, LIPDataValSet
import DORN_nyu
import numpy as np

args = utils.parse_command()
print(args)
best_result=np.ones(7)
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


def NYUDepth_loader(data_path, batch_size=32, isTrain=True):
    if isTrain:
        traindir = os.path.join(data_path, 'train')
        print(traindir)

        if os.path.exists(traindir):
            print('Train dataset file path is existed!')
        trainset = nyu_dataloader.NYUDataset(traindir, type='train')
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True)  # 
        return train_loader
    else:
        valdir = os.path.join(data_path, 'val')
        print(valdir)

        if os.path.exists(valdir):
            print('Test dataset file path is existed!')
        valset = nyu_dataloader.NYUDataset(valdir, type='val')
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=1, shuffle=False  # 
        )
        return val_loader


def main():
    global args, best_result, output_directory

    if torch.cuda.device_count() > 1:
        args.batch_size = args.batch_size * torch.cuda.device_count()
        train_loader = data.DataLoader(LIPParsingEdgeDataSet(args.data_dir, args.data_list, max_iters=args.max_iter, crop_size=args.input_size, 
                     mirror=args.random_mirror, mean=IMG_MEAN), 
                    batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True)

        #train_loader = NYUDepth_loader(args.data_path, batch_size=args.batch_size, isTrain=True)
        val_loader = data.DataLoader(LIPDataValSet(args.test_dir,  args.test_list, crop_size=args.input_size, mean=IMG_MEAN), 
                                    batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
        #val_loader = NYUDepth_loader(args.data_path, batch_size=args.batch_size, isTrain=False)
    else:
    #    train_loader = NYUDepth_loader(args.data_path, batch_size=args.batch_size, isTrain=True)
         train_loader = data.DataLoader(LIPParsingEdgeDataSet(args.data_dir, args.data_list, max_iters=args.max_iter, crop_size=args.input_size, 
                     mirror=args.random_mirror, mean=IMG_MEAN), 
                    batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True)

        #val_loader = NYUDepth_loader(args.data_path, isTrain=False)
         val_loader = data.DataLoader(LIPDataValSet(args.test_dir,  args.test_list, crop_size=args.input_size, mean=IMG_MEAN), 
                                    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    if args.resume:
        assert os.path.isfile(args.resume), \
            "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        if torch.cuda.device_count() > 1:
            model_dict = checkpoint['model'].module.state_dict()  # Multigpu module
        else:
            model_dict = checkpoint['model'].state_dict()
        model = DORN_nyu.DORN()
        model.load_state_dict(model_dict)
        # otimizer SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> creating Model")
        model = DORN_nyu.DORN()
        print("=> model created.")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        start_epoch = 0
    # Multi GPU
    if torch.cuda.device_count():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.cuda()
    criterion = criteria.ordLoss()
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    best_txt = os.path.join(output_directory, 'best.txt')

    log_path = os.path.join(output_directory, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    for epoch in range(start_epoch, args.epochs):
        # lr = utils.adjust_learning_rate(optimizer, args.lr, epoch)  # 

        train(train_loader, model, criterion, optimizer, epoch, logger)  # train for one epoch
        result, img_merge = validate(val_loader, model, epoch, logger)  # evaluate on validation set

        # remember best rmse and save checkpoint
        is_best = result[0] < best_result[0]
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}\nrmse={:.3f}\nabs_rel={:.3f}\nrmse_log={:.3f}\na1={:.3f}\na2={:.3f}\na3={:.3f}\n".
                    format(epoch, result[3], result[5], result[4], result[0], result[1], result[2]))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer,
        }, is_best, epoch, output_directory)

    logger.close()


def train(train_loader, model, criterion, optimizer, epoch, logger):
    #average_meter = AverageMeter()
    model.train(False)  # switch to train mode
    end = time.time()

    batch_num = len(train_loader)
    current_step = batch_num * args.batch_size * epoch

    for i, batch in enumerate(train_loader):
        lr = utils.update_ploy_lr(optimizer, args.lr, current_step, args.max_iter)
        input, target , ori_size , name =batch
        input, target = input.cuda(), target.cuda()
        data_time = time.time() - end

        current_step += input.data.shape[0]

        if current_step == args.max_iter:
            logger.close()
            print('Iteration finished!')
            break

        torch.cuda.synchronize()

        # compute pred
        end = time.time()
        pred_d, pred_ord = model(input)  # 

        loss = criterion(pred_ord, target)
        optimizer.zero_grad()
        loss.backward()  # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        depth = nyu_dataloader.get_depth_sid(pred_d)
        
        target_dp = nyu_dataloader.get_depth_sid(target.float())
      #  print(depth,'depth')  
      #  print(target_dp,'target_dp.data')  
        result.evaluate(depth.data, target_dp.data)
       # average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  'learning_rate={lr:.8f} '
                  't_Data={data_time:.3f} '
                  't_GPU={gpu_time:.3f}\n\t'
                  'Loss={loss:.3f} '
                  'RMSE={result.rmse:.3f}'
                  'RML={result.abs_rel:.3f} '
                  'Log10={result.rmse_log:.3f} '
                  'Delta1={result.a1:.3f} '
                  'Delta2={result.a2:.3f} '
                  'Delta3={result.a3:.3f}'.format(
                epoch, i + 1, batch_num, lr=lr, data_time=data_time, loss=loss.item(),
                gpu_time=gpu_time, result=result))

            logger.add_scalar('Learning_rate', lr, current_step)
            logger.add_scalar('Train/Loss', loss.item(), current_step)
            logger.add_scalar('Train/RMSE', result.rmse, current_step)
            logger.add_scalar('Train/rml', result.abs_rel, current_step)
            logger.add_scalar('Train/Log10', result.rmse_log, current_step)
            logger.add_scalar('Train/Delta1', result.a1, current_step)
            logger.add_scalar('Train/Delta2', result.a2, current_step)
            logger.add_scalar('Train/Delta3', result.a3, current_step)

    avg = average_meter.average()



def validate(val_loader, model, epoch, logger, write_to_file=True):
   # average_meter = AverageMeter()

    model.eval()  # switch to evaluate mode

    end = time.time()
    errors = np.zeros(7)
    for i, batch in enumerate(val_loader):
        input, target , ori_size , name =batch
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred_d, pred_ord = model(input)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        depth = nyu_dataloader.get_depth_sid(pred_d)
        interp = nn.Upsample(size=(target.size(1),target.size(2)),mode='bilinear')
        depth= interp(depth).squeeze()
        errors += result.evaluate(depth.data, target.data)

        end = time.time()

        # save 8 images for visualization
        skip = 1

        rgb = input

        if i == 0:
            img_merge = utils.merge_into_row(rgb, target, depth,IMG_MEAN)
        elif (i < 8 * skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, depth,IMG_MEAN)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8 * skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)

    errors = errors/len(val_loader)
    logger.add_scalar('Test/Delta1', errors[0], epoch)
    logger.add_scalar('Test/Delta2', errors[1], epoch)
    logger.add_scalar('Test/Delta3', errors[2], epoch)
    logger.add_scalar('Test/rmse', errors[3], epoch)
    logger.add_scalar('Test/rmse_log', errors[4], epoch)
    logger.add_scalar('Test/Rel', errors[5], epoch)
    logger.add_scalar('Test/sq_rel', errors[6], epoch)
    return errors, img_merge


if __name__ == '__main__':
    main()
