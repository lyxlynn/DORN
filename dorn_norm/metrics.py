import time
import numpy as np
class Result():
    def __init(self):
        self.rmse = 0
        self.abs_rel =0
        self.sq_rel =0
        self.rmse_log=0
        self.a1=0
        self.a2=0
        self.a3=0
        self.gpu_time=time.perf_counter()

    def evaluate(self,pred,gt):
        
        min_depth = 1e-3
        max_depth = 80
        pred = np.asarray(pred,dtype = np.float32)        
        gt = np.asarray(gt,dtype = np.float32)        
        # print(np.min(gt))

        pred[pred < min_depth] = min_depth
        pred[pred > max_depth] = max_depth
        gt[gt < min_depth] = min_depth
        gt[gt > max_depth] = max_depth
     #   print(type(gt),'gt')
      #  print(type(pred),'pred')
      #  print(gt,'gt')
      #  print(pred,'pred')
        thresh = np.maximum((gt / pred), (pred / gt))
        self.a1 = (thresh < 1.25).mean()
        self.a2 = (thresh < 1.25 ** 2).mean()
        self.a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        self.rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        self.rmse_log = np.sqrt(rmse_log.mean())

        self.abs_rel = np.mean(np.abs(gt - pred) / gt)

        self.sq_rel = np.mean(((gt - pred)**2) / gt)
        self.gpu_time =- time.perf_counter()
        return self.a1,self.a2,self.a3,self.rmse,self.rmse_log,self.abs_rel, self.sq_rel
