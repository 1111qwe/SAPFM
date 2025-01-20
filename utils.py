import torch
import numpy as np
import logging
class EarlyStopping(object):
    def __init__(self, patience=100, mode='min'):
        self.patience = patience
        self.mode = mode
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.counter = 0
        self.stop = False

    def __call__(self, current_score):
        if (self.mode == 'min' and current_score < self.best_score) or \
           (self.mode == 'max' and current_score > self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                
class ModelCheckpoint:
    def __init__(self, checkpoint_path, save_model=True ,save_best_only=True, monitor='val_loss', mode='min'):
        self.checkpoint_path = checkpoint_path
        self.save_model = save_model
        self.save_best_only = save_best_only
        self.mode = mode
        self.monitor = monitor
        self.best_score = float('inf') if mode == 'min' else float('-inf')

    def __call__(self, model, current_score):
        if (not self.save_best_only) or \
           (self.save_best_only and
            (self.mode == 'min' and current_score < self.best_score) or
            (self.mode == 'max' and current_score > self.best_score)):
            if self.save_model:
                torch.save(model.state_dict(), self.checkpoint_path)
            logging.info("{} loss:{:.4f} --> {:.4f} saving {}".format(self.monitor, self.best_score, current_score, self.checkpoint_path))
            self.best_score = current_score

class Calculate_avg(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def update(self, v):
        if isinstance(v, torch.Tensor):
            count = v.numel()  #获取loss的个数
            v = v.sum() #获取loss值

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def accumulate(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def tensorCC(y_pred, y_true):
    # 计算相关系数
    y_pred[y_pred<0.1] = 0  
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)

    a1 = y_true - mean_true
    a2 = y_pred - mean_pred

    s1 = torch.sum(torch.mul(a1, a2))
    s2 = torch.sqrt(torch.mul(torch.sum(torch.square(a1)),torch.sum(torch.square(a2))))
    cc = s1 / s2
    return cc

def tensorKGE(y_pred, y_true):
    cc = tensorCC(y_pred, y_true)
    
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    std_true = torch.std(y_true)
    std_pred = torch.std(y_pred)
    
    beta = mean_pred / mean_true
    gama = (std_pred/mean_pred) / (std_true/mean_true)
    
    kge = 1 - torch.sqrt((1-cc)**2 + (1-beta)**2 + (1-gama)**2)
    return kge


def POD(y_true,y_pred,threshold=0.1):
    if threshold == 0:
        h = ((y_true > 0) & (y_pred > 0)).sum()
        m = ((y_true > 0) & (y_pred == 0)).sum()
    else:
        h = ((y_true >= threshold) & (y_pred >= threshold)).sum()
        m = ((y_true >= threshold) & (y_pred < threshold)).sum()
    return np.around(h/(h+m),decimals=3)

def FAR(y_true,y_pred,threshold=0.1):
    if threshold == 0:
        h = ((y_true > 0) & (y_pred > 0)).sum()
        f = ((y_true == 0) & (y_pred > 0)).sum()
    else:
        h = ((y_true >= threshold) & (y_pred >= threshold)).sum()
        f = ((y_true < threshold) & (y_pred >= threshold)).sum()
    return np.around(f/(f+h),decimals=3)

def CSI(y_true,y_pred,threshold=0.1):
    if threshold == 0:
        h = ((y_true > 0) & (y_pred > 0)).sum()
        f = ((y_true == 0) & (y_pred > 0)).sum()
        m = ((y_true > 0) & (y_pred == 0)).sum()
    else:
        h = ((y_true >= threshold) & (y_pred >= threshold)).sum()
        f = ((y_true < threshold) & (y_pred >= threshold)).sum()
        m = ((y_true >= threshold) & (y_pred < threshold)).sum()
        
    return np.around(h/(h+m+f),decimals=3)

def fbias(y_true,y_pred,threshold=0.1):
    if threshold == 0:
        h = ((y_true > 0) & (y_pred > 0)).sum()
        f = ((y_true == 0) & (y_pred > 0)).sum()
        m = ((y_true > 0) & (y_pred == 0)).sum()
    else:
        h = ((y_true >= threshold) & (y_pred >= threshold)).sum()
        f = ((y_true < threshold) & (y_pred >= threshold)).sum()
        m = ((y_true >= threshold) & (y_pred < threshold)).sum()
    return np.around((h+f)/(h+m),decimals=3)    
    
def kge_pro(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0,1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gama = (np.std(y_pred)/ np.mean(y_pred))/ (np.std(y_true)/np.mean(y_true))
    # Kling-Gupta Efficiency
    kge = 1 - np.sqrt((1 - r)**2 + (1 - gama)**2 + (1 - beta)**2)
    return np.around(kge,decimals=3),np.around(r,decimals=3),np.around(beta,decimals=3),np.around(gama,decimals=3)


def RMSE(y_true,y_pred):
    return np.around(np.sqrt(np.mean((y_pred - y_true) ** 2)),decimals=3)
 
def MAE(y_true,y_pred):
    return np.around(np.mean(np.abs(y_pred - y_true)),decimals=3)

def RB(y_true,y_pred):
    return np.around(sum(y_pred - y_true)/ sum(y_true), decimals=3)

if __name__ == "__main__":
    pass