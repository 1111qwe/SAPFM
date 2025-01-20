from torch.utils.data import Dataset,DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torchvision.transforms as transforms
import torch
from parameter import get_parameters
from scipy import stats

class Augmentation(nn.Module):
    def __init__(self):
        super(Augmentation, self).__init__()
        self.data = nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-45, 45)),
            )

    def forward(self, x):
        #（b, t, c, h, w)
        N,T,C,H,W = x.shape
        x = x.reshape(N,C*T,H,W)
        x = self.data(x)
        x = x.reshape(N, T, C, H, W)
        return x
    
class Dataset(Dataset):
    def __init__(self,  data, labels, info='train', config=None):
        self.augmentation = Augmentation()
        self.x = data.astype('float32')
        self.x = self.x.transpose(0,2,1,3,4) #(B,T,C,H,W)
        self.x = torch.from_numpy(self.x)
        if info == 'train':
            self.x = self.augmentation(self.x)
            # self.x = self.x
            self.y_reg = labels
            self.y_reg = torch.from_numpy(self.y_reg.astype('float32'))
        else:
            self.y_reg = torch.from_numpy(labels.astype('float32'))
            
        self.y_cls = torch.from_numpy((labels>=0.1).astype('float32'))
        
        if self.x.shape[0] != self.y_reg.shape[0]:
            print('error')

    def BoxCox(self, label):
        y_positive = label + 1 
        y_box_cox, lambda_ = stats.boxcox(y_positive)
        return y_box_cox, lambda_ 
    
    def __getitem__(self, index):
        return self.x[index], (self.y_reg[index], self.y_cls[index])  #self.x.shape=(B,T,C,H,W)

    def __len__(self):
        return len(self.x)
        # return 1
    
    
class DataLoaders():
    def __init__(self, train_path, test_path):
        
        #train_dataset
        trainData = np.load(train_path, allow_pickle=True).item()
        trainNums = list(trainData.keys())
        xTrain,yTrain = self.getDataset(trainData, trainNums)
        # xTrain,yTrain = self.underSampling(xTrain,yTrain)
        # yTrain,lambda_ = self.BoxCox(yTrain)
        # config.lambda_ = lambda_

        
        #test_dataset
        testData = np.load(test_path, allow_pickle=True).item()
        testNums = list(testData.keys())
        xTest, yTest, times, nums= self.getDataset(testData, testNums, 'test')
        print('xTrain:' ,xTrain.shape) 
        print('xTest:' ,xTest.shape)  
     
        xTrain = np.delete(xTrain,[6,9,10],1)
        xTest = np.delete(xTest,[6,9,10],1)
        # ['ERA5', 'IMERG', 'GLDAS', 'GsMap', 'CMORPH', 'SP', 'T2M', 'RH', 'DEM', 'lat', 'lon'] 
        self.xTrain = xTrain
        self.yTrain = yTrain      
        self.xTest = xTest
        self.yTest = yTest
        self.times = times
        self.nums = nums
        print('xTrain:' ,self.xTrain.shape)  
        print('xTest:' ,self.xTest.shape)  
        
        self.dataNormal()

    def underSampling(self, x, y):

        from sklearn.utils import resample
        from collections import Counter
        class_0_indices = np.where(y < 0.1)[0]
        class_1_indices = np.where(y >= 0.1)[0]
        
        x_class_0 = x[class_0_indices]
        x_class_1 = x[class_1_indices]

        y_class_0 = y[class_0_indices]
        y_class_1 = y[class_1_indices]
        
        x_class_0_downsampled, y_class_0_downsampled = resample(x_class_0, y_class_0, 
                                                        replace=False, 
                                                        n_samples=len(class_1_indices), 
                                                        random_state=42)
        
        x_resampled = np.vstack((x_class_0_downsampled, x_class_1))
        y_resampled = np.hstack((y_class_0_downsampled, y_class_1))

        # print(f'origin Class 0: {len(class_0_indices)} samples')
        # print(f'Downsampled Class 0: {len(y_class_0_downsampled)} samples')
        # print(f'Class 1: {len(y_class_1)} samples')

        print('Resampled dataset shape %s' % Counter(y_resampled))
        return x_resampled, y_resampled
        
    def dataNormal(self):
        xTrainMean = np.mean(self.xTrain,axis=(0,2,3,4), keepdims=True)
        xTrainStd = np.std(self.xTrain,axis=(0,2,3,4), keepdims=True)

        self.xTrain = (self.xTrain - xTrainMean) / xTrainStd
        self.xTest = (self.xTest - xTrainMean)/ xTrainStd
    
    def getDataset(self, allData, trainNums, info='train'):
        xTrain = [allData[num]['data'] for num in trainNums] # shape=(N, C, T, H, W)
        xTrain = np.vstack(xTrain)
        yTrain = [allData[num]['label'] for num in trainNums]
        yTrain = np.hstack(yTrain)
        if info == 'test':
            times = [allData[num]['time'] for num in trainNums]
            time_length = times[0].shape[0]
            times = np.hstack(times)
            padding_test_num = np.array([item for item in trainNums for _ in range(time_length)])
            return xTrain,yTrain, times, padding_test_num
        return xTrain,yTrain
        
    def loader(self, batch_size, num_workers, config):
        trainData = Dataset(self.xTrain, self.yTrain, 'train', config)
        testData = Dataset(self.xTest, self.yTest, 'test', config)
        
        trainLoader = DataLoader(trainData, 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 num_workers=num_workers, 
                                 )
        testLoader = DataLoader(testData, 
                                 batch_size=batch_size, 
                                 num_workers=num_workers, 
                                 )        
        return trainLoader, testLoader, self.times, self.nums

if __name__=="__main__":
    config = get_parameters()
    dataLoader = DataLoaders(config.train_path, config.test_path)
    trainLoader, testLoader, times, nums= dataLoader.loader(12, 1, config)
    print(times[:10])
    for data,(reg,cls) in trainLoader:
        
        print(data.shape, reg, cls)
        break