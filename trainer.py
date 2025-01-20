import os
import time
import torch
import torch.nn as nn
from SAMPM import MergePrecipitation
from utils import Calculate_avg,tensorCC,tensorKGE,EarlyStopping,ModelCheckpoint
import logging
import torch
import torch.nn.functional as F
import numpy as np

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.loss = nn.MSELoss()
        # self.loss = nn.L1Loss() 
        # self.loss = nn.SmoothL1Loss()
        
    def forward(self, reg, target, cls):
        # cls_1_indices = torch.where(cls>=0.5)[0]
        
        # reg_1 = reg[cls_1_indices]
        # y_1 = y[cls_1_indices]
        # rmse = torch.sqrt(self.loss(reg, target))
        # swish = rmse * torch.sigmoid(rmse)
        
        swish =self.loss(reg, target)
        
        return swish
        

class Trainer(object):
    def __init__(self, data_loader, config):
        self.config = config
        print(config)
        # Model hyper-parameters
        self.input_dim = config.input_dim
        self.sequence = config.sequence
        self.size = config.size
        self.version = config.version
        
        self.nepoch = config.nepoch
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.lr = config.lr
        self.patience = config.patience
        
        self.save_log = config.save_log
        self.save_model = config.save_model
        self.pretrained_model = config.pretrained_model
        
        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.predict_path = os.path.join(config.pred_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version, config.model_save_name)
        if not os.path.exists(os.path.join(config.model_save_path, self.version)):
            os.makedirs(os.path.join(config.model_save_path, self.version))
       
        #load dataser 
        self.trainLoader, self.testLoader, self.test_times, self.test_nums= data_loader.loader(self.batch_size, self.num_workers, self.config)
        
        #save loss
        self.trainLossAvg = Calculate_avg()
        self.trainCLossAvg = Calculate_avg()
        self.trainCcAvg = Calculate_avg()
        self.trainAccAvg = Calculate_avg()
        
        self.testLossAvg = Calculate_avg()
        self.testCLossAvg = Calculate_avg()
        self.testCcAvg = Calculate_avg()
        self.testAccAvg = Calculate_avg()
        
        #run time
        self.startTimeTrain = 0
        self.startTimeTest = 0
        
        self.build_model()

        self.early_stopping = EarlyStopping(patience=config.patience, mode='max')
        self.check_point = ModelCheckpoint(checkpoint_path= self.model_save_path,save_model=self.save_model, save_best_only=True, monitor='val_loss', mode='max')
   
        if self.save_log:
            self.build_log()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()
        
    def train(self):
        self.M.train()   
        self.startTimeTrain = time.time()
        for xTrain, (yTrain_reg,yTrain_cls) in self.trainLoader:            
            xTrain = xTrain.to(self.device) #[b,t,c,h,w]
            yTrain_reg = yTrain_reg.reshape(-1,1)
            yTrain_reg = yTrain_reg.to(self.device)
            yTrain_cls = yTrain_cls.reshape(-1,1)
            yTrain_cls = yTrain_cls.to(self.device)
            
            outputs,classfication = self.M(xTrain)
            pred_cls = (classfication>0.5).float()
            
            # trian_reg_loss = self.r_loss(outputs, yTrain_reg)
            trian_reg_loss = self.r_loss(yTrain_reg, outputs, classfication)
            trian_cls_loss = self.c_loss(classfication, yTrain_cls)
            total_loss = trian_reg_loss + trian_cls_loss

            self.m_optimizer.zero_grad()
            total_loss.backward()
            self.m_optimizer.step()
        
            train_cc = tensorKGE(outputs, yTrain_reg)
            correct = (pred_cls == yTrain_cls).float()
            accuracy = correct.sum() / len(correct)
                   
            self.trainLossAvg.update(trian_reg_loss)
            self.trainCLossAvg.update(trian_cls_loss)
            self.trainCcAvg.update(train_cc)
            self.trainAccAvg.update(accuracy)

    def test(self):
        self.M.eval()    
        with torch.no_grad():
            allPred = []
            allTrue = []
            allTrue_cls = []
            allPred_cls = []
            for xTest, (yTest_reg,yTest_cls) in self.testLoader:
                xTest = xTest.to(self.device) #[b,c,t,h,w]
                yTest_reg = yTest_reg.reshape(-1,1)
                yTest_reg = yTest_reg.to(self.device) 
                yTest_cls = yTest_cls.reshape(-1,1)
                yTest_cls = yTest_cls.to(self.device)
                
                pred,classfication = self.M(xTest)

                pred[pred<0.1] = 0
                pred_cls = (classfication>0.5).float()
                pred[pred_cls==0] = 0
                
                # test_r_loss = self.r_loss(pred, yTest_reg)
                test_r_loss = self.r_loss(pred, yTest_reg, classfication)
                test_c_loss = self.c_loss(classfication, yTest_cls)
                
                allPred.append(pred)
                allTrue.append(yTest_reg)
                allPred_cls.append(pred_cls)
                allTrue_cls.append(yTest_cls)
                
                correct = (pred_cls == yTest_cls).float()
                accuracy = correct.sum() / len(correct)
                
                self.testLossAvg.update(test_r_loss)
                self.testCLossAvg.update(test_c_loss)
                
                self.testAccAvg.update(accuracy)
            allPred = torch.cat(allPred, dim=0)
            allTrue = torch.cat(allTrue, dim=0)   
            allPred_cls = torch.cat(allPred_cls, dim=0)
            allTrue_cls = torch.cat(allTrue_cls, dim=0)  
               
            test_cc = tensorKGE(allPred, allTrue)   
            self.testCcAvg.update(test_cc)              
            logging.info("pred:{} {}".format(np.around(pred.cpu().numpy().reshape(-1)[:10], decimals=1),
                                             pred_cls.cpu().numpy().reshape(-1)[:10]))  
            logging.info("true:{} {}".format(np.around(yTest_reg.cpu().numpy().reshape(-1)[:10], decimals=1),
                                             yTest_cls.cpu().numpy().reshape(-1)[:10]))                  
            return allTrue.cpu().numpy().reshape(-1), allPred.cpu().numpy().reshape(-1),allPred_cls.cpu().numpy().reshape(-1), allTrue_cls.cpu().numpy().reshape(-1)
    
    def predict(self):
        allTrue, allPred, allPred_cls, allTrue_cls = self.test()
        self.early_stopping(self.testCcAvg.accumulate().item())
        self.check_point(self.M, self.testCcAvg.accumulate().item())

        import csv
        if not os.path.exists(self.predict_path):
            os.makedirs(self.predict_path)
        file = open(os.path.join(self.predict_path,'2.csv'), 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(['num', 'time', 'true', 'pred']) 
        for row in zip(list(self.test_nums), list(self.test_times), list(allTrue), list(allPred)):
            writer.writerow(row) 
        file.close()
            
    def run(self):
                
        for epoch in range(self.nepoch):
            self.train()
            self.test()
            logging.info('epoch: [{}/{}] trainRegLoss: {:.4f} trainClsLoss: {:.4f} trainKGE: {:.4f} trainAcc: {:.4f}'
                         .format(self.nepoch, epoch+1, 
                                 self.trainLossAvg.accumulate().item(),
                                 self.trainCLossAvg.accumulate().item(),
                                 self.trainCcAvg.accumulate().item(),
                                 self.trainAccAvg.accumulate().item(),
                                 round(time.time() - self.startTimeTrain)
                                 ))
            
            logging.info('testRegLoss: {:.4f} testClsLoss: {:.4f} testKGE: {:.4f} testAcc: {:.4f} time: {}/s'
                         .format(
                                 self.testLossAvg.accumulate().item(),
                                 self.testCLossAvg.accumulate().item(),
                                 self.testCcAvg.accumulate().item(),
                                 self.testAccAvg.accumulate().item(),
                                 round(time.time() - self.startTimeTrain)
                                 ))
            
            self.early_stopping(self.testCcAvg.accumulate().item())
            self.check_point(self.M, self.testCcAvg.accumulate().item() )
            if self.early_stopping.stop:
                logging.info("Early stopping triggered.")
                logging.info("best_loss: {}".format(self.early_stopping.best_score))
                break;
            
            self.reset_loss()
            
    def build_model(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.M = MergePrecipitation(self.input_dim, self.sequence, self.size).to(self.device)
        
        # Loss and optimizer
        self.m_optimizer = torch.optim.AdamW(self.M.parameters(), self.lr)
        self.r_loss = RegLoss()
        self.c_loss = torch.nn.BCELoss()

    def build_log(self):
        from logger import logger_init
        logger_init(log_dir=self.log_path)

    def load_pretrained_model(self):
        path = os.path.join(self.config.model_save_path, self.version, self.pretrained_model)
        if os.path.exists(path):
            logging.info("Pre-trained model path: {}".format(path))
            self.M.load_state_dict(torch.load(path))
            self.test()
            self.early_stopping(self.testCcAvg.accumulate().item() )
            self.check_point(self.M, self.testCcAvg.accumulate().item())
            logging.info("## Successfully loaded existing model for additional training ......")

    def reset_grad(self):
        self.m_optimizer.zero_grad()
        
    def reset_loss(self):
        self.trainLossAvg.reset()
        self.trainCLossAvg.reset()
        self.trainAccAvg.reset()
        self.trainCcAvg.reset()
        
        self.testLossAvg.reset()
        self.testCcAvg.reset()    
        self.testAccAvg.reset()
        self.testCcAvg.reset()

