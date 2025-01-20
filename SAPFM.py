import torch.nn as nn
import torch.nn.functional as F 
import torch
from SelfLSTM import SelfLSTM
import torchvision.transforms as transforms

class BatchNormal(nn.Module):
    def __init__(self, num_features):
        super(BatchNormal, self).__init__()
        self.m = nn.BatchNorm3d(num_features, affine=False)
        
    def forward(self, x):
        #(b, t, c, h, w)
        x = x.permute(0, 2, 1, 3, 4) #(b, t, c, h, w) -> #(b, c, t, h, w)
        x = self.m(x)
        x = x.permute(0, 2, 1, 3, 4) #(b, c, t, h, w) -> #(b, t, c, h, w)
        return x

class biSelfLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1, batch_first=True, bidirectional=False, activition='tanh'):
        super(biSelfLSTM, self).__init__()
        self.selfLSTM = SelfLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.embedding = nn.Conv3d(in_channels=hidden_dim * 2, out_channels= hidden_dim, kernel_size=1)
        if activition == 'tanh':
            self.activition = nn.Tanh()
        elif activition == 'relu':
            self.activition = nn.ReLU() 
    def forward(self, x):
        
        h ,_ = self.selfLSTM(x)
        # h = self.activition(h)
        h = h.permute(0,2,1,3,4)
        h = self.embedding(h)
        h = h.permute(0,2,1,3,4)
        return h

class MergePrecipitation(nn.Module):
    def __init__(self, input_dim, sequence, size):
        super(MergePrecipitation, self).__init__()
        # print('3',input_dim, sequence, size)
        
        self.convLSTM1 = nn.Sequential(
            biSelfLSTM(input_dim=input_dim, hidden_dim=8, kernel_size=(1,1), bidirectional=True, activition='relu'),
            BatchNormal(8),
            biSelfLSTM(input_dim=8, hidden_dim=8, kernel_size=(1,1), bidirectional=True, activition='relu'),
            BatchNormal(8),             
        )

        self.convLSTM2 = nn.Sequential(
            biSelfLSTM(input_dim=8, hidden_dim=4, kernel_size=(1,1), bidirectional=True, activition='relu'),
            BatchNormal(4),
            biSelfLSTM(input_dim=4, hidden_dim=4, kernel_size=(1,1), bidirectional=True, activition='relu'),
            BatchNormal(4),       
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features= 4*sequence*size*size, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
        )
        
        self.reg = nn.Linear(in_features=256, out_features=1)
        self.cls = nn.Sequential(nn.Linear(in_features=256, out_features=1),
                                 nn.Sigmoid())
    def forward(self, x):
        
        x1 = self.convLSTM1(x)
        x2 = self.convLSTM2(x1)
        x2 = x2.flatten(start_dim=1)
        x2 = self.fc(x2)
        
        reg = self.reg(x2)
        cls = self.cls(x2)
        return reg, cls
    
if __name__ == "__main__":
    model = MergePrecipitation(input_dim=7, sequence=11, size=3)
    x = torch.rand((1, 11, 7, 3, 3)) #(b, t, c, h, w)
    reg,cls =model(x)
    print(reg.shape, cls.shape)