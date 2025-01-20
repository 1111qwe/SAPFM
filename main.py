from parameter import get_parameters
from trainer import Trainer    
from data_loader import DataLoaders
config = get_parameters()
dataLoader = DataLoaders(config.train_path, config.test_path)
train = Trainer(dataLoader,config)
# train.run()
train.predict()