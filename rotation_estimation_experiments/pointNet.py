# -*- coding: utf-8 -*-
"""
Pointnet implementation inspired by github repository https://github.com/fxia22/pointnet.pytorch

"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

import pytorch_lightning as pl

import sys
import os
from pathlib import Path


#################################
#
#       CODE FOR THE MODEL
#
#################################

"""
   

        A PointNet for estimation of rotation.
        
        The only constructor argument is the learning rate,
        structure from experiments in paper hardcoded
            
    
"""

class Pointnet_RotEstimator(pl.LightningModule):
    
    def __init__(self,lr):
        super(Pointnet_RotEstimator,self).__init__()
        
        # permutation equivariant layers
        self.conv1 = torch.nn.Conv1d(4, 32, 1)
        self.conv1_5 = torch.nn.Conv1d(32, 64, 1)
        self.conv2 = torch.nn.Conv1d(64,128, 1)
        self.conv3 = torch.nn.Conv1d(128, 64, 1)
        self.conv4 = torch.nn.Conv1d(64,64, 1)
        self.conv5 = torch.nn.Conv1d(64,64, 1)

        # head 
        self.fc1 = torch.nn.Linear(64,64)
        self.fc2 = torch.nn.Linear(64,32)
        self.fc3 = torch.nn.Linear(32,16)
        self.fc4 = torch.nn.Linear(16,2)        

        self.relu = nn.ReLU()
        
        # learnable batchnorm-layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(64)
        

        
        self.learning_rate=lr
        self.loss_function = torch.nn.MSELoss()


    def forward(self, x):
        
        # send data through permutation equivariant layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv1_5(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.conv3(x)))
        x = F.relu(self.bn5(self.conv4(x)))
        x = F.relu(self.bn6(self.conv5(x)))
        
        
        # max pool
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 64)
        
        # feed through
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)


        return x

    
    #pytorch lightning functions
    def my_step(self, batch, batch_idx):
        cloud1,cloud2,rot,_ = batch
        
        
        # concatenate the point coordinates to four input channels
        data= torch.cat((cloud1,cloud2),2)
        pred = self.forward(data.transpose(1,2)).unsqueeze(1)
        
        loss = self.loss_function(pred,rot)
       

        return loss
    
    def training_step(self,batch,batch_idx):
        
        loss = self.my_step(batch,batch_idx)
        
        self.log('train_loss', loss)
         
        return loss
    
    def validation_step(self,batch,batch_idx):
        
        loss = self.my_step(batch,batch_idx)
        
        self.log('val_pred_loss',loss)

    
        return loss

    def configure_optimizers(self):
        # Use SGD with momentum = 0.9, as in paper
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.learning_rate,momentum=.9) 
        
        return {
            "optimizer": optimizer,
            "monitor": "val_loss",
        }
    
#######################################
#
#       CODE FOR TRAINING/TESTING
#
#######################################

# boilerplate code for constructing loaders and trainers for pytorch lightning
       
def create_loaders(config):
    train_data, train_labels, test_data, test_labels = torch.load(
        config["dataset_path"])

    train_ds = torch.utils.data.TensorDataset(train_data[0], train_data[1],train_labels[0],train_labels[1])
    val_ds = torch.utils.data.TensorDataset(test_data[0], test_data[1], test_labels[0],test_labels[1])            
            
       
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"])
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"])
    return train_loader, val_loader

def create_logger(config):
    if config["log_dir"]:
        log_dir = config["log_dir"]
    else:
        log_dir = Path.cwd()
    tb_logger = pl.loggers.TensorBoardLogger(log_dir)
    return tb_logger


def create_trainer_and_loaders(config):
    
    train_loader, val_loader = create_loaders(config)

    callbacks = []

    tb_logger = create_logger(config)

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=1,
                             max_epochs=config["num_epochs"],
                             logger=tb_logger,
                             callbacks=callbacks,
                             deterministic=True,
                             terminate_on_nan=True)
                             
    else:
        trainer = pl.Trainer(max_epochs=config["num_epochs"],
                             logger=tb_logger,
                             callbacks=callbacks,
                             deterministic=True,
                             terminate_on_nan=True)

    return trainer, train_loader, val_loader


#######################################
#
#       CONFIGURATE AND LAUNCH
#
#
# run: python pointNet.py 
#
# =============================================================================


def train_model(config):
    # The reason for using a config dict here is to facilitate
    # parameter tuning with ray tune.
    

    model = Pointnet_RotEstimator(config['learning_rate'])

    trainer, train_loader, val_loader = create_trainer_and_loaders(config)
    
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        print("Cuda not available. Training on CPU")

    trainer.fit(model, train_loader, val_loader)

    return model


def train_rotation_model(config,seed = 187438):
    pl.seed_everything(seed) # for reproduceability
    
    # train model
    model=train_model(config)

    # save model
    name = 'trained_model_'+config['model_name'] +'_'+os.path.split(config['dataset_path'])[-1].replace(".pt",'')+'.pt'
    torch.save(model.state_dict(),os.path.join("trained_models", name))




def main(user_input):
    

    log_dir = user_input[0]
    name = user_input[1]
    data = user_input[2]
    tau = user_input[3]
    epochs = user_input[4]
    lr = user_input[5]
    seed = user_input[6]    


        
    print('Training ' + name + ' model on ' + data )

    
        
    config = {
            "batch_size": 16,
            "num_workers": 8,
            
            "learning_rate": lr,
            "num_epochs": epochs,
            
            "log_dir": Path.cwd() / log_dir,
            "dataset_path": Path.cwd() / data,
            
            "tau": tau,
            "gain": 1,
            
            "model_name": name
            
            }
    
    
    torch.autograd.set_detect_anomaly(True)
    train_rotation_model(config,seed)


if __name__ == '__main__':
    
    # run python pointNet.py ratio data_name epochs lr seed
    
   
    # read user_specified parameter
    
    ratio = float(sys.argv[1])
    
    name = 'pointnet'
    
    data = sys.argv[2]
    
    # if user specifies a ratio, user pre-generated data
    if data == 'paper':
        data = os.path.join("data", "train_valid_data_paper", 'rotated_pairs_'+str(ratio)+'.pt')
    else:
        data = os.path.join("data",data)
    
    if len(sys.argv)<4:
        epochs = 400
    else:
        epochs = int(sys.argv[3])
        
    if len(sys.argv)<6:
        lr= 1e-3
    else:
        lr = float(sys.argv[4])
        
    if len(sys.argv)<7:
        seed = 11211112 
    else:
        seed = int(sys.argv[5])
   
    
    
    
    user_input=[None,None,None,None,None,None,None]
    
    user_input[0] = os.path.join('logs','pointNet_'+str(ratio))
    user_input[1] = name + '_' + str(ratio)
    user_input[2] = data
    user_input[3] = .1 
    user_input[4] = epochs
    user_input[5] = lr
    user_input[6] = seed


	   
    main(user_input)

    



