# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data

import pytorch_lightning as pl

import sys
from pathlib import Path
import os


#################################
#
#       CODE FOR THE MODEL
#
#################################


"""
    The ACNModules that the ACNe model consist of
    
    Constructor arguments
        in_channels number of inchannels
        out-channel - number of outchannels
        normalize - Should context normalization be applied? 
                    Should be set to false in last unit
"""


class ACNModule(torch.nn.Module):
    
    
    def __init__(self,in_channels,out_channels,normalize=True):
        super(ACNModule,self).__init__()
        self.localweight = torch.nn.Conv1d(out_channels, 1, 1)
        self.globalweight = torch.nn.Conv1d(out_channels, 1, 1)
        
        self.perceptron = torch.nn.Conv1d(in_channels,out_channels,1)
        
        self.relu = torch.nn.LeakyReLU()
        self.normalize=normalize
        
        
    def forward(self,features):
        #in size of features is (batch, in_channels,points )
        
        # transform features
        
        features = self.perceptron(features)
        features = self.relu(features)
        
        # local weights
        wl = self.localweight(features)
        wl = torch.sigmoid(wl)
        
        #global weights
        wg = self.globalweight(features)
        wg = torch.nn.functional.softmax(wg,dim=2)
        
        #mix and normalize
        
        w=wg*wl
        w = w/w.sum(2,keepdim=True)
        
        # context_normalize
        
        if self.normalize:
            features = features - (w*features).sum(dim=2,keepdim=True)
            features = features / torch.sqrt(((features**2*w).sum(dim=2,keepdim=True)+1e-6))
        
        
        return features,w



"""

        An "ACNe-"-model  for estimation of rotation.
        
        Constructor arguments
            layer_structure - an array of channel sizes
            lr - learning rate
    
"""
class Acne_RotEstimator(pl.LightningModule):
    def __init__(self,layer_structure,lr):
        super(Acne_RotEstimator,self).__init__()
        
        
        # initial perceptron
        self.initial_perceptron = torch.nn.Conv1d(layer_structure[0],layer_structure[1],1)
        
        # bulk of aCN-units
        self.aCN_units = torch.nn.ModuleList(ACNModule(layer_structure[i],layer_structure[i+1]) 
                                              for i in range(1,len(layer_structure)-2))
        
        # last aCN - no normalization for this model
        self.last_aCN= ACNModule(layer_structure[-2],layer_structure[-1], normalize=False)

        self.relu = torch.nn.ReLU()
        self.learning_rate=lr
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        
        # feed data through initial perceptron
        x= self.initial_perceptron(x)
        x= self.relu(x)
        
        # feed through all but last acn-unit
        for unit in self.aCN_units:
            x,_=unit(x)
            
        # last acn_unit
        x,w = self.last_aCN(x)
        # permutation invariaze through weighted average
        x = (x*w).sum(2)
        
        return x

    
    #pytorch lightning functions
    def my_step(self, batch, batch_idx):
        cloud1,cloud2,rot,_ = batch
        
        
        #concatenate coordinates in correspondences to two input channels
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
        
        self.log('val_pred_loss',loss,on_step=False, on_epoch=True)
    
        return loss

    def configure_optimizers(self):
        # use ADAM
        optimizer = torch.optim.Adam(self.parameters(),
                                     self.learning_rate)
        
        # set up scheduler as in paper
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,300], gamma=0.5),
            "monitor": "val_loss",
        }
    
#######################################
#
#       CODE FOR TRAINING/TESTING
#
#######################################
    
# boilerplate code for loaders and trainers for pytorch lightning
        
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
# run: python aCNE.py ratio data_name epochs lr seed
#
# =============================================================================


def train_model(config):
    

    model = Acne_RotEstimator(config['layer_structure'],config['learning_rate'])

    trainer, train_loader, val_loader = create_trainer_and_loaders(config)
    
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        print("Cuda not available. Training on CPU")

    trainer.fit(model, train_loader, val_loader)

    return model


def train_rotation_model(config,seed = 11211112):
    pl.seed_everything(seed) # for reproduceability
    
    # train model
    model=train_model(config)
    
    # save model
    name = 'trained_model_'+config['model_name'] +'_'+os.path.split(config['dataset_path'])[-1].replace(".pt",'')+'.pt'
    torch.save(model.state_dict(),os.path.join("trained_models", name))


def main(user_input):
    

    log_dir = user_input[0]
    data = user_input[1]
    name = user_input[2]
    epochs = user_input[4]
    lr = user_input[5]
    seed = user_input[6]    


    print('Training ' + name + ' model on ' + data )
    if len(user_input)<6:
        seed = 11211112 #default seed
    else:
        seed = user_input[5]
    
    #layer structure.
    # If you want to test other sizes, this is the thing to change
    layer_structure = [4,32,32,64,64,32,32,2]


        


    
        
    config = {
            "batch_size": 16,
            "num_workers": 8,
            
            "learning_rate": lr,
            "momentum": 0,
            "num_epochs": epochs,
            
            "log_dir": Path.cwd() / log_dir,
            "dataset_path": Path.cwd() / data,
    
            "layer_structure" : layer_structure,
            
            "model_name": name
            
            }
    
    
    torch.autograd.set_detect_anomaly(True)
    train_rotation_model(config,seed)


if __name__ == '__main__':

    user_input=[None,None,None,None,None,None,None]

    name = 'acne'
    
    ratio = float(sys.argv[1])
    
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
        
    if len(sys.argv)<5:
        lr= 1e-3
    else:
        lr = float(sys.argv[4])
        
    if len(sys.argv)<6:
        seed = 11211112 
    else:
        seed = int(sys.argv[5])
        
        
    user_input[0] = os.path.join('logs',name+'_'+str(ratio))
    user_input[1] = data
    user_input[2] = name
    user_input[3] = .1 
    user_input[4] = epochs
    user_input[5] = lr
    user_input[6] = seed
	   
    main(user_input)




