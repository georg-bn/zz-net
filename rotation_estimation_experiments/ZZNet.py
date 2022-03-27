import torch
from linearlayers import EquivTensorLayer, EquivVectorLayer, tensorify_z,cmultiply
from nonlinearities import RadRelu
import pytorch_lightning as pl
import sys
from pathlib import Path
import os


####
#
#    CODE FOR THE MODEL
#
####

class ZZRotEstimator(pl.LightningModule):
    
    """
        A ZZNet for estimation of rotation.
        
        Constructor arguments
            early_layer_structure - An array of sizes of early layers of ZZ-units
            late_layer_structure - An array of sizes of late layers of ZZ-units
            vector_layer_structure - An array of sizes of vector layers of ZZ-units
            full_weights - if True, fully real-linear layers are used
                           if False, complex-linear layers are used. If complex_weights=False,
                                    the coefficients of them are real.
            weight_init_gain - parameter for initialization of layers
            tau - initialization of the learnable parameters of the complex ReLUs.
            learning_rate - learning rate
            
    """
    
    
   
    
    def __init__(self,early_layer_structure,late_layer_structure,
                 vector_layer_structure,
                 complex_weights=False,bias=False,
                 full_weights=False,
                 weight_init_gain=1,tau=.1,learning_rate=5e-3):
        
        super(ZZRotEstimator,self).__init__()
        
        ## Initialize layers
        
        self.early_layers_A = torch.nn.ModuleList(
                            torch.nn.ModuleList(EquivTensorLayer(early_layer_structure[k][i],early_layer_structure[k][i+1],
                                             weight_init_gain=weight_init_gain,bias=bias,
                                             complex_weights=complex_weights,
                                             full_weights=full_weights)
                            for i in range(len(early_layer_structure[k])-1))
                            for k in range(len(early_layer_structure))
                            )
        
        self.late_layers_A = torch.nn.ModuleList(
                            torch.nn.ModuleList(EquivVectorLayer(late_layer_structure[k][i],late_layer_structure[k][i+1],
                                             complex_weights=complex_weights, bias=bias,
                                             weight_init_gain=weight_init_gain, full_weights=full_weights)
                            for i in range(len(late_layer_structure[k])-1))
                            for k in range(len(late_layer_structure))
                            )
        
        self.vector_layers = torch.nn.ModuleList(
                            torch.nn.ModuleList(EquivVectorLayer(vector_layer_structure[k][i],vector_layer_structure[k][i+1],
                                             complex_weights=True,bias=False,
                                             weight_init_gain=weight_init_gain,full_weights=False)
                            for i in range(len(vector_layer_structure[k])-1))
                            for k in range(len(vector_layer_structure))
                            )
        
        self.early_layers_B = torch.nn.ModuleList(
                            torch.nn.ModuleList(EquivTensorLayer(early_layer_structure[k][i],early_layer_structure[k][i+1],
                                             weight_init_gain=weight_init_gain,
                                             complex_weights= complex_weights, bias=bias,full_weights=full_weights)
                            for i in range(len(early_layer_structure[k])-1))
                            for k in range(len(early_layer_structure))
                            )
        
        self.late_layers_B = torch.nn.ModuleList(
                            torch.nn.ModuleList(EquivVectorLayer(late_layer_structure[k][i],late_layer_structure[k][i+1],
                                             weight_init_gain=weight_init_gain,
                                             complex_weights=complex_weights, bias=bias,full_weights=full_weights)
                            for i in range(len(late_layer_structure[k])-1))
                            for k in range(len(late_layer_structure))
                            )
        
        
        
        # non-linearity for weight units
        self.lrelu =  torch.nn.ReLU()
        
        # non-linearity for vector unit
        self.relus = torch.nn.ModuleList(torch.nn.ModuleList(RadRelu(vector_layer_structure[k][i],tau)
                                      for i in range(1,len(vector_layer_structure[k])-1))
                                      for k in range(len(vector_layer_structure))
                                      )
        
        self.learning_rate = learning_rate
        self.loss_function = torch.nn.MSELoss()

    
        
    def forward(self,cloud1,cloud2):
        
        # go from None channels to 1 channel, if necessary
        
        if cloud1.dim()<4:
            cloud1=cloud1.unsqueeze(1)
        
        if cloud2.dim()<4:
            cloud2=cloud2.unsqueeze(1)

        
        for i in range(len(self.early_layers_A)-1):
            # apply the unit to get weights and clouds
            
            weight1, weight2, cloud1, cloud2 = self.applySubUnit(i,cloud1,cloud2)

            
            
            # reweight the points
            cloud1 = cmultiply(weight1,cloud1)
            cloud2 = cmultiply(weight2,cloud2)           
            
            
            # renormalize
            cloud1 = cloud1/torch.sqrt((cloud1**2).sum((2,3),keepdim=True)) 
            cloud2 = cloud2/torch.sqrt((cloud2**2).sum((2,3),keepdim=True))
                    
        #apply last unit 
        weight1, weight2, cloud1, cloud2 = self.applySubUnit(-1,cloud1,cloud2)
        
        #renormalize weights
        
        weight1 = weight1/torch.sqrt((weight1**2).sum((2,3),keepdim=True))
        weight2 = weight2/torch.sqrt((weight2**2).sum((2,3),keepdim=True))
        
        
        # contract the weights and clouds
        
        cloud1 = (cmultiply(cloud1,weight1)).sum(2)
        cloud2 = (cmultiply(cloud2,weight2)).sum(2)
        
        # multiply the outputs
        
        out = cmultiply(cloud2,cloud1,True)
        
        return out
    
    def applySubUnit(self,i,cloud1,cloud2):
        
        # go from cloud to cloud cloud*
        weight1 = tensorify_z(cloud1)
        weight2 = tensorify_z(cloud2)
        
      
        
        # apply the early ('tensor') layers of the weight unit
        for k in range(len(self.early_layers_A[i])-1):
            temp_weight1 = self.early_layers_A[i][k](weight1) + self.early_layers_B[i][k](weight2)
            weight2 = self.early_layers_A[i][k](weight2) + self.early_layers_B[i][k](weight1)
            
            weight1 = self.lrelu(temp_weight1)
            weight2 = self.lrelu(weight2)
        
        temp_weight1 = self.early_layers_A[i][-1](weight1) + self.early_layers_B[i][-1](weight2)
        weight2 = self.early_layers_A[i][-1](weight2) + self.early_layers_B[i][-1](weight1)
        weight1 = temp_weight1
        
        # invariaze the tensors to vectors
        
        weight1 = weight1.sum(3)
        weight2 = weight2.sum(3)
        
        # apply the late ('vector') layers of the weight unit
        for k in range(len(self.late_layers_A[i])-1):
            temp_weight1 = self.late_layers_A[i][k](weight1) + self.late_layers_B[i][k](weight2)
            weight2 = self.late_layers_A[i][k](weight2) + self.late_layers_B[i][k](weight1)
            
            
            weight1 = self.lrelu(temp_weight1)
            weight2 = self.lrelu(weight2)
            
         
        temp_weight1 = self.late_layers_A[i][-1](weight1) + self.late_layers_B[i][-1](weight2)
        weight2 = self.late_layers_A[i][-1](weight2) + self.late_layers_B[i][-1](weight1)
        weight1 = temp_weight1
        

       
        # apply the vector unit
        for k in range(len(self.vector_layers[i])-1):
            cloud1 = self.vector_layers[i][k](cloud1)
            cloud2 = self.vector_layers[i][k](cloud2)
            cloud1 = self.relus[i][k](cloud1)
            cloud2 = self.relus[i][k](cloud2)
            
            
        cloud1 = self.vector_layers[i][-1](cloud1)
        cloud2 = self.vector_layers[i][-1](cloud2)
        
        
        return weight1, weight2, cloud1, cloud2
    
    #pytorch lightning functions
    def my_step(self, batch, batch_idx):
        cloud1,cloud2,rot,mask = batch
        
        pred = self.forward(cloud1,cloud2)
        
        loss = self.loss_function(pred,rot)        

        return loss
    
    def training_step(self,batch,batch_idx):
        
        loss = self.my_step(batch,batch_idx)

        
        self.log('train_pred_loss',loss)
         
        return loss   
    def validation_step(self,batch,batch_idx):
        
        loss = self.my_step(batch,batch_idx)
        
        self.log('val_pred_loss',loss,on_step=False, on_epoch=True)
       
        
        return loss

    def configure_optimizers(self):
         
        # Use Adam for optimization
        optimizer = torch.optim.Adam(self.parameters(),
                                    self.learning_rate)        
        
        # Schedule a reduction of learning rate as in paper.
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[70,150],gamma=.5,verbose=True),
            "monitor": "val_pred_loss",
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
                             deterministic=True, # for reproducability
                             terminate_on_nan=True,
                             )
    else:
        trainer = pl.Trainer(max_epochs=config["num_epochs"],
                             logger=tb_logger,
                             callbacks=callbacks,
                             deterministic=True, # for reproducability
                             terminate_on_nan=True,
                             )
    return trainer, train_loader, val_loader

    

#######################################
#
#       CONFIGURE AND LAUNCH
#
#
# run: python full_rotation_net.py name ratio
#
# =============================================================================


def train_model(config):
    # The reason for using a config dict here is to facilitate
    # parameter tuning with ray tune.
    
    model = ZZRotEstimator(config["els"], config["lls"],config['vls'], 
                                 weight_init_gain=config["gain"],
                                 complex_weights=config["cw"],
                                 full_weights = config["fw"],
                                 bias=config["bias"],
                                 tau=config["tau"],
                                 learning_rate=config["learning_rate"])

    trainer, train_loader, val_loader = create_trainer_and_loaders(config)
    
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        print("Cuda not available. Training on CPU")

    trainer.fit(model, train_loader, val_loader)

    return model


def train_rotation_model(config,seed =  11211112):
    pl.seed_everything(seed) # for reproduceability, default seed used in experiments for paper
    
    # train model
    model=train_model(config)

    # save model
    name = 'trained_model_'+config['model_name'] +'_'+os.path.split(config['dataset_path'])[-1].replace(".pt",'')+'.pt'
    torch.save(model.state_dict(),os.path.join("trained_models", name))




def main(user_input):
    
    # read user-specified parameters
    log_dir = user_input[0]
    data = user_input[1]
    name = user_input[2]
    tau = float(user_input[3])
    epochs = int(user_input[4])
    lr = float(user_input[5])
    seed = user_input[6]
    

    print('Training ' + name + ' model on ' + data )
    

    # the config are defined  here, accessed through names.
    if name == "broad":
        
        early_layer_structure = [[1,4,4]]
        late_layer_structure  = [[4,16,4,1]]
        vector_layer_structure =[[1,32,1]]
        tau=.1
        g=1       
        complex_weights = True
        bias = True
        full_weights=True

    elif name == "deep":
        early_layer_structure = [[1,4],   [4,4],   [4,4]]
        late_layer_structure  = [[4,8,4], [4,8,4], [4,8,1]]
        vector_layer_structure =[[1,4],   [4,4],   [4,1]]
        tau=.1
        g=1       
        complex_weights = True
        bias = True
        full_weights=True
        
        
        ## If you want to build your own model, this is the place to do it!
    
    else:
        print("Model name not known.")
        print("Aborting")
        return

    
    # config the experiment as in paper
    config = {
            "batch_size": 24,
            "num_workers": 8,
            
            "learning_rate": lr,
            "num_epochs": epochs,
            
            "log_dir": Path.cwd() / log_dir,
            "dataset_path": Path.cwd() / data,
            
            "tau": tau,
            "gain": g,
            
            "els": early_layer_structure,
            "lls": late_layer_structure,
            "vls": vector_layer_structure,
            "cw" : complex_weights,
            "fw" : full_weights,
            "bias": bias,
            
            "model_name": name
            
            }
    
    
    torch.autograd.set_detect_anomaly(True)
    train_rotation_model(config,seed)
    
    


if __name__ == '__main__':
    
    # run python ZZNet.py name ratio data_name epochs lr seed
    
   
    # read user_specified parameters
    # if non, set to default (as in paper)
    
    user_input=[None,None,None,None,None,None,None]
    name = sys.argv[1]
    
    ratio = float(sys.argv[2])
    
    data = sys.argv[3]
    # if user specifies a ratio, user pre-generated data
    if data == 'paper':
        data = os.path.join("data", "train_valid_data_paper", 'rotated_pairs_'+str(ratio)+'.pt')
    else:
        data = os.path.join("data",data)
    
    if len(sys.argv)<5:
        epochs = 300
    else:
        epochs = int(sys.argv[4])
        
    if len(sys.argv)<6:
        lr= 5e-3
    else:
        lr = float(sys.argv[5])
        
    if len(sys.argv)<7:
        seed = 11211112 
    else:
        seed = int(sys.argv[6])
    

    user_input[0] = os.path.join('logs',name+'_'+str(ratio))
    user_input[1] = data
    user_input[2] = name
    user_input[3] = .1 
    user_input[4] = epochs
    user_input[5] = lr
    user_input[6] = seed
	   
    main(user_input)
    




