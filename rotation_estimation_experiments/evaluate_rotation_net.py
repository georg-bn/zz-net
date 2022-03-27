# -*- coding: utf-8 -*-

import torch
from ZZNet import ZZRotEstimator
from aCNE import Acne_RotEstimator
import sys
import os
import numpy as np
import pointNet


def load_ZZ_model(filepath,modeltype):
    
    #get modelsizes
    # If you have built your own model type, this needs to be adjusted!
    if modeltype =="broad":
        early_layer_structure = [[1,4,4]]
        late_layer_structure  = [[4,16,4,1]]
        vector_layer_structure =[[1,32,1]]
        
        tau=.1
        g=1       
        complex_weights = True
        bias = True
        full_weights=True
        
    elif modeltype == "deep":
        early_layer_structure = [[1,4],   [4,4],   [4,4]]
        late_layer_structure  = [[4,8,4], [4,8,4], [4,8,1]]
        vector_layer_structure =[[1,4],   [4,4],   [4,1]]
        tau=.1
        g=1       
        complex_weights = True
        bias = True
        full_weights=True
        
    else:
        sys.exit("ZZ-type not known. Aborting")
    
    
    model = ZZRotEstimator(early_layer_structure,late_layer_structure,
                 vector_layer_structure,
                 complex_weights,full_weights=full_weights, bias=bias)
    
    model.load_state_dict(torch.load(filepath))
    
    model.eval()
    
    if torch.cuda.is_available():
        model.to("cuda")
    
    return model
    

def load_point_net(filepath):
    
    model= pointNet.Pointnet_RotEstimator(0)
    
    model.load_state_dict(torch.load(filepath))

    model.eval()
    
    if torch.cuda.is_available():
        model.to("cuda")
    
    return model

def load_acne_model(filepath):
    
    #get modelsize
    # If you have built your own model type, this needs to be adjusted!
    layer_structure = [4,32,32,64,64,32,32,2]

    model= Acne_RotEstimator(layer_structure, 0)
    
    model.load_state_dict(torch.load(filepath))
    model.eval()
    
    if torch.cuda.is_available():
        model.to("cuda")
        
    return model
    

def create_loader(datapath):
    test_data, test_labels = torch.load(
        datapath)

    
    test_ds = torch.utils.data.TensorDataset(test_data[0], test_data[1], test_labels[0],test_labels[1])            
            
       
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=20,
        shuffle=False,
        num_workers=8)
    return test_loader

def evaluate_model(model,datapath,mtype):
    
    # create loader of correct type
    dataloader= create_loader(datapath)
    
    loss_function = torch.nn.MSELoss(reduction='none')
    
    # thresholds for errors
    thres=[2-2*np.cos(1/180*np.pi),2-2*np.cos(5/180*np.pi),
           2-2*np.cos(10/180*np.pi)]
    
    if torch.cuda.is_available():
        dvc="cuda"
    else:
        print("Cuda not available. Testing on CPU")
        dvc="cpu"
    

    print("Testing...")
    succ=[0,0,0]
    l=0
    # go through data set
    for cloud0,cloud1,true_rot,_ in dataloader:

        l+=1
        
        # load data in correct fashion for type
        if mtype == 'pointnet' or mtype =='acne':
            data = torch.cat((cloud0,cloud1),2)
            pred = model(data.transpose(1,2)).unsqueeze(1)
        elif mtype =='ZZ':
            pred = model(cloud0.to(dvc),cloud1.to(dvc))
        else:
            pred,_ = model(cloud0.to(dvc),cloud1.to(dvc))
        
        # feed through network and check how often error is under threshold
        err = loss_function(pred,true_rot.to(dvc)).sum(-1)
        
        for k,t in enumerate(thres):
            succ[k] += (err<t).sum()
            

    print('============')
    print('Results:')
    print('1 degree: ' +str(succ[0]/len(dataloader.dataset)))
    print('5 degrees: ' +str(succ[1]/len(dataloader.dataset)))
    print('10 degrees: ' +str(succ[2]/len(dataloader.dataset)))

        
        
        
        
        

    
if  __name__ == '__main__':
    
    # get modelname and dataname
    # automatically append directory structure
    
    # default: python evaluate_rotation_net.py paper ratio type
    # other: python evaluate_rotation_net.py modelname dataname type
    
    # type : zd,zb,a,p
    
    
    # choose modeltype
    if sys.argv[3]=='p':
        mtype = 'pointnet'
    elif sys.argv[3]=='zb':
        mtype = 'ZZ'
        subtype = 'broad'
    elif sys.argv[3]=='zd':
        mtype = 'ZZ'
        subtype = 'deep'
    elif sys.argv[3]=='a':
        mtype = 'acne'
    else:
        sys.exit("Modeltype not known. Aborting")

    
    # define paths for model and data
    if sys.argv[1] =='paper': # default
        ratio = str(float(sys.argv[2]))
        if mtype =='ZZ':
            modelpath = os.path.join('trained_models','pretrained','trained_model_' + subtype + '_'+ratio +'.pt')
        else: 
            modelpath = os.path.join('trained_models','pretrained','trained_model_' + mtype + '_'+ratio +'.pt')
            
        datapath = os.path.join('data','test_data_paper','rotated_pairs_' + ratio + '_test.pt')
    else:
        modelpath = os.path.join('trained_models',sys.argv[1])
        datapath = os.path.join('data',sys.argv[2])
    if mtype == 'pointnet':
        model= load_point_net(modelpath)
    elif mtype == 'ZZ':
        model = load_ZZ_model(modelpath,subtype)
    elif mtype == 'acne':
        model = load_acne_model(modelpath)

    
    evaluate_model(model,datapath,mtype)
    