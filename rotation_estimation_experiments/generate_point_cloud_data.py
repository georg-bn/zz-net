# -*- coding: utf-8 -*-

import torch
import numpy as np
from linearlayers import cmultiply
import sys
import os


def random_point_on_disk(N):
    r = torch.rand(N)
    t = torch.rand(N)*2*np.pi
    
    output=torch.empty(N+tuple([2]))
    output[...,0] = torch.sqrt(r)*torch.cos(t)
    output[...,1] = torch.sqrt(r)*torch.sin(t)
    
    return output
    
def generate_noisy_triangles(outlier_ratio, nbr_points, nbr_clouds,inlier_noise=.01):
    
    with torch.no_grad():
     # generate the three corners defining the triangle
     # Note: currently in format (clouds,2,nbr_points)!
     corners = random_point_on_disk((nbr_clouds,3)).transpose(1,2)
     # embed them in R^3, z=1
     corners = torch.cat((corners, torch.ones(nbr_clouds,1,3)),1)
     
     # for picking out pairs of points
     choice = torch.zeros(3,1,1,3)
     
     choice[0,:,:,0]=1
     choice[0,:,:,1]=1
     choice[1,:,:,0]=1
     choice[1,:,:,2]=1
     choice[2,:,:,1]=1
     choice[2,:,:,2]=1
     
     # for rotating
     
     # generate rotation
     
     theta = torch.randn(nbr_clouds,1,2)
     theta = theta/torch.sqrt((theta**2).sum(2)).unsqueeze(2)
     
     # create_inliers
     inliers = random_point_on_disk((3,nbr_clouds,nbr_points)).transpose(2,3)
     inliers = torch.cat((inliers,torch.ones(3,nbr_clouds,1,nbr_points)),2)
     
     for k in range(3):
         
         definers = choice[k,...]*corners
         U , _, _ = torch.svd(definers@definers.transpose(1,2))


         # orthogonal line
         y = U[:,:,-1].unsqueeze(2)

         # project inliers
         cof = (inliers[k,...]*y).sum(1)/(y[:,:2,:]**2).sum(1)
         inliers[k,...] = inliers[k,...] - cof.unsqueeze(1)*y

     # choose lines for inliers
     crit = torch.rand(nbr_clouds,nbr_points).unsqueeze(1)
     
     inliers = inliers[0,...]*(crit<1/3) + inliers[1,...]*(crit>1/3)*(crit<2/3) +\
               inliers[2,...]*(crit>2/3)
               
     # project back to R^2 and add noise
     
     inliersA = inliers[:,:-1,:] +  inlier_noise*torch.randn(nbr_clouds,2,nbr_points)
     inliersB = cmultiply(theta,inliersA.transpose(1,2)).transpose(1,2) + inlier_noise*torch.randn(nbr_clouds,2,nbr_points)
     
     # create outliers
     outliersA = random_point_on_disk((nbr_clouds,nbr_points)).transpose(1,2)
     outliersB = random_point_on_disk((nbr_clouds,nbr_points)).transpose(1,2)
     
     # create mask

     mask = (torch.rand(nbr_clouds,nbr_points)>outlier_ratio).float().unsqueeze(1)

     # choose points and reshape

     pointsA = (1.0 - mask)*outliersA +  mask*inliersA
     pointsB = (1.0 - mask)*outliersB +  mask*inliersB
     cloud1 = pointsA.transpose(1,2)
     cloud2 = pointsB.transpose(1,2)
     
     return (cloud1,cloud2),(theta,mask.transpose(1,2))
     

def save_data(train_data,
              train_labels,
              test_data,
              test_labels,
              filename):
    torch.save((train_data, train_labels, test_data, test_labels), filename)
    
def generate_and_save_pair_data(user_input):
    
    name, outlier_ratios, nbr_points, nbr_train, nbr_val, nbr_test,inlier_noise = user_input

    
    for ratio in outlier_ratios:
        #train and validation data
        train_data, train_labels = generate_noisy_triangles(outlier_ratio = ratio, 
                                                        nbr_points = nbr_points,
                                                        nbr_clouds = nbr_train,
                                                        inlier_noise = inlier_noise)
        
        val_data, val_labels = generate_noisy_triangles(outlier_ratio = ratio, 
                                                        nbr_points = nbr_points,
                                                        nbr_clouds = nbr_val,
                                                        inlier_noise = inlier_noise)
        
        # test_data
        test_data, test_labels = generate_noisy_triangles(outlier_ratio = ratio, 
                                                        nbr_points = nbr_points,
                                                        nbr_clouds = nbr_test,
                                                        inlier_noise = inlier_noise)
        
        # save test_and_val data
        torch.save((train_data, train_labels, test_data, test_labels), name+'_'+str(ratio) +'.pt')
        
        # save testdata 
        
        torch.save(( test_data, test_labels), name+'_'+str(ratio)+ '_test.pt')
        


if __name__ == '__main__':
    
    #python generate_point_cloud_data.py name ratio nbr_points nbr_train nbr_val n_test  inlier noise
    
    # take in user input.
    # if 'paper' -use same parameters as in paper
    if sys.argv[1]=='paper':
        name = 'control_rotated_pairs_'
        ratios = [0.4, 0.6, 0.8, 0.85]
        nbr_points = 100
        nbr_train = 2000
        nbr_val = 500
        nbr_test =300
        inlier_noise = .03
    else:
    
        if len(sys.argv)<2:
            name = 'rotated_pairs_'
        else:
            name = sys.argv[1]
        
        if len(sys.argv)<3:
            ratios = [0.4,0.6,0.8,0.85]
        else:
            ratios = [ float(sys.argv[2])]
        
        if len(sys.argv)<4:
            nbr_points = 100
        else:
            nbr_points = int(sys.argv[3])
            
        if len(sys.argv)<5:
            nbr_train = 2000
        else:
            nbr_train = int(sys.argv[4])
        
        if len(sys.argv)<6:
            nbr_val =500
        else:
            nbr_val = int(sys.argv[5])
        
        if len(sys.argv)<7:
            nbr_test =300
        else:
            nbr_test = int(sys.argv[6])
        
        if len(sys.argv)<8:
            inlier_noise = 0.03
        else:
            inlier_noise = float(sys.argv[7])
            
            
    user_input = [None]*7
        
    user_input[0] = os.path.join('data',name)
    user_input[1] = ratios
    user_input[2] = nbr_points
    user_input[3] = nbr_train
    user_input[4] = nbr_val
    user_input[5] = nbr_test
    user_input[6] = inlier_noise
        
    
    generate_and_save_pair_data(user_input)

