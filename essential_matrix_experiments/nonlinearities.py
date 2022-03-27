import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import init
# import numpy as np

# =============================================================================
#
#  Two rotationally invariant non-linearties
#  > RadRelu - the radial relu
#  > VecRelu - the directional relu, as in "Vector Neurons"
#
#  Loss function for the direction for the line-fitting
#  > DirectionalLoss
#
#  Weighted least squares
#  
# =============================================================================


class RadRelu(torch.nn.Module):

    """
        The radial relu.
            x -> pos(|x|-t)/|x|*x

        Input: point cloud (batch_size, n_in,m,d)
        Output: point cloud (batch_size, n_in,m,d)
        #Input:  (batch_size, n_in, d)
        #Output: (batch_size, n_in, d)

        Input parameters:
            n_in - number of point clouds
            tau - initial thres-value

        Attributes:
            thres - the ReLU-cutoff point
    """
    def __init__(self, n_in, tau=.5):
        super(RadRelu, self).__init__()
        self.thres = torch.nn.Parameter(torch.empty((1, n_in),
                                                    dtype=torch.float))
        self.reset_parameters(tau)

    def reset_parameters(self, tau):
        init.constant_(self.thres, tau)

    def forward(self, input):

        norms = torch.sqrt(((input**2).sum(3))) + 1e-7

        input = input/norms.unsqueeze(3)

        # input[input.isnan()] = 0

        return F.relu(norms-self.thres[:, :, None])[:, :, :, None]*input


class VecRelu(torch.nn.Module):

    """
        The vectorial ReLu.
            x -> q+relu(-<q,k>/||k||)k/||k||

        Input:  (batch_size, n_in, m,d), (batch_size,n_in,d) or
            (batch_size,n_in,m,d)
        Output: (batch_size, n_in, m,d)

        Input parameters:
            None

        Attributes:
            None
    """
    def __init__(self):
        super(VecRelu, self).__init__()

    def forward(self, q, k):

        if k.dim() == 3:
            k = k.unsqueeze(2)
            k = k.expand(q.size())

        assert q.size()[0] == k.size()[0], \
            "k-vector and q-vector not of same batch size in VecRelu"
        assert q.size()[3] == k.size()[3], \
            "k-vector and q-vector not of same dimension  in VecRelu"

        sprods = (q*k).sum(2)

        normsquare = (k*k).sum(2)

        sprods[normsquare == 0] = 1
        normsquare[normsquare == 0] = 1

        q = q+F.relu(-sprods).unsqueeze(2)*k/normsquare.unsqueeze(2)

        return q

class DirectionalLoss(torch.nn.Module):

    """
       The directional loss function
            x,v -> min_{s=+-1} || x- sv||^2 + correction

        Input:  (batch_size,d), (batch_size,d)
        Output: (1)

        Input parameters:
            None

        Attributes:
            None
    """


    def __init__(self):
        super(DirectionalLoss, self).__init__()

    def forward(self, x, v):

        normcorrection= (.25-(x**2).sum(1))*torch.gt(.25-(x**2).sum(1),0)
        return (x**2 + v**2).sum(1).mean() - 2*torch.abs((v*x).sum(1)).mean(0) + 0*normcorrection.mean(0)


class WeightedSVD(torch.nn.Module):
    
    
    """
       Input:
           Z (points) (batch_size,nbr_points, d), 
           W (weights) (batch_size,nbr_points)
           
       Output: 
           The solution y(in (batch_size,d+1)) to the problem
           
               min sum_i w_i |<[z_i,1],y>|^2
           
           This vector defines the line through
           
                <[z,1],y> =0
    """
    
    def __init__(self):
        
        super(WeightedSVD,self).__init__()
        
    def forward(self,Z,W):
        
        Z= torch.cat((Z, torch.ones(Z.shape[0],Z.shape[1],1,device=Z.device)),2) 
        
        U,_,_= torch.svd(torch.transpose(W.unsqueeze(2)*Z,1,2)@Z)
        
        return U[:,:,-1]

class EFMatrify(torch.nn.Module):
    
    """
       Input: two vectors alpha,beta 
         Size: (batch_size,d) , (batch_size, d)
           
       Output: the corresponding 'protofundamental' matrix
             conj(alpha) beta + conj(alpha_perp)beta_perp
         Size: (batch_size,d,d)
        
    """
   
    def __init__(self):
        
        super(EFMatrify,self).__init__()
        
    def forward(self,alpha,beta):
        
        b,d = alpha.size()
    
    
        output = torch.zeros((b,d,d),device=alpha.device)
        
        alphan = alpha /(torch.sqrt((alpha**2).sum(1).unsqueeze(1))+1e-7)
        betan = beta /(torch.sqrt((beta**2).sum(1).unsqueeze(1))+1e-7)
        

        
        # conj(alpha)*beta
        output[:,0,0] = alpha[:,0]*beta[:,0] + alphan[:,1]*betan[:,1]
        output[:,0,1] = alpha[:,0]*beta[:,1] - alphan[:,1]*betan[:,0]
        output[:,1,0] = alpha[:,1]*beta[:,0] - alphan[:,0]*betan[:,1]
        output[:,1,1] = alpha[:,1]*beta[:,1] + alphan[:,0]*betan[:,0]
        
        # conj(alpha_perp) * beta_perp
        #output[:,0,0] = alphan[:,1]*betan[:,1] + alphan[:,0]*betan[:,0]
        #output[:,0,1] = alphan[:,0]*betan[:,1] - alphan[:,1]*betan[:,0]
        #output[:,1,0] = -alphan[:,0]*betan[:,1]  + alphan[:,1]*betan[:,0]
        #output[:,1,1] = alphan[:,1]*betan[:,1] + alphan[:,0]*alphan[:,0]
        
        
        return output
    
class epicentrify(torch.nn.Module):
    
    """
       Input: point cloud P and epipolar point v 
         Size: (batch_size,m,2) , (batch_size, 2)
           
       Output: a rotated and scaled version of P with the origin as an epipolar point
             
         Size: (batch_size,m,2)
        
    """
    
    def __init__(self):
        super(epicentrify,self).__init__()
    
    
    def forward(self,P,v):
        P = torch.cat((P,torch.ones((P.shape[0],P.shape[1],1))),2)

        theta = torch.asin(torch.sqrt((v[...,:2]**2).sum()))
        phi = torch.atan2(v[...,1],v[...,0])
    
        theta=-theta
        phi=-phi
    
        R = Rz(phi)
        S = Ry(theta)
    
        P = P @ R.transpose(1,2) 
        P = P @ S.transpose(1,2)
        G = F @ R.transpose(1,2) 
        G = G @ S.transpose(1,2)
    
        return P/P[...,2].unsqueeze(-1),G,v[:,:2]

def Rz(t):
    
    R= torch.zeros(t.shape[0],3,3)
    
    R[...,2,2]=1
    
    R[...,0,0] = torch.cos(t)
    R[...,0,1] = -torch.sin(t)
    R[...,1,0] = torch.sin(t)
    R[...,1,1] = torch.cos(t)
    
    return R

def Ry(t):
    R= torch.zeros(t.shape[0],3,3)
    
    R[...,1,1]=1
    
    R[...,0,0] = torch.cos(t)
    R[...,0,2] = torch.sin(t)
    R[...,2,0] = -torch.sin(t)
    R[...,2,2] = torch.cos(t)
    
    return R
    
    
    
