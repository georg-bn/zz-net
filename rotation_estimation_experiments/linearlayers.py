# -*- coding: utf-8 -*-

import torch


class EquivTensorLayer(torch.nn.Module):
    """
        Takes 2d pointcloud
        Input dim: (batch, in_channels, nbr_points, nbr_points, 2)
        Output dim: (batch, out_channels, nbr_points, nbr_points, 2)

        If full_weights is True, each channel is a real-linear map C->C
        Else If complex_weights is True, each channel is a complex-linear map C->C
        Else each map is of the form z->lambda*z with lambda in R
    """
    def __init__(self,
                 nbr_in_chan,
                 nbr_out_chan,
                 complex_weights=False,
                 full_weights =False,
                 bias=False,
                 weight_init_gain=1.0):
        super().__init__()
        self.nbr_in_chan = nbr_in_chan
        self.nbr_out_chan = nbr_out_chan

        self.mode = 'default'

        if full_weights:
            self.mode = 'full'
        elif complex_weights:
            self.mode = 'complex'

        if self.mode == 'full':
           self.weights = torch.nn.Parameter(
                torch.empty((nbr_in_chan, nbr_out_chan, 2, 2, 15))
                )
        elif self.mode =='complex':
            self.weights = torch.nn.Parameter(
                torch.empty((nbr_in_chan, nbr_out_chan, 2, 15))
            )
        else:
            self.weights = torch.nn.Parameter(
                torch.empty((nbr_in_chan, nbr_out_chan, 15))
            )

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(nbr_out_chan,2,2))
        self.bias_active = bias

        self.gain = weight_init_gain
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weights, gain=self.gain)
        if self.bias_active:
            torch.nn.init.xavier_normal_(self.bias, gain= self.gain/10)

    def forward(self, in_tensor):
        diag = in_tensor.diagonal(dim1=2, dim2=3).transpose(2, 3)
        col_sum = in_tensor.sum(dim=2)
        row_sum = in_tensor.sum(dim=3)
        # class 3
        out_tensor = self.apply_class3_maps(in_tensor, self.weights[...,13:15])

        # class 1
        nbr_points = in_tensor.shape[2]

        out_tensor += self.apply_class1_maps(diag.mean(dim=2),
                                            self.weights[...,0:2],nbr_points)
        out_tensor += self.apply_class1_maps(in_tensor.sum(dim=(2,3)),
                                            self.weights[...,0:2],nbr_points)


        # class 2

        out_tensor += self.apply_class2_maps(diag, self.weights[...,4:7])
        out_tensor += self.apply_class2_maps(col_sum, self.weights[...,7:10])
        out_tensor += self.apply_class2_maps(row_sum, self.weights[...,10:13])


        # add bias
        if self.bias_active:
            out_tensor += self.bias[...,0][None,:,None,None,:]
            out_tensor += torch.diag_embed(self.bias[...,1].unsqueeze(-1).expand(-1,-1,nbr_points)
                                           ,dim1=1, dim2=2)[None,...]


        return out_tensor

    def apply_class3_maps(self,v,A):
        # applies two weight-chunks to a vector v and returns
        # the sum. Different behaviour depending on self.mode!
        # A.shape (channel_in_channel_out, ..., 2), where
        #       ... = (2,2) if self.mode == 'full'
        #       ... = (2) if self.mode == 'complex'
        #       ... = _ else
        # v.shape (batch,channel_in,cloud_size,2)
        if self.mode == 'full':
            out_tensor = torch.einsum("cokl,bcijl->boijk",
                                  A[...,0], v) + \
                           torch.einsum("cokl,bcijl->bojik",
                             A[...,0], v )
        elif self.mode == 'complex':
            # imaginary part of weights
            out_tensor = torch.einsum("co,bcijk->boijk",
                                  A[...,1,0],
                                  v) + \
                          torch.einsum("co,bcijk->bojik",
                             self.weights[..., 1, 1],
                             v)
            out_tensor = out_tensor.roll(1,-1)
            out_tensor[...,0] = - out_tensor[...,0]

            # real part of weights
            out_tensor += torch.einsum("co,bcijk->boijk",
                                  A[...,0,0], v) + \
                          torch.einsum("co,bcijk->bojik",
                             self.weights[..., 0, 1], v)


        else:
             out_tensor = torch.einsum("co,bcijk->boijk",
                                  A[...,0], v) + \
                           torch.einsum("co,bcijk->bojik",
                             A[...,0], v )
        return out_tensor


    def apply_class2_maps(self,v,A):
        # applies three weight-chunks to a vector v and returns
        # the sum. Behaviour depends on self.complex_weights
        # A.shape (channel_in,channel_out,,..., 3) , where
        #       ... = (2,2) if self.mode == 'full'
        #       ... = (2) if self.mode == 'complex'
        #       ... = _ else
        # v.shape (batch,channel_in,cloud_size,2)

        if self.mode == 'full':
            u = torch.einsum("coij,bcmj->boim",A[..., 0],v)
            out = torch.diag_embed(u, dim1=2, dim2=3)
            u= torch.einsum("coij,bcmj->bomi",A[...,1],v)
            out += u[:, :, None, :, :]
            u = torch.einsum("coij,bcmj->bomi",A[..., 2],v)
            out += u[:, :, :, None, :]

        elif self.mode == 'complex':

            # imaginary parts

            u = torch.einsum("co,bcmk->bokm",A[..., 1,0],v)
            out = torch.diag_embed(u, dim1=2, dim2=3)
            u= torch.einsum("co,bcmk->bomk",A[...,1,1],v)
            out += u[:, :, None, :, :]
            u = torch.einsum("co,bcmk->bomk",A[...,1,2],v)
            out += u[:, :, :, None, :]

            out = out.roll(1,-1)
            out[...,0] = - out[...,0]

            # real parts

            u = torch.einsum("co,bcmk->bokm",A[..., 0,0],v)
            out += torch.diag_embed(u, dim1=2, dim2=3)
            u= torch.einsum("co,bcmk->bomk",A[...,0,1],v)
            out += u[:, :, None, :, :]
            u = torch.einsum("co,bcmk->bomk",A[...,0,2],v)
            out += u[:, :, :, None, :]


        else:
            u = torch.einsum("co,bcmk->bokm",A[..., 0],v)
            out = torch.diag_embed(u, dim1=2, dim2=3)
            u= torch.einsum("co,bcmk->bomk",A[...,1],v)
            out += u[:, :, None, :, :]
            u = torch.einsum("co,bcmk->bomk",A[..., 2],v)
            out += u[:, :, :, None, :]

        return out


    def apply_class1_maps(self,v,A,nbr_points):
        # applies two weight-chunks to a vector v and returns
        # the sum. Behaviour depends on self.complex_weights
        # Needs to know nbr_points in the original matrix to embed diagonal
        # properly
        # A.shape (channel_in,channel_out,...,3), where
        #       ... = (2,2) if self.mode == 'full'
        #       ... = (2) if self.mode == 'complex'
        #       ... = _ else
        # v.shape (batch,channel_in,2)



        if self.mode == 'full':
            out = torch.diag_embed(torch.einsum("coij,bcj->boj",
                                   A[..., 1],
                                   v)[..., None].expand(-1,-1,-1,nbr_points),
                                    dim1=2, dim2=3)
            out += torch.einsum("coij,bcj->boj",
                                   A[..., 0],v)[:, :, None, None, :]

        elif self.mode == 'complex':

            # imaginary parts

            out = torch.diag_embed(torch.einsum("co,bck->bok",
                                   A[...,1, 1],
                                   v)[..., None].expand(-1,-1,-1,nbr_points),
                                    dim1=2, dim2=3)
            out += torch.einsum("co,bck->bok",
                                   A[..., 1,0],v)[:, :, None, None, :]

            out = out.roll(1,-1)
            out[...,0] = - out[...,0]

            # real parts

            out += torch.einsum("co,bck->bok",
                                   A[..., 0,0],v)[:, :, None, None, :]

            out += torch.diag_embed(torch.einsum("co,bck->bok",
                                   A[..., 0,1],
                                   v)[..., None].expand(-1,-1,-1,nbr_points),
                                    dim1=2, dim2=3)


        else:
            out = torch.diag_embed(torch.einsum("co,bck->bok",
                                   A[..., 1],
                                   v)[..., None].expand(-1,-1,-1,nbr_points),
                                    dim1=2, dim2=3)
            out += torch.einsum("co,bck->bok",
                                   A[..., 0],v)[:, :, None, None, :]



        return out


def tensorify_z(z):
    # calculates zz* from z
    # in: (batch_size,channels,m,2)
    # out : (batch_size,channels,m,m,2)

    batch_size, channels, m, _ = z.size()

    out = torch.empty((batch_size, channels, m, m, 2), device=z.device)

    out[..., 0] = z[:, :, :, None, 0] * z[:, :, None, :, 0] + \
        z[:, :, :, None, 1] * z[:, :, None, :, 1]
    out[..., 1] = -z[:, :, :, None, 0] * z[:, :, None, :, 1] + \
        z[:, :, :, None, 1] * z[:, :, None, :, 0]

    return out


def context_normalize(x, normalize_spatial=False):
    # Normalize within each point cloud, i.e.
    # not over batches and channels
    # x (batch_size, channels, ..., spatial)

    if normalize_spatial:
        cloud_dims = tuple(range(2, x.dim()))
    else:
        cloud_dims = tuple(range(2, x.dim()-1))

    x = x - x.mean(dim=cloud_dims, keepdims=True)
    if normalize_spatial:
        x = x / torch.max(x.std(dim=cloud_dims, keepdims=True),
                          1.0e-4*torch.ones(x.size(), device=x.device))
    else:
        x = x / torch.max(x.std(dim=(*cloud_dims, -1), keepdims=True),
                          1.0e-4*torch.ones(x.size(), device=x.device))

    return x


class EquivVectorLayer(torch.nn.Module):
    """
        Input dim: (batch, in_channels, nbr_points, 2)
        Output dim: (batch, out_channels, nbr_points, 2)

        If full_weights is True, each channel is a real-linear map C->C
        Else If complex_weights is True, each channel is a complex-linear map C->C
        Else each map is of the form z->lambda*z with lambda in R
    """
    def __init__(self,
                 nbr_in_chan,
                 nbr_out_chan,
                 complex_weights = False,
                 full_weights = False,
                 bias = False,
                 weight_init_gain=1.0):
        super().__init__()
        self.nbr_in_chan = nbr_in_chan
        self.nbr_out_chan = nbr_out_chan

        self.mode ='default'

        if full_weights:
            self.mode = 'full'
        elif complex_weights:
            self.mode = 'complex'

        if self.mode == 'full':
            self.weights = torch.nn.Parameter(
                torch.empty((nbr_in_chan, nbr_out_chan,2,2,2)))
        elif self.mode == 'complex':
            self.weights = torch.nn.Parameter(
                torch.empty((nbr_in_chan, nbr_out_chan, 2,2))
                )
        else:
            self.weights = torch.nn.Parameter(
                torch.empty((nbr_in_chan, nbr_out_chan, 2))
                )

        self.bias_active = bias

        if self.bias_active:
            self.bias = torch.nn.Parameter(
                torch.empty((nbr_out_chan,2)))

        self.gain = weight_init_gain
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weights, gain=self.gain)
        if self.bias_active:
            torch.nn.init.xavier_normal_(self.bias, gain=self.gain/10)

    def forward(self,in_vector):

        im=torch.zeros((1,1,1,2),device=in_vector.device)
        im[...,1]=1

        if self.mode =='full':
            out_vector = torch.einsum('coij,bcmj->bomi',
                                  self.weights[...,0],in_vector)
            out_vector += torch.einsum('coij,bcj->boi',
                                  self.weights[...,1],in_vector.mean(2)).unsqueeze(2)
        elif self.mode == 'complex':
            #imaginary parts
            out_vector = torch.einsum('co,bcik->boik',
                                 self.weights[...,1,0],in_vector)
            out_vector += torch.einsum('co,bck->bok',
                                  self.weights[...,1,1],in_vector.mean(2)).unsqueeze(2)

            out_vector = out_vector.roll(1,-1)
            out_vector[...,0] = - out_vector[...,0]
            #out_vector=cmultiply(im,out_vector)

            #real parts
            out_vector += torch.einsum('co,bcik->boik',
                                  self.weights[...,0,0],in_vector)
            out_vector += torch.einsum('co,bck->bok',
                                  self.weights[...,0,1],in_vector.mean(2)).unsqueeze(2)


        else:
            out_vector = torch.einsum('co,bcik->boik',
                                  self.weights[...,0],in_vector)
            out_vector += torch.einsum('co,bck->bok',
                                  self.weights[...,1],in_vector.mean(2)).unsqueeze(2)

        # add bias
        if self.bias_active:
            out_vector += self.bias[None,:,None,:]

        return out_vector


def cmmultiply(A,x):
    # multiply weight matrix of size n_in,n_out,2
    # with cloud of size batch_size, n_in,m,2



    output = torch.zeros((x.size()[0], A.size()[1], x.size()[2], 2),
                         dtype=x.dtype, device=x.device)

    output[...,0] = torch.einsum('ik,bim->bkm',A[...,0],x[...,0]) \
            - torch.einsum('ik,bim->bkm',A[...,1],x[...,1])
    output[...,1] = torch.einsum('ik,bim->bkm',A[...,1],x[...,0]) \
            + torch.einsum('ik,bim->bkm',A[...,0],x[...,1])

    return output


def cmultiply(a,b,conj=False):
        # calculates a*b. If conj is true, a*conj(b) is calculated
        # in : (batch_size,channels,m,2), (batch_size,channels,m,2)
        # out : (bathc_size,channels,m,2)

        if conj:
            sigma=-1
        else:
            sigma=1

        out = torch.empty_like(b)+torch.empty_like(a)

        out[...,0] = a[...,0]*b[...,0]-sigma*a[...,1]*b[...,1]
        out[...,1] = sigma*a[...,0]*b[...,1] + a[...,1]*b[...,0]

        return out

if __name__ == "__main__":
    pass
