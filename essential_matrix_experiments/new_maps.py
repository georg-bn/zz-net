import torch
from nonlinearities import RadRelu

class CloudToWeight(torch.nn.Module):

    """
    Transforms one 2D point cloud to one vectors of weights
    in a permutation equivariant and rotation invariant way.
    The weights are to be used for "attentive contect normalization".
    Args:
        input_channels (default: 1),
        output_channels (default: 1),
        layer_structure: list of
            [(channels, features), ... ]
            describing the number of input channels
            and features of intermediate layers
            default: empty list, meaning only one layer in this Module.
            The first and last layer input/output
            are by default (input_channels, 5)
            and (output_channels, 1).
            The first layer operates on output of cloud_to_five,
            which is called by this Module.
    Inputs:
        cloud: (batch, input_channels, nbr_points, 2)
    Outputs:
        weight: (batch, output_channels, nbr_points)


    ### Minimal modification of Cloud*s*toWeight*s*
    """
    def __init__(self,
                 input_channels=1,
                 output_channels=1,
                 layer_structure=[]):
        super().__init__()
        self.layer_structure = \
            [(input_channels, 5)] \
            + layer_structure \
            + [(output_channels, 1)]
        self.layers = torch.nn.ModuleList()
        for j in range(len(self.layer_structure) - 1):
            self.layers.append(
                # in_channels, out_channels, in_dim, out_dim
                CloudToCloud(self.layer_structure[j][0], self.layer_structure[j + 1][0],
                             self.layer_structure[j][1], self.layer_structure[j + 1][1])
            )

    def forward(self, cloud):
        cloud= cloud_to_five(cloud)
        for j in range(len(self.layers) - 1):
            cloud = self.layers[j](cloud)
            cloud = torch.nn.functional.relu(cloud)

        # last layer:
        cloud = self.layers[-1](cloud)
        # squeeze the last dim, which is 1
        cloud= cloud.squeeze(dim=-1)
        cloud = torch.nn.functional.softmax(cloud,dim=2)

        return cloud


class CloudsToWeights(torch.nn.Module):
    """
    Transforms two 2D point clouds to two vectors of weights
    in a permutation equivariant and rotation invariant way.
    The weights are to be used for "attentive contect normalization".
    Args:
        input_channels (default: 1),
        output_channels (default: 1),
        layer_structure: list of
            [(channels, features), ... ]
            describing the number of input channels
            and features of intermediate layers
            default: empty list, meaning only one layer in this Module.
            The first and last layer input/output
            are by default (input_channels, 5)
            and (output_channels, 1).
            The first layer operates on output of cloud_to_five,
            which is called by this Module.
    Inputs:
        cloud_1, cloud_2: (batch, input_channels, nbr_points, 2)
    Outputs:
        weights_1, weights_2: (batch, output_channels, nbr_points)
    """
    def __init__(self,
                 input_channels=1,
                 output_channels=1,
                 layer_structure=[]):
        super().__init__()
        self.layer_structure = \
            [(input_channels, 5)] \
            + layer_structure \
            + [(output_channels, 1)]
        self.layers = torch.nn.ModuleList()
        for j in range(len(self.layer_structure) - 1):
            self.layers.append(
                # in_channels, out_channels, in_dim, out_dim
                TwoCloudsToTwoClouds(self.layer_structure[j][0],
                                     self.layer_structure[j + 1][0],
                                     self.layer_structure[j][1],
                                     self.layer_structure[j + 1][1])
            )

    def forward(self, cloud_1, cloud_2):
        cloud_1 = cloud_to_five(cloud_1)
        cloud_2 = cloud_to_five(cloud_2)
        for j in range(len(self.layers) - 1):
            cloud_1, cloud_2 = self.layers[j](cloud_1, cloud_2)
            cloud_1, cloud_2 = \
                torch.nn.functional.relu(cloud_1), \
                torch.nn.functional.relu(cloud_2)

        # last layer:
        cloud_1, cloud_2 = self.layers[-1](cloud_1, cloud_2)
        # squeeze the last dim, which is 1
        cloud_1, cloud_2 = cloud_1.squeeze(dim=-1), cloud_2.squeeze(dim=-1)
        cloud_1, cloud_2 = \
            torch.nn.functional.softmax(cloud_1, dim=2), \
            torch.nn.functional.softmax(cloud_2, dim=2)

        return cloud_1, cloud_2


class CloudToCloud(torch.nn.Module):
    """
    Takes one point cloud of rotation equiv. real features
    and returns a new one in a permutation equivariant way.

    Args: in_channels, out_channels, in_dim, out_dim
    Input:
        cloud (batch, in_channels, nbr_points, in_dim)

    Output:
        out_cloud (batch, out_channels, nbr_points, out_dim)


     ### Minimal modification of TwoCloudsToTwoClouds ###

    """
    def __init__(self, in_channels, out_channels, in_dim, out_dim):
        super().__init__()

        self.A_feat = torch.nn.Linear(in_dim, out_dim)
        self.A_chan = torch.nn.Linear(in_channels, out_channels, bias=False)

    def _apply_A(self, cloud):
        return self.A_chan(self.A_feat(cloud).transpose(1, 3)).transpose(1, 3)

    def forward(self, cloud):
        out_cloud = self._apply_A(cloud)
        return out_cloud


class TwoCloudsToTwoClouds(torch.nn.Module):
    """
    Takes two point clouds of rotation equiv. real features
    and returns two new ones in a permutation equivariant way.
    There is a further symmetry in that the changing the order of
    the two input clouds changes the order of the two output clouds.
    The features in output cloud 1 are Af_1 + Bf_2 where f_1 and f_2
    are the features of input clouds 1 and 2 respectively.
    Similarly the features in output cloud 2 are Bf_1 + Af_2.

    Args: in_channels, out_channels, in_dim, out_dim
    Input:
        cloud1 (batch, in_channels, nbr_points, in_dim)
        cloud2 (batch, in_channels, nbr_points, in_dim)
    Output:
        out_cloud1 (batch, out_channels, nbr_points, out_dim)
        out_cloud2 (batch, out_channels, nbr_points, out_dim)
    """
    def __init__(self, in_channels, out_channels, in_dim, out_dim):
        super().__init__()

        self.A_feat = torch.nn.Linear(in_dim, out_dim)
        self.A_chan = torch.nn.Linear(in_channels, out_channels, bias=False)

        self.B_feat = torch.nn.Linear(in_dim, out_dim)
        self.B_chan = torch.nn.Linear(in_channels, out_channels, bias=False)

    def _apply_A(self, cloud):
        return self.A_chan(self.A_feat(cloud).transpose(1, 3)).transpose(1, 3)

    def _apply_B(self, cloud):
        return self.B_chan(self.B_feat(cloud).transpose(1, 3)).transpose(1, 3)

    def forward(self, cloud1, cloud2):
        out_cloud1 = self._apply_A(cloud1) + self._apply_B(cloud2)
        out_cloud2 = self._apply_A(cloud2) + self._apply_B(cloud1)
        return out_cloud1, out_cloud2


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


class PNLayer(torch.nn.Module):
    """
        A permutation invariant linear layer.
        Input dim: (batch, in_channels, nbr_points)
        Output dim: (batch, out_channels, nbr_points)
    """
    def __init__(self,
                 nbr_in_chan,
                 nbr_out_chan):
        super().__init__()
        self.local_transf = torch.nn.Linear(nbr_in_chan, nbr_out_chan)
        self.global_transf = torch.nn.Linear(nbr_in_chan, nbr_out_chan,
                                             bias=False)

    def forward(self, cloud):
        tr_cloud = cloud.transpose(dim0=1, dim1=2)
        tr_loc = self.local_transf(tr_cloud)
        tr_glob = self.global_transf(tr_cloud.mean(dim=1, keepdims=True))
        out_cloud = (tr_loc + tr_glob).transpose(dim0=1, dim1=2)
        return out_cloud


class CNLayer(torch.nn.Module):
    """
        Input dim: (batch, in_channels, nbr_points)
        Output dim: (batch, out_channels, nbr_points)
    """
    def __init__(self,
                 nbr_in_chan,
                 nbr_out_chan,
                 weight_init_gain=1.0):
        super().__init__()

        self.weights = torch.nn.Parameter(
            torch.empty((nbr_in_chan, nbr_out_chan))
        )

        self.gain = weight_init_gain

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weights, gain=self.gain)

    def forward(self, cloud):
        cloud = torch.einsum("bin,io->bon", cloud, self.weights)
        cloud = context_normalize(cloud, True)
        return cloud


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


def cloud_to_five(cloud):
    """
    Takes 2D point clouds and returns 5D clouds of
    rotation invariant, permutation equivariant features.
    Doesn't change number of channels.
    Input:
        cloud (batch, channels, nbr_points, 2)
    Output:
        five_cloud (batch, channels, nbr_points, 5)
    """

    N = cloud.shape[2]  # nbr points
    ones = torch.ones(N, device=cloud.device)
    # sum of squared norms tensor 1
    F1 = torch.einsum('bcnd,m->bcm', cloud**2, ones)
    # square of sum tensor 1
    F2 = torch.sum(torch.einsum('bcnd,m->bcmd', cloud, ones)**2, axis=3)
    # squared norm
    F3 = torch.sum(cloud**2, axis=3)
    # real(z_i conj(sum(z_j)))
    F4 = torch.einsum('bcn,bcm->bcn', cloud[..., 0], cloud[..., 0]) \
        - torch.einsum('bcn,bcm->bcn', cloud[..., 1], -cloud[..., 1])
    # imag(z_i conj(sum(z_j)))
    F5 = torch.einsum('bcn,bcm->bcn', cloud[..., 0], -cloud[..., 1]) \
        + torch.einsum('bcn,bcm->bcn', cloud[..., 1], cloud[..., 0])

    # normalization of constant features
    F1 = F1 / N


def cloud_to_five(cloud):
    """
    Takes 2D point clouds and returns 5D clouds of
    rotation invariant, permutation equivariant features.
    Doesn't change number of channels.
    Input:
        cloud (batch, channels, nbr_points, 2)
    Output:
        five_cloud (batch, channels, nbr_points, 5)
    """

    N = cloud.shape[2]  # nbr points
    ones = torch.ones(N, device=cloud.device)
    # sum of squared norms tensor 1
    F1 = torch.einsum('bcnd,m->bcm', cloud**2, ones)
    # square of sum tensor 1
    F2 = torch.sum(torch.einsum('bcnd,m->bcmd', cloud, ones)**2, axis=3)
    # squared norm
    F3 = torch.sum(cloud**2, axis=3)
    # real(z_i conj(sum(z_j)))
    F4 = torch.einsum('bcn,bcm->bcn', cloud[..., 0], cloud[..., 0]) \
        - torch.einsum('bcn,bcm->bcn', cloud[..., 1], -cloud[..., 1])
    # imag(z_i conj(sum(z_j)))
    F5 = torch.einsum('bcn,bcm->bcn', cloud[..., 0], -cloud[..., 1]) \
        + torch.einsum('bcn,bcm->bcn', cloud[..., 1], cloud[..., 0])

    # normalization of constant features
    F1 = F1 / N
    F2 = F2 / N
    # normalization of non-constant features
    F3 = F3 - F3.mean(dim=2, keepdims=True)
    F4 = F4 - F4.mean(dim=2, keepdims=True)
    F5 = F5 - F5.mean(dim=2, keepdims=True)
    F3 = F3 / torch.max(F3.std(dim=2, keepdims=True),
                        1e-3 * torch.ones_like(F3))
    F4 = F4 / torch.max(F4.std(dim=2, keepdims=True),
                        1e-3 * torch.ones_like(F4))
    F5 = F5 / torch.max(F5.std(dim=2, keepdims=True),
                        1e-3 * torch.ones_like(F5))

    return torch.stack([F1, F2, F3, F4, F5], dim=3)


class REUnit(torch.nn.Module):

    """
        A RotationEquivariant unit operating point-wise on a point cloud.

        Parameters:
            layer_structure - the sizes of the layers
            weight_init_gain : for initialization of the weights
            tau: for initalizing the RadRelus

        Input: (batch_size, channels_0, cloud_size,2)
        Output: (batch_size, channels_last, cloud_size,2)
    """

    def __init__(self, layer_structure=(1,10,1), weight_init_gain=1, tau=.1):

        super(REUnit, self).__init__()

        self.weights = []

        for k in range(len(layer_structure)-1):
            K = torch.nn.Parameter(
                torch.empty(
                    (layer_structure[k],
                     layer_structure[k + 1],
                     2),
                    dtype=torch.float))
            self.weights.append(K)
            self.register_parameter('weight'+str(k),K)

        self.relus = torch.nn.ModuleList(
            RadRelu(layer_structure[k + 1], tau)
            for k in range(len(layer_structure) - 1))

        self.gain = weight_init_gain
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            torch.nn.init.xavier_normal_(weight, gain=self.gain)


    def forward(self,x):

        for k in range(len(self.weights)-1):
            x = cmmultiply(self.weights[k], x)
            x = self.relus[k](x)

        return cmmultiply(self.weights[-1], x)


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



class CNUnit(torch.nn.Module):
    """
    Takes a vector-valued feature and a vector of weights and returns the weighted
    context normalized vector valued feature.
    Does not change the number of channels.
    ! Assumes that the number of channels is equal in the vector and in the weights
    ! the weights should obey abs(weights).sum(2)=1
    Input:
        cloud (batch, channels, nbr_points, 2)
        weight (batch, channels, nbr_points)
    Output:
        cloud (batch, channels, nbr_points, 2)

    """

    def __init__(self):
        super(CNUnit,self).__init__()

    def forward(self,weight,cloud):

        assert weight.shape[1]==cloud.shape[1], \
            "Weight and feature in CNUnit must have same number of channels."

        weight = weight.unsqueeze(-1)

        # calculate and substract the weighted mean.
        mu = torch.sum(weight*cloud,2)
        cloud = cloud-mu.unsqueeze(2)

        # calculate and divide with standard deviation

        var = torch.sum((cloud**2) * weight, (2, 3), keepdims=True)
        std = torch.max(var, 1e-3 * torch.ones_like(var))

        cloud = cloud/torch.sqrt(var)

        return cloud




if __name__ == "__main__":
    pass
