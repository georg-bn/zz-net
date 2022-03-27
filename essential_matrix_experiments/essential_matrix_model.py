import ess_loss
import pytorch_lightning as pl
import new_maps
import nonlinearities
import torch


class EssMatrixTensorNet(pl.LightningModule):
    """
    A rotation equivariant permutation invariant net for estimating
    essential matrices.
    It is composed of several TwoCloudModules (ZZ-units).
    It predicts 2 rotation equivariant angles and 3 rotation invariant
    angles comprising a factorization of an essential matrix.

    Arguments:
               early_structs - list of early structures for the
                    internal TwoCloudModule:s.
               late_structs - list of late structures for the
                    internal TwoCloudModule:s.
               vector_structs - list of vector structures for the
                    internal TwoCloudModule:s.
               bias, complex_weights, full_weights - bools for
                    determining what type of alpha-units to use,
                    set to True unless for testing.
               skip_conn - bool for using skip connections
               lr - learning rate
               optimizer - adam or sgd
               tau - threshold for complex relus
    Input: correspondences (batch, 1, nbr_points, 4)
    Output: angle1 (batch)
            angle2 (batch)
            angle3 (batch)
            angle4 (batch)
            angle5 (batch)
    """

    def __init__(self,
                 early_structs=[[1, 8, 8], [3, 8, 8], [3, 8]],
                 late_structs=[[8, 8, 3], [8, 8, 3], [8, 8]],
                 vector_structs=[[1, 8, 3], [3, 8, 3], [3, 8]],
                 bias=True,
                 complex_weights=True,
                 full_weights=True,
                 skip_conn=True,
                 cn_type=-1,
                 lr=1e-2,
                 optimizer="adam",
                 tau=0.01):
        super().__init__()
        self.lr = lr
        self.optimizer = optimizer
        self.skip_conn = skip_conn

        if early_structs[0][0] != 1 or vector_structs[0][0] != 1:
            raise ValueError("Input channels must be 1")

        if len(early_structs) != len(late_structs) or \
           len(early_structs) != len(vector_structs):
            raise ValueError("The number of structures must be compatible.")

        if skip_conn:
            # Add channels for skip connections
            channels_into_angle = sum(e[0] for e in early_structs) + 8
            for j in range(len(early_structs)-1, 0, -1):
                early_structs[j][0] = sum(e[0] for e in early_structs[:j+1])
                vector_structs[j][0] = early_structs[j][0]
        else:
            channels_into_angle = late_structs[-1][-1]

        self.two_cloud_modules = torch.nn.ModuleList([
            TwoCloudModule(early_struct=early_structs[j],
                           late_struct=late_structs[j],
                           vector_struct=vector_structs[j],
                           bias=bias,
                           complex_weights=complex_weights,
                           full_weights=full_weights,
                           cn_type=cn_type,
                           tau=tau)
            for j in range(len(early_structs))
        ])

        self.equiv_angle_module = TwoCloudModule(
            early_struct=[channels_into_angle, 8],
            late_struct=[8, 1],
            vector_struct=[channels_into_angle, 8, 1],
            bias=bias,
            complex_weights=complex_weights,
            full_weights=full_weights,
            cn_type=cn_type,
            tau=tau)
        self.inv_angle_module = InvGlobalTwoCloudModule(
            # * 2 below because complex channels are bundled with all other
            [late_structs[-1][-1] * 2, 32, 64, 4]
        )

    def forward(self, correspondences):
        cloud1, cloud2 = \
            correspondences[..., :2], \
            correspondences[..., 2:]

        for j in range(len(self.two_cloud_modules)):
            n_cloud1, n_cloud2, weight1, weight2 = \
                self.two_cloud_modules[j](cloud1, cloud2)

            # From weights in last layer, calculate a sort of proxy
            # y in [0, 1],
            # for whether a point is an inlier or not.
            if j == len(self.two_cloud_modules) - 1:
                ys = weight1.norm(dim=3) + weight2.norm(dim=3)
                # Use tanh here since the norm is guaranteed to be >= 0
                ys = torch.tanh(ys)

            # combine the weights and clouds
            if self.skip_conn:
                cloud1 = torch.cat((new_maps.cmultiply(weight1,
                                                       n_cloud1),
                                   cloud1), dim=1)
                cloud2 = torch.cat((new_maps.cmultiply(weight2,
                                                       n_cloud2),
                                   cloud2), dim=1)
            else:
                cloud1 = new_maps.cmultiply(weight1,
                                            n_cloud1)
                cloud2 = new_maps.cmultiply(weight2,
                                            n_cloud2)

        e_angles1, e_angles2, e_weight1, e_weight2 = \
            self.equiv_angle_module(cloud1, cloud2)
        # contract the weights and clouds
        e_angles1 = new_maps.cmultiply(e_weight1, e_angles1).mean(2)
        e_angles2 = new_maps.cmultiply(e_weight2, e_angles2).mean(2)

        # reshape so that complex channels are bundled with others
        weight1 = weight1.transpose(dim0=2, dim1=3)
        weight2 = weight2.transpose(dim0=2, dim1=3)
        weight1 = weight1.reshape(weight1.shape[0], -1, weight1.shape[-1])
        weight2 = weight2.reshape(weight2.shape[0], -1, weight2.shape[-1])
        i_angles1, i_angles2 = self.inv_angle_module(weight1, weight2)

        # calculate the equivariant angles from
        # complex outputs
        # angles are ordered from left to right in the factorization
        # E = R1 R2 R3 diag(1, 1, 0) R4^t R5^t
        # I.e. the first ones correspond to input cloud2 and the
        # last ones to input cloud1. R3 has info from both clouds.
        angle1 = torch.atan2(e_angles2[:, 0, 1],
                             e_angles2[:, 0, 0])
        angle5 = torch.atan2(e_angles1[:, 0, 1],
                             e_angles1[:, 0, 0])
        # calculate the invariant angles from
        # real outputs
        angle2 = torch.atan2(i_angles2[:, 0], i_angles2[:, 1])
        angle3 = torch.atan2(i_angles2[:, 2], i_angles2[:, 3]) \
            - torch.atan2(i_angles1[:, 2], i_angles1[:, 3])
        angle4 = torch.atan2(i_angles1[:, 0], i_angles1[:, 1])

        return torch.stack(
            [angle1, angle2, angle3, angle4, angle5],
            dim=1
        ), ys

    # pytorch-lightning functions.
    def my_step(self, batch, batch_idx):
        angles, ys = self(batch["xs"])
        se_loss = ess_loss.sym_ep_loss(angles, batch["virtPts"])
        return se_loss

    def training_step(self, batch, batch_idx):
        loss = self.my_step(batch, batch_idx)
        self.log("train_loss", loss)
        # this prints to stdout:
        self.print(f"train_loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.my_step(batch, batch_idx)
        self.log("val_loss", loss)
        # this prints to stdout:
        self.print(f"val_loss: {loss}")
        return loss

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError("Wrong optimizer")
        return optimizer


class InvGlobalTwoCloudModule(torch.nn.Module):
    """
    A permutation equivariant module for point clouds.
    It is composed of several XYZ
    and XYZ
    to generate invariant global quantities.
    Applies relu between intermediate layers,
    sums to global feature at the end.

    Arguments:
            layer_struct: tuple/list of nbr of channels in each layer
    Input:
            cloud1 (batch, nbr_chan_in, nbr_points)
            cloud2 (batch, nbr_chan_in, nbr_points)
    Output:
            cloud1 (batch, nbr_chan_out, nbr_points)
            cloud2 (batch, nbr_chan_out, nbr_points)
    """
    def __init__(self,
                 layer_struct=[1, 1]):
        super().__init__()

        self.layers_A = torch.nn.ModuleList([
            new_maps.PNLayer(layer_struct[j], layer_struct[j + 1])
            for j in range(len(layer_struct) - 1)
        ])

        self.layers_B = torch.nn.ModuleList([
            new_maps.PNLayer(layer_struct[j], layer_struct[j + 1])
            for j in range(len(layer_struct) - 1)
        ])

    def forward(self, cloud1, cloud2):

        for k in range(len(self.layers_A) - 1):
            tmp_cloud1 = self.layers_A[k](cloud1) + \
                self.layers_B[k](cloud2)
            cloud2 = self.layers_A[k](cloud2) + \
                self.layers_B[k](cloud1)
            cloud1 = torch.nn.functional.relu(tmp_cloud1)
            cloud2 = torch.nn.functional.relu(cloud2)

        tmp_cloud1 = self.layers_A[-1](cloud1) + \
            self.layers_B[-1](cloud2)
        cloud2 = self.layers_A[-1](cloud2) + \
            self.layers_B[-1](cloud1)

        # contract to single invariant number per channel
        cloud1 = tmp_cloud1.mean(2)
        cloud2 = cloud2.mean(2)

        return cloud1, cloud2


class TwoCloudModule(torch.nn.Module):
    """
    A rotation equivariant permutation equivariant module for estimating
    inliers/outliers.
    It is composed of several XYZ
    and XYZ
    to generate weights for all correspondences.

    Arguments:
        early_struct: list/tuple of channel sizes in the rot inv tensor layers
        late_struct: list/tuple of channel sizes in the rot inv vector layers
        vector_struct:
            list/tuple of channel sizes in the rot equiv vector layers
    Input:
            cloud1 (batch, nbr_chan_in, nbr_points, 2)
            cloud2 (batch, nbr_chan_in, nbr_points, 2)
    Output:
            cloud1 (batch, nbr_chan_out, nbr_points, 2)
            cloud2 (batch, nbr_chan_out, nbr_points, 2)
            weight1 (batch, nbr_chan_out, nbr_points, 2)
            weight2 (batch, nbr_chan_out, nbr_points, 2)
    """
    def __init__(self,
                 early_struct=(1, 8),
                 late_struct=(8, 3),
                 vector_struct=(1, 8, 3),
                 bias=True,
                 complex_weights=True,
                 full_weights=True,
                 cn_type=-1,
                 tau=0.01):
        super().__init__()

        self.cn_type = cn_type

        if early_struct[0] != vector_struct[0]:
            raise ValueError("Number of input channels must be consistent.")

        if early_struct[-1] != late_struct[0]:
            raise ValueError(
                "early_struct must end with as many channels " +
                "as late_struct starts.")

        if late_struct[-1] != vector_struct[-1]:
            raise ValueError("Number of output channels must be consistent.")

        self.early_layers_A = torch.nn.ModuleList([
            new_maps.EquivTensorLayer(early_struct[j], early_struct[j + 1],
                                      full_weights=full_weights,
                                      bias=bias)
            for j in range(len(early_struct) - 1)
        ])

        self.late_layers_A = torch.nn.ModuleList([
            new_maps.EquivVectorLayer(late_struct[j], late_struct[j + 1],
                                      complex_weights=complex_weights,
                                      bias=bias)
            for j in range(len(late_struct) - 1)
        ])

        self.early_layers_B = torch.nn.ModuleList([
            new_maps.EquivTensorLayer(early_struct[j], early_struct[j + 1],
                                      full_weights=full_weights,
                                      bias=bias)
            for j in range(len(early_struct) - 1)
        ])

        self.late_layers_B = torch.nn.ModuleList([
            new_maps.EquivVectorLayer(late_struct[j], late_struct[j + 1],
                                      complex_weights=complex_weights,
                                      bias=bias)
            for j in range(len(late_struct) - 1)
        ])

        self.vector_layers = torch.nn.ModuleList([
            new_maps.EquivVectorLayer(vector_struct[j], vector_struct[j + 1],
                                      complex_weights=complex_weights,
                                      bias=False)
            for j in range(len(vector_struct) - 1)
        ])

        self.relus = torch.nn.ModuleList([
            nonlinearities.RadRelu(vector_struct[j], tau)
            for j in range(1, len(vector_struct))
        ])

    def forward(self, cloud1, cloud2):

        weight1 = new_maps.tensorify_z(cloud1)
        weight2 = new_maps.tensorify_z(cloud2)

        for k in range(len(self.early_layers_A) - 1):
            temp_weight1 = self.early_layers_A[k](weight1) + \
                self.early_layers_B[k](weight2)
            weight2 = self.early_layers_A[k](weight2) + \
                self.early_layers_B[k](weight1)
            weight1 = torch.nn.functional.relu(temp_weight1)
            weight2 = torch.nn.functional.relu(weight2)

        temp_weight1 = self.early_layers_A[-1](weight1) + \
            self.early_layers_B[-1](weight2)
        weight2 = self.early_layers_A[-1](weight2) + \
            self.early_layers_B[-1](weight1)
        weight1 = temp_weight1

        # tensors to vectors
        weight1 = weight1.mean(3)
        weight2 = weight2.mean(3)

        if self.cn_type == -1 or self.cn_type == 1:
            weight1 = new_maps.context_normalize(weight1)
            weight2 = new_maps.context_normalize(weight2)

        # apply the late ('vector') layers of the weight unit
        for k in range(len(self.late_layers_A) - 1):
            temp_weight1 = self.late_layers_A[k](weight1) + \
                self.late_layers_B[k](weight2)
            weight2 = self.late_layers_A[k](weight2) + \
                self.late_layers_B[k](weight1)
            weight1 = torch.nn.functional.relu(temp_weight1)
            weight2 = torch.nn.functional.relu(weight2)

        temp_weight1 = self.late_layers_A[-1](weight1) + \
            self.late_layers_B[-1](weight2)
        weight2 = self.late_layers_A[-1](weight2) + \
            self.late_layers_B[-1](weight1)
        weight1 = temp_weight1

        # apply the vector unit
        for k in range(len(self.vector_layers)-1):
            cloud1 = self.vector_layers[k](cloud1)
            cloud2 = self.vector_layers[k](cloud2)
            cloud1 = self.relus[k](cloud1)
            cloud2 = self.relus[k](cloud2)

        cloud1 = self.vector_layers[-1](cloud1)
        cloud2 = self.vector_layers[-1](cloud2)

        if self.cn_type == -1 or self.cn_type == 2:
            cloud1 = new_maps.context_normalize(cloud1)
            cloud2 = new_maps.context_normalize(cloud2)

        return cloud1, cloud2, weight1, weight2


if __name__ == "__main__":
    pass
