import torch
import utils


def sym_ep_loss(angles, matches):
    """
    Calculates (squared) symmetric epipolar distance loss
    based on virtual matches.
    Input: angles: (b, 5)
           matches: (b, num_pts, 4)
    Output: ys (b, num_pts)
    """
    batch_size, num_pts = matches.shape[0], matches.shape[1]
    x1 = torch.cat([matches[..., :2],
                    matches.new_ones(batch_size, num_pts, 1)],
                   dim=-1)  # (b, num_pts, 3)
    x2 = torch.cat([matches[..., 2:],
                    matches.new_ones(batch_size, num_pts, 1)],
                   dim=-1)  # (b, num_pts, 3)
    E = utils.normalize_E_batch(
        utils.essential_matrix_from_angles(angles))  # (b, 3, 3)

    Ex1 = torch.bmm(E, x1.transpose(dim0=1, dim1=2))  # (b, 3, num_pts)
    x2E = torch.bmm(x2, E)  # (b, num_pts, 3)
    x2Ex1 = torch.einsum("bni,bin->bn", x2, Ex1)  # (b, num_pts)
    ys = x2Ex1**2 * (
        1.0 / (Ex1[:, 0, :]**2 + Ex1[:, 1, :]**2 + 1e-15) +
        1.0 / (x2E[:, :, 0]**2 + x2E[:, :, 1]**2 + 1e-15)
    )
    loss = ys.mean()
    # print("SYM EP LOSS")
    # print(loss)
    return loss


if __name__ == "__main__":
    pass
