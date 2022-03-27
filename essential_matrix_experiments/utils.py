# Partially based on OANet
# https://github.com/zjhthu/OANet
import numpy as np
import torch


def dict_to_cuda(data):
    for key in data:
        if type(data[key]) == torch.Tensor:
            data[key] = data[key].cuda()
    return data


def twoD_rot(angle, device="cpu"):
    """
    Creates a 2D rotation matrix of angle angle.
    """
    return torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                         [torch.sin(angle), torch.cos(angle)]],
                        device=device)


def twoD_rot_batch(angle_batch):
    """
    Creates a batch of 2D rotations from batch of angles.
    Input: (batch)
    Output: (batch, 2, 2)
    """
    return torch.stack([
        torch.stack([torch.cos(angle_batch), torch.sin(angle_batch)],
                    dim=1),
        torch.stack([-torch.sin(angle_batch), torch.cos(angle_batch)],
                    dim=1)
    ], dim=2)


def create_Ry(a, use_torch=False):
    """
    Creates a rotation matrix about y of angle a
    """
    if use_torch:
        return torch.tensor([[torch.cos(a), 0.0, torch.sin(a)],
                             [0.0, 1.0, 0.0],
                             [-torch.sin(a), 0.0, torch.cos(a)]])
    else:
        return np.array([[np.cos(a), 0.0, np.sin(a)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(a), 0.0, np.cos(a)]])


def create_Ry_batch(a):
    """
    Creates a batch of rotation matrices about y of angles a.
    Input (batch)
    Output (batch, 3, 3)
    """
    return torch.stack([
        torch.stack([torch.cos(a),
                     torch.zeros_like(a),
                     -torch.sin(a)],
                    dim=1),
        torch.stack([torch.zeros_like(a),
                     torch.ones_like(a),
                     torch.zeros_like(a)],
                    dim=1),
        torch.stack([torch.sin(a),
                     torch.zeros_like(a),
                     torch.cos(a)],
                    dim=1)
    ], dim=2)


def create_Rz(a, use_torch=False):
    """
    Creates a rotation matrix about z of angle a
    """
    if use_torch:
        return torch.tensor([[torch.cos(a), -torch.sin(a), 0.0],
                             [torch.sin(a), torch.cos(a), 0.0],
                             [0.0, 0.0, 1.0]])
    return np.array([[np.cos(a), -np.sin(a), 0.0],
                     [np.sin(a), np.cos(a), 0.0],
                     [0.0, 0.0, 1.0]])


def create_Rz_batch(a):
    """
    Creates a batch of rotation matrices about z of angles a.
    Input (batch)
    Output (batch, 3, 3)
    """
    return torch.stack([
        torch.stack([torch.cos(a),
                     torch.sin(a),
                     torch.zeros_like(a)],
                    dim=1),
        torch.stack([-torch.sin(a),
                     torch.cos(a),
                     torch.zeros_like(a)],
                    dim=1),
        torch.stack([torch.zeros_like(a),
                     torch.zeros_like(a),
                     torch.ones_like(a)],
                    dim=1)
    ], dim=2)


def factor_a_euler_angles(R, use_torch=False):
    """
    Factors a rotation matrix R into Euler angles a, b, c
    corresponding to rotations Rz Ry Rz'.
    https://en.wikipedia.org/wiki/Euler_angles
    """
    if use_torch:
        a = torch.atan2(R[1, 2], R[0, 2])
        b = torch.atan2(torch.sqrt(1 - R[2, 2]**2), R[2, 2])
        c = torch.atan2(R[2, 1], -R[2, 0])
    else:
        a = np.arctan2(R[1, 2], R[0, 2])
        b = np.arctan2(np.sqrt(1 - R[2, 2]**2), R[2, 2])
        c = np.arctan2(R[2, 1], -R[2, 0])
    return a, b, c


def factor_R_euler_angles(R, use_torch=False):
    """
    Factors a rotation matrix R into Euler rotations Rz Ry Rz'
    https://en.wikipedia.org/wiki/Euler_angles
    """
    a, b, c = factor_a_euler_angles(R, use_torch)
    return create_Rz(a, use_torch), \
        create_Ry(b, use_torch), \
        create_Rz(c, use_torch)


def angle_factorization_essential_matrix(R, t, use_torch=False):
    """
    Factors an essential matrix [t]_xR[t] into rotations as
    E = Rz_2(a) Ry_2(b) Rz(c) diag(1,1,0) Ry_1(d) Rz_1(e)
    Returns the angles of the rotations:
    a, b, c, d, e
    """
    E = essential_matrix_from_R_t(R, t, use_torch)
    # E = u @ np.diag(s) @ vh = (u * s) @ vh
    if use_torch:
        u, s, v = torch.svd(E)
        vh = v.t()
        assert torch.allclose(s, torch.tensor([1.0, 1.0, 0.0],
                                              device=s.device),
                              rtol=1e-4,
                              atol=1e-6), \
            f"s does not have expected form, s: {s}"
        # make sure u and vh are rotations:
        if not torch.isclose(torch.linalg.det(u),
                             torch.tensor([1.0],
                                          device=u.device)):
            u = -u
        if not torch.isclose(torch.linalg.det(vh),
                             torch.tensor([1.0],
                                          device=vh.device)):
            vh = -vh
    else:
        u, s, vh = np.linalg.svd(E)
        assert np.allclose(s, np.array([1.0, 1.0, 0.0])), \
            f"s does not have expected form, s: {s}"
        # make sure u and vh are rotations:
        if not np.isclose(np.linalg.det(u), 1.0):
            u = -u
        if not np.isclose(np.linalg.det(vh), 1.0):
            vh = -vh
    a, b, c1 = factor_a_euler_angles(u, use_torch)
    c2, d, e = factor_a_euler_angles(vh, use_torch)
    return a, b, c1 + c2, d, e


def rotation_factorization_essential_matrix(R, t, use_torch=False):
    """
    Factors an essential matrix [t]_xR[t] into rotations as
    E = Rz_2 Ry_2 Rz diag(1,1,0) Ry_1 Rz_1
    Returns the rotation matrices
    Rz_2, Ry_2, Rz, Ry_1, Rz_1
    """
    a, b, c, d, e = angle_factorization_essential_matrix(R, t, use_torch)
    return create_Rz(a), create_Ry(b), create_Rz(c), \
        create_Ry(d), create_Rz(e)


def create_S_batch(batch_size, device):
    """
    Creates a batch of diag([1,1,0]) matrices
    Input: int
    Output (batch, 3, 3)
    """
    return torch.stack([
        torch.stack([torch.ones(batch_size, device=device),
                     torch.zeros(batch_size, device=device),
                     torch.zeros(batch_size, device=device)],
                    dim=1),
        torch.stack([torch.zeros(batch_size, device=device),
                     torch.ones(batch_size, device=device),
                     torch.zeros(batch_size, device=device)],
                    dim=1),
        torch.stack([torch.zeros(batch_size, device=device),
                     torch.zeros(batch_size, device=device),
                     torch.zeros(batch_size, device=device)],
                    dim=1)
    ], dim=2)


def essential_matrix_from_angles(angles,
                                 use_torch=True,
                                 flip_sign=True,
                                 batch=True):
    """
    Creates essential matrix.
    if flip_sign:
    E = Rz_2 Ry_2 Rz diag(1,1,0) Ry_1^t Rz_1^t
    NOTE: transposes to the right above!
    if not flip_sign:
    E = Rz_2 Ry_2 Rz diag(1,1,0) Ry_1 Rz_1
    Input: (b, 5)
    Output: (b, 3, 3)
    """
    if use_torch:
        if batch:
            E = create_Rz_batch(angles[:, 0])
            E = E.bmm(create_Ry_batch(angles[:, 1]))
            E = E.bmm(create_Rz_batch(angles[:, 2]))
            E = E.bmm(create_S_batch(angles.shape[0], angles.device))
            if flip_sign:
                E = E.bmm(create_Ry_batch(-angles[:, 3]))
                E = E.bmm(create_Rz_batch(-angles[:, 4]))
            else:
                E = E.bmm(create_Ry_batch(angles[:, 3]))
                E = E.bmm(create_Rz_batch(angles[:, 4]))
        else:
            E = create_Rz(angles[0], use_torch=True)
            E = E @ create_Ry(angles[1], use_torch=True)
            E = E @ create_Rz(angles[2], use_torch=True)
            E = E @ torch.diag(torch.tensor([1.0, 1.0, 0.0]))
            E = E @ create_Ry(angles[3], use_torch=True)
            E = E @ create_Rz(angles[4], use_torch=True)
    else:
        if batch:
            raise NotImplementedError()
        else:
            if flip_sign:
                E = create_Rz(angles[0])
                E = E @ create_Ry(angles[1])
                E = E @ create_Rz(angles[2])
                E = E @ np.diag([1.0, 1.0, 0.0])
                E = E @ create_Ry(-angles[3])
                E = E @ create_Rz(-angles[4])
            else:
                E = create_Rz(angles[0])
                E = E @ create_Ry(angles[1])
                E = E @ create_Rz(angles[2])
                E = E @ np.diag([1.0, 1.0, 0.0])
                E = E @ create_Ry(angles[3])
                E = E @ create_Rz(angles[4])
    return E


def essential_matrix_from_R_t(R, t, use_torch=False):
    """
    Creates an essential matrix [t]_x R
    """
    if use_torch:
        tx = torch.cross(torch.cat([t, t, t], dim=1),
                         torch.eye(3, device=t.device))
        E = tx @ R
    else:
        tx = np.cross(t, np.identity(3))
        E = tx @ R

    return E


def normalize_E_batch(Es, use_torch=True):
    if use_torch:
        return Es / \
            torch.norm(
                torch.reshape(Es, (-1, 9)),
                dim=1,
                keepdim=True)[:, :, None]
    else:
        return Es / \
            np.linalg.norm(
                np.reshape(Es, (-1, 9)),
                axis=1,
                keepdims=True)[:, :, None]


def torch_skew_symmetric(t):
    zero = torch.zeros_like(t[:, 0])

    M = torch.stack([
        zero, -t[:, 2], t[:, 1],
        t[:, 2], zero, -t[:, 0],
        -t[:, 1], t[:, 0], zero,
    ], dim=1)

    return M


if __name__ == "__main__":
    pass
