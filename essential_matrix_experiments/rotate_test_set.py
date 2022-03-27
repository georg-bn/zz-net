import numpy as np
from distutils.dir_util import copy_tree
from pathlib import Path
import pickle


def create_rotated_data(data_folder_in, data_folder_out, angle_max):
    R_file = data_folder_in / f"Rs_{part}.pkl"
    x_file = data_folder_in / f"xs_{part}.pkl"

    R_file_out = data_folder_out / f"Rs_{part}.pkl"
    x_file_out = data_folder_out / f"xs_{part}.pkl"

    # load data
    Rs = pickle.load(open(R_file, "rb"))
    xs = pickle.load(open(x_file, "rb"))

    # rotate x_1 and R
    for j in range(len(Rs)):
        rand_angle = np.random.rand() * np.pi * 2. * angle_max / 180. \
                - np.pi * angle_max / 180.
        R_z = np.array(
            [[np.cos(rand_angle), -np.sin(rand_angle), 0.0],
             [np.sin(rand_angle), np.cos(rand_angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        Rs[j] = Rs[j] @ R_z.T
        xs[j][..., :2] = xs[j][..., :2] @ R_z[:2, :2].T

    # save data
    pickle.dump(Rs, open(R_file_out, "wb"))
    pickle.dump(xs, open(x_file_out, "wb"))


if __name__ == "__main__":
    part = "te"  # te
    nbr_in_name = 1000
    for angle_max in [30, 60, 180]:
        data_folder_in = Path(
            f"cne_datasets/reichstag/numkp-2000/nn-1/nocrop/{part}-{nbr_in_name}/")
        data_folder_out = Path(
            f"cne_datasets/reichstag_rot{angle_max}/numkp-2000/nn-1/nocrop/{part}-{nbr_in_name}/")
        data_folder_out.mkdir(exist_ok=True, parents=True)
        # copy data, the xs and Rs are then overwritten by rotated versions.
        copy_tree(str(data_folder_in), str(data_folder_out))
        create_rotated_data(data_folder_in, data_folder_out, angle_max)
