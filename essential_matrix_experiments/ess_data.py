# Heavily based on OANet
# https://github.com/zjhthu/OANet
import cv2
import h5py
import numpy as np
import pickle
from pathlib import Path
import torch
import utils


def correct_matches(e_gt):
    step = 0.1
    xx, yy = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step))
    # Points in first image before projection
    pts1_virt_b = np.float32(np.vstack((xx.flatten(), yy.flatten())).T)
    # Points in second image before projection
    pts2_virt_b = np.float32(np.vstack((xx.flatten(), yy.flatten())).T)

    pts1_virt_b, pts2_virt_b = pts1_virt_b.reshape(1, -1, 2), \
        pts2_virt_b.reshape(1, -1, 2)

    # Move points so that the points in
    # the first image match the points in the second.
    pts1_virt_b, pts2_virt_b = \
        cv2.correctMatches(e_gt.reshape(3, 3), pts1_virt_b, pts2_virt_b)

    # Filtering of nan-values, perhaps it shouldn't be necessary?
    pts1_virt_b = np.where(np.isfinite(pts1_virt_b), pts1_virt_b, 0)
    pts2_virt_b = np.where(np.isfinite(pts2_virt_b), pts2_virt_b, 0)

    return pts1_virt_b.squeeze(), pts2_virt_b.squeeze()


class EssentialDatasetPkl(torch.utils.data.Dataset):
    def __init__(self, dataset_name, partition, data_folder="cne_datasets", indices=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.data = None
        if partition == "train":
            self.folder_path = Path(data_folder) / self.dataset_name / \
                "numkp-2000/nn-1/nocrop" / "tr-10000"
            self.part_short = "tr"
        elif partition == "validation":
            self.folder_path = Path(data_folder) / self.dataset_name / \
                "numkp-2000/nn-1/nocrop" / "va-1000"
            self.part_short = "va"
        elif partition == "test":
            self.folder_path = Path(data_folder) / self.dataset_name / \
                "numkp-2000/nn-1/nocrop" / "te-1000"
            self.part_short = "te"
        else:
            raise ValueError("partition should be train, validation or test")

        if indices is None:
            self.length = self._len()
            self.indices = range(self.length)
        else:
            self.length = len(indices)
            self.indices = indices

    def __getitem__(self, index):
        if self.data is None:
            self.data = {}
            with open(self.folder_path /
                      f"Rs_{self.part_short}.pkl", "rb") as pkl_file:
                self.data['Rs'] = pickle.load(pkl_file)
            with open(self.folder_path /
                      f"ts_{self.part_short}.pkl", "rb") as pkl_file:
                self.data['ts'] = pickle.load(pkl_file)
            with open(self.folder_path /
                      f"xs_{self.part_short}.pkl", "rb") as pkl_file:
                self.data['xs'] = pickle.load(pkl_file)
            with open(self.folder_path /
                      f"ys_{self.part_short}.pkl", "rb") as pkl_file:
                self.data['ys'] = pickle.load(pkl_file)

        if self.indices is None:
            idx = index
        else:
            idx = self.indices[index]

        xs = (self.data['xs'][idx])
        ys = (self.data['ys'][idx])
        R = (self.data['Rs'][idx])
        t = (self.data['ts'][idx])
        angles = utils.angle_factorization_essential_matrix(
            R, t)

        e_gt = utils.normalize_E_batch(
            utils.essential_matrix_from_angles(angles,
                                               use_torch=False,
                                               flip_sign=False,
                                               batch=False),
            use_torch=False)

        pts1_virt, pts2_virt = correct_matches(e_gt)
        pts_virt = np.concatenate([pts1_virt, pts2_virt],
                                  axis=1).astype('float64')

        return {
            'xs': xs,
            'ys': ys,
            'R': R,
            't': t,
            'angles': angles,
            'virtPts': pts_virt
        }

    def reset(self):
        # if self.data is not None:
        #     self.data.close()
        self.data = None

    def __len__(self):
        return self.length

    def _len(self):
        if self.data is None:
            self.data = {}
            with open(self.folder_path /
                      f"Rs_{self.part_short}.pkl", "rb") as pkl_file:
                self.data['Rs'] = pickle.load(pkl_file)
            length = len(self.data['Rs'])
            self.reset()
        else:
            length = len(self.data['Rs'])
        return length

    def __del__(self):
        # if self.data is not None:
        #    self.data.close()
        pass


class EssentialDataset(torch.utils.data.Dataset):
    def __init__(self, filename, indices=None):
        super().__init__()
        self.filename = filename
        self.data = None
        if indices is None:
            self.length = self._len()
        else:
            self.length = len(indices)
            self.indices = indices

    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File(self.filename, 'r')
        if self.indices is None:
            idx = index
        else:
            idx = self.indices[index]

        xs = np.asarray(self.data['xs'][str(idx)])
        ys = np.asarray(self.data['ys'][str(idx)])
        R = np.asarray(self.data['Rs'][str(idx)])
        t = np.asarray(self.data['ts'][str(idx)])
        angles = np.asarray(self.data['angles'][str(idx)])

        e_gt = utils.normalize_E_batch(
            utils.essential_matrix_from_angles(angles,
                                               use_torch=False,
                                               flip_sign=False,
                                               batch=False),
            use_torch=False)

        pts1_virt, pts2_virt = correct_matches(e_gt)
        pts_virt = np.concatenate([pts1_virt, pts2_virt],
                                  axis=1).astype('float64')

        return {
            'xs': xs,
            'ys': ys,
            'R': R,
            't': t,
            'angles': angles,
            'virtPts': pts_virt
        }

    def reset(self):
        if self.data is not None:
            self.data.close()
        self.data = None

    def __len__(self):
        return self.length

    def _len(self):
        if self.data is None:
            self.data = h5py.File(self.filename, 'r')
            length = len(self.data['xs'])
            self.reset()
        else:
            length = len(self.data['xs'])
        return length

    def __del__(self):
        if self.data is not None:
            self.data.close()


def collate_fn(batch):
    """
    Transfers data to pytorch tensors.
    Also normalizes the number of points per batch.
    """
    nbr_points = [sample['xs'].shape[1] for sample in batch]
    cur_nbr_p = min(nbr_points)

    data = {key: [] for key in ['R', 't', 'angles', 'xs', 'ys', 'virtPts']}

    for sample in batch:
        data['R'].append(sample['R'])
        data['t'].append(sample['t'])
        data['angles'].append(sample['angles'])
        data['virtPts'].append(sample['virtPts'])
        if sample['xs'].shape[1] > cur_nbr_p:
            sub_idx = np.random.choice(sample['xs'].shape[1],
                                       cur_nbr_p,
                                       replace=False)
            data['xs'].append(sample['xs'][:, sub_idx, :])
            data['ys'].append(sample['ys'][sub_idx, :])
        else:
            data['xs'].append(sample['xs'])
            data['ys'].append(sample['ys'])

    for key in data:
        data[key] = torch.from_numpy(
            np.stack(data[key])).float()

    return data


if __name__ == "__main__":
    pass
