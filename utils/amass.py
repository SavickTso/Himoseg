from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt
import torch
import os
import re
import IPython
from utils import ang2joint


class Datasets(Dataset):
    def __init__(self, actions=None, split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = "./datasets/amass/"
        self.split = split
        # self.in_n = opt.input_n
        # self.out_n = opt.output_n
        # self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)
        # seq_len = self.in_n + self.out_n
        labels = np.load("./movilabels.npy")

        pattern = r"Subject_(\d+)_F_(\d+)_poses.npz"

        skel = np.load("./body_models/smpl_skeleton.npz")
        p3d0 = torch.from_numpy(skel["p3d0"]).float().cuda()
        parents = skel["parents"]
        parent = {}
        for i in range(len(parents)):
            parent[i] = parents[i]
        n = 0
        # for ds in amass_splits[split]:
        ds = "BMLmovi"
        if not os.path.isdir(self.path_to_data + ds):
            print(ds)
            exit
        print(">>> loading {}".format(ds))
        for sub in os.listdir(self.path_to_data + ds):
            if not os.path.isdir(self.path_to_data + ds + "/" + sub):
                continue
            for act in os.listdir(self.path_to_data + ds + "/" + sub):
                if not act.endswith(".npz"):
                    continue
                # if not ('walk' in act or 'jog' in act or 'run' in act or 'treadmill' in act):
                #     continue
                pose_all = np.load(self.path_to_data + ds + "/" + sub + "/" + act)
                try:
                    poses = pose_all["poses"]
                except:
                    # print("no poses at {}_{}_{}".format(ds, sub, act))
                    continue
                frame_rate = pose_all["mocap_framerate"]

                # fn = poses.shape[0]
                # sample_rate = 1  # int(frame_rate // 25)
                # fidxs = range(0, fn, sample_rate)
                # fn = len(fidxs)
                # poses = poses[fidxs]
                poses = torch.from_numpy(poses).float().cuda()
                poses = poses.reshape([poses.shape[0], -1, 3])
                # remove global rotation
                poses[:, 0] = 0
                p3d0_tmp = p3d0.repeat([poses.shape[0], 1, 1])
                p3d = ang2joint.ang2joint(p3d0_tmp, poses, parent)
                # print("for act {}, the shape of p3d is {}".format(act, p3d.shape))
                # self.p3d[(ds, sub, act)] = p3d.cpu().data.numpy()
                self.p3d.append(p3d.cpu().data.numpy())

                # valid_frames = np.arange(0, fn, skip_rate)

                # # tmp_data_idx_1 = [(ds, sub, act)] * len(valid_frames)
                print("extracted number in this act", act)

                match = re.search(pattern, act)
                # print(int(match.group(1)), "   ", int(match.group(2)))
                if int(match.group(2)) == 22:
                    self.keys.append("scratching_head")
                else:
                    self.keys.append(
                        labels[int(match.group(1)) - 1][int(match.group(2)) - 1]
                    )

                # tmp_data_idx_1 = [n] * len(valid_frames)
                # tmp_data_idx_2 = list(valid_frames)
                # self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                # n += 1
        IPython.embed()

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        # fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[key]  # [fs]  # , key
