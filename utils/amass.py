import json
import os
import re
import sys
from os.path import join as ospj

import IPython
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from h5py import File
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import ang2joint, data_utils

EXCLUDE_CLASS_LIST = [
    "bicep_curls_rm",
    "coughing_rm",
    "dj-ing_rm",
    "dribbling_rm",
    "fencing_rm",
    "front_swimming_rm",
    "hopping_rm",
    "juggling_rm",
    "lunges_rm",
    "punching_rm",
    "pushups_rm",
    "rowing_rm",
    "serving_rm",
    "stretching_rm",
    "wearing_belt_rm",
    "yoga_rm",
    "free_throw_rm",
    "throwing_frisbee_rm",
    "swinging_arms_rm",
    "swinging_racket_rm",
]


def motion_downsample(fn, poses, sample_rate):
    fidxs = range(0, fn, sample_rate)
    fn = len(fidxs)
    poses = poses[fidxs]
    return fn, poses


def custom_collate(batch):
    # Find the length of the longest data sample in the batch
    max_len = max(len(data) for data in batch)

    # Pad or truncate each data sample to match the max length
    padded_data = [
        torch.nn.functional.pad(data, (0, max_len - len(data))) for data in batch
    ]

    # Stack the padded data samples to create a batch
    padded_data = torch.stack(padded_data)

    return padded_data


def load_babel():
    d_folder = "babel_v1.0_release"  # Data folder
    l_babel_dense_files = ["train", "val", "test"]
    l_babel_extra_files = ["extra_train", "extra_val"]

    # BABEL Dataset
    babel = {}
    for file in l_babel_dense_files:
        babel[file] = json.load(open(ospj(d_folder, file + ".json")))

    for file in l_babel_extra_files:
        babel[file] = json.load(open(ospj(d_folder, file + ".json")))

    return babel


def get_submotion_frame_range_test(babel_split, motion_featp_str, motion_len_frame):
    sublabel = []
    sublabel_seg = []
    for clip in babel_split:
        if babel_split[clip]["feat_p"] == motion_featp_str:
            if babel_split[clip]["frame_ann"] == None:
                return [babel_split[clip]["seq_ann"]["labels"][0]["proc_label"]], [
                    0,
                    motion_len_frame,
                ]
            for submotion in babel_split[clip]["frame_ann"]["labels"]:
                sublabel.append(submotion["proc_label"])
                start = round(
                    motion_len_frame * submotion["start_t"] / babel_split[clip]["dur"]
                )
                end = round(
                    motion_len_frame * submotion["end_t"] / babel_split[clip]["dur"]
                )
                sublabel_seg.append([start, end])
            break
    # Sort the list based on the second element of each tuple
    sorted_list = sorted(sublabel_seg, key=lambda x: x[0])
    sorted_label_list = [
        sublabel[i]
        for i in sorted(range(len(sublabel_seg)), key=lambda k: sublabel_seg[k][0])
    ]

    return sorted_label_list, sorted_list


class Datasets(Dataset):
    def __init__(self, actions=None):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        # self.path_to_data = "../../dataset/amass/"
        self.path_to_data = "./datasets/amass/"
        # self.in_n = opt.input_n
        # self.out_n = opt.output_n
        # self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)
        # seq_len = self.in_n + self.out_n
        self.babel = load_babel()
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
            sys.exit()
        print(">>> loading {}".format(ds))
        for sub in os.listdir(self.path_to_data + ds):
            if not os.path.isdir(self.path_to_data + ds + "/" + sub):
                continue
            for act in os.listdir(self.path_to_data + ds + "/" + sub):
                if not act.endswith(".npz"):
                    continue

                pose_all = np.load(self.path_to_data + ds + "/" + sub + "/" + act)
                try:
                    poses = pose_all["poses"]
                except:
                    # print("no poses at {}_{}_{}".format(ds, sub, act))
                    continue
                frame_rate = pose_all["mocap_framerate"]
                fn = poses.shape[0]

                ### start of down sample
                # sample_rate = int(frame_rate // 25)
                # fn, poses = motion_downsample(fn, poses, sample_rate)
                ### end of down sample
                print("processing {}".format(ds + "/" + sub + "/" + act))
                sub_key = ds + "/" + sub + "/" + act
                poses = torch.from_numpy(poses).float().cuda()
                poses = poses.reshape([fn, -1, 3])
                # remove global rotation
                poses[:, 0] = 0
                p3d0_tmp = p3d0.repeat([fn, 1, 1])
                p3d = ang2joint.ang2joint(p3d0_tmp, poses, parent)
                # p3d_pad = pad_sequence(p3d, batch_first=True)
                # print("for act {}, the shape of p3d is {}".format(act, p3d.shape))
                # self.p3d[(ds, sub, act)] = p3d.cpu().data.numpy()
                # valid_frames = np.arange(0, fn, skip_rate)

                # # tmp_data_idx_1 = [(ds, sub, act)] * len(valid_frames)
                # print("extracted number in this act", act)

                match = re.search(pattern, act)
                # print(int(match.group(1)), "   ", int(match.group(2)))

                ### skip classes that have less than 5 samples

                if int(match.group(2)) == 22:
                    self.keys.append("scratching_head")
                elif (
                    labels[int(match.group(1)) - 1][int(match.group(2)) - 1]
                    in EXCLUDE_CLASS_LIST
                ):
                    continue
                else:
                    self.keys.append(
                        labels[int(match.group(1)) - 1][int(match.group(2)) - 1]
                    )

                self.p3d.append(p3d.cpu().data.numpy())
                featpstr = "BMLmovi/" + ds + "/" + sub + "/" + act
                print("featpstr is", featpstr)
                print("shape of current clip is", p3d.cpu().data.numpy().shape)
                motion_len = p3d.cpu().data.numpy().shape[0]
                sublabel, sublabel_seg = get_submotion_frame_range_test(
                    self.babel["train"], featpstr, motion_len
                )
                print(sublabel)
                print(sublabel_seg)

                # tmp_data_idx_1 = [n] * len(valid_frames)
                # tmp_data_idx_2 = list(valid_frames)
                # self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                # n += 1
        # self.data = self.p3d

        self.data = pad_sequence(
            [torch.tensor(arr) for arr in self.p3d], batch_first=True
        )
        print("self.data's shape before transpose:", self.data.shape)
        self.data = torch.einsum("nctw->nwct", self.data)
        # self.data = (
        #     self.data.permute(0, 2, 1, 3).contiguous().flatten(start_dim=2, end_dim=3)
        # )
        # self.data = self.data.flatten(start_dim=2, end_dim=3)
        print("self.data's shape after transpose:", self.data.shape)
        string_labels = self.keys
        label_encoder = LabelEncoder()
        self.numeric_labels = label_encoder.fit_transform(string_labels)
        print(label_encoder.classes_, len(label_encoder.classes_))
        # IPython.embed()
        # print(self.numeric_labels)

        # df = pd.DataFrame({"labels": self.keys})
        # numeric_labels = pd.get_dummies(df, columns=["labels"])
        # for lb in numeric_labels.columns:
        #     print(lb, sum(numeric_labels[lb]))
        # IPython.embed()
        # print(numeric_labels)

    def __len__(self):
        return len(self.numeric_labels)
        # return np.shape(self.data_idx)[0]

    def __iter__(self):
        return self

    def __getitem__(self, item):
        # key, start_frame = self.data_idx[item]
        # fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        data = self.data[item]
        label = self.numeric_labels[item]
        sublabel = torch.tensor([])
        return data, label, sublabel  # [fs]  # , key
