from copyreg import pickle
import torch
from torch.utils.data import Dataset
from torch import Tensor
import pickle

class MMWaveTracking(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.path = "hd-generalized-tracking.p"
        self.desired_obj = "Person"

        self.d = []

        with open(self.path, "rb") as f:
            self.d = pickle.load(f)


        self.max_pts = max([len(x['pointcloud']) for x in self.d])
        self.max_objs = max([len(x['objects']) for x in self.d])

        # print(len(d))
        print(self.d[0].keys())

        print("max objects", max([len(x['objects']) for x in self.d]))
        print("min ptcloud", min([len(x['pointcloud']) for x in self.d]))

        # print("first object", self.d[0]['objects'][0])
        # print("first pt", self.d[0]['pointcloud'][0])

        # print([{'label': x['label']} for x in self.d[0]['objects']])

    def __len__(self):
        return len(self.d)

    def __getitem__(self, index) -> tuple:
        pts = torch.tensor(self.d[index]['pointcloud'])
        objs = torch.tensor([x['position'] for x in self.d[index]['objects'] if x['label'] == self.desired_obj])

        out_pts = torch.zeros((self.max_pts, 4))
        out_pts[:pts.shape[0], :] = pts

        out_objs = torch.zeros((self.max_objs, 3))
        # print("objs", objs.shape, out_objs.shape)
        if objs.shape[0] > 0:
            out_objs[:objs.shape[0], :] = objs

        obj_scores = torch.zeros((self.max_objs, 1))
        obj_scores[:objs.shape[0], :] = 1.0 
        
        # pts_mask = torch.zeros(out_pts)
        # pts_mask

        return out_pts, out_objs, obj_scores