from .sampler import MATHStateSampler, MATHTask, MATHSampler

import os
import datasets
from torch.utils.data import Dataset


class MathDataset(Dataset):
    def __init__(self, train_file:str, split: str = "train"):
        if os.path.exists(train_file):
            ds = datasets.load_dataset("json", data_files=[train_file])["train"]
            data = [item for item in ds]
        else:
            ds = datasets.load_dataset(train_file, split=split)

            data = [
                {
                    "problem": item["prompt"][-1]["content"],
                    "answer": item["reward_model"]["ground_truth"],
                    "subject": item["ability"],
                    "level": 0
                }
                for item in ds if item["ability"] == "math"
            ]
        self.data = data
        if split != "train":
            self.data = self.data[:10]  # use a smaller dataset for validation/test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return MATHTask(
            problem=self.data[index]["problem"],
            answer=self.data[index]["answer"],
            subject=self.data[index]["subject"],
            level=self.data[index]["level"]
        )