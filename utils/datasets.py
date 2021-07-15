import os
import torch
import pandas as pd

from torch.utils.data import Dataset
from .augmentations import transform


class ListDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    Labels must start with 1
    Boxes in TopLeft,BottomRigth format
    """

    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'VAL' or 'TEST'
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST', 'VAL'}

        self.data_folder = data_folder

        # Read data files
        data = pd.read_csv(os.path.join(data_folder, f"{self.split}_annotation.csv"))
        df = pd.DataFrame(data)
        self.countries = list(set(df["MRZ КОД СТРАНЫ"].tolist()))
        self.df = df.copy()
        self.df_cur = self.df

        if self.split == 'TRAIN':
            self.select_train_data()

        del data, df

    def __getitem__(self, i):

        # get current row in data frame
        cur_row = self.df_cur.iloc[i]
        # Determine a path to image
        path2image = cur_row["ПУТЬ К ФАЙЛУ"]
        # Read objects in this image (box, angle, orient)
        box = list(map(int, cur_row["КООРДИНАТЫ МРЗ"].rstrip(")").lstrip("(").split(",")))
        angle = cur_row["УГОЛ НАКЛОНА"]
        orient = cur_row["ОРИЕНТАЦИЯ"]

        # Apply transformations
        image, box8points, target = transform(path2image, box, angle, orient, self.split, cut=cur_row["CUT"])

        return image, box8points, target

    def __len__(self):
        return len(self.df_cur.index)

    def select_train_data(self):

        new_df = pd.DataFrame([])
        self.countries = list(set(self.df[self.df["CUT"] == True]["MRZ КОД СТРАНЫ"].tolist()))

        for i, code_country in enumerate(self.countries):
            
            cur_df = self.df[self.df["CUT"] == True]
            cur_df = cur_df[cur_df["MRZ КОД СТРАНЫ"] == code_country].sample(n=200, replace=True)

            if i == 0:
                new_df = cur_df
            else:
                new_df = pd.concat([new_df, cur_df])
        
        new_df = pd.concat([new_df, self.df[self.df["CUT"] == False][:1500]])
        self.df_cur = new_df


if __name__ == "__main__":
    # Custom DataLoaders
    train_dataset = ListDataset("../annotation", split='train')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                               collate_fn=train_dataset.collate_fn, num_workers=1,
                                               pin_memory=True)  # note that we're passing the collate function here

    for _ in range(10):
        print('--' * 10)
        train_dataset.select_train_data()
        for image, boxes, tar in train_loader:
            print('--' * 10)
            break
