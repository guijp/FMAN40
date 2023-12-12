import torch
import os
import numpy as np
import random
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from utils.augmentation import random_rotation_and_flip, gaussian_blur, color_distortion
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self,
                 image_paths,
                 targets,
                 transforms,
                 device,
                 shuffle=True):
        self.image_paths = image_paths
        self.targets = targets
        self.transforms = transforms
        self.device = device

        if shuffle:
            rand = random.Random(0)
            rand.shuffle(self.image_paths)
            rand.shuffle(self.targets)

        #self.input_dtype = torch.float32
        #self.target_dtype = torch.float32

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])  # Rows, Columns, Channels
        image = image[:256, :256,:]
        target = self.targets[index]
        #image = np.moveaxis(image.squeeze(), source=-1, destination=0)
        #image = torch.from_numpy(image)
        image = self.transforms(image).to(self.device)
        # image = self._augmentation(image)

        #self._dispay_image(image, target)

        return image, target

    def _dispay_image(self, image, target):
        npimg = np.moveaxis(image.numpy(), source=0, destination=-1)
        plt.figure(0)
        plt.imshow(npimg)
        plt.title('Target: {}'.format(target))
        plt.show()
        plt.pause(4)
        plt.close('all')

    def _augmentation(self, image):
        # How much augmentation should be applied??
        x = random_rotation_and_flip(image)
        x = color_distortion(x, s=0.5)
        #x = gaussian_blur(x, sigma=(0.5, 1.5))

        return x


class ValidDataset(Dataset):
    def __init__(self,
                 image_paths,
                 targets,
                 transforms,
                 device,
                 shuffle=True):
        self.image_paths = image_paths
        self.targets = targets
        self.transforms = transforms
        self.device = device

        if shuffle:
            rand = random.Random(0)
            rand.shuffle(self.image_paths)
            rand.shuffle(self.targets)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])  # Rows, Columns, Channels
        image = image[:256, :256,:]
        target = self.targets[index]
        image = self.transforms(image).to(self.device)

        return image, target

def get_weights(targets, num_classes, one_hot=True):
    instances = torch.zeros(num_classes)
    for y in targets:
        if one_hot:
            idx = int(torch.nonzero(y))
        else:
            idx = int(y)
        instances[idx] += 1
    weights = torch.max(instances)/instances
    return weights


def get_datasets(datasets, device, val_split=0.1, binary=False):
    def load_img_paths(base_dir):
        input_img_paths = []
        targets = []

        for i,gg in enumerate(['benign', 'G3', 'G4', 'G5']):
            img_dir = os.path.join(base_dir, gg)
            for fname in os.listdir(img_dir):
                if fname.endswith(".jpg"):
                    input_img_paths.append(os.path.join(img_dir, fname))
                    if binary:
                        """tar = torch.zeros(2)
                        tar[int(gg != 'benign')] = 1.0"""
                        targets.append(int(gg != 'benign'))
                    else:
                        tar = torch.zeros(4)
                        tar[i] = 1.0
                        targets.append(tar)


        return input_img_paths, targets

    all_image_paths = []
    all_targets = []
    for data_dir in datasets:
        this_paths, this_labels = load_img_paths(data_dir)
        all_image_paths += this_paths
        all_targets += this_labels

    # Split into validation and training sets
    if val_split > 0.0:
        val_samples = int(len(all_image_paths) * val_split)
        random.Random(0).shuffle(all_image_paths)
        random.Random(0).shuffle(all_targets)
        tr_img_paths = all_image_paths[:-val_samples]
        val_img_paths = all_image_paths[-val_samples:]
        tr_targets = all_targets[:-val_samples]
        val_targets = all_targets[-val_samples:]
    else:
        tr_img_paths, tr_targets = all_image_paths, all_targets
        val_img_paths, val_targets = [], []

    tr_dataset = TrainDataset(tr_img_paths, tr_targets, transforms=transforms.ToTensor(), device=device, shuffle=False)
    val_dataset = ValidDataset(val_img_paths, val_targets, transforms=transforms.ToTensor(), device=device, shuffle=False)

    return tr_dataset, val_dataset


if __name__ == '__main__':
    """dir = "/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Rotterdam/test_299"

    dataset = GleasonDataset(dir, labels=['benign', 'malignant'], mode='pca', patch_size=256)
    dataloader = DataLoader(dataset, batch_size=10, drop_last=False)"""

    datasets = ["/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Rotterdam/test_299"]
    dataset, _ = get_datasets(datasets, val_split=0.0, binary=True)
    dataloader = DataLoader(dataset, batch_size=10, drop_last=False)

    for i, (images, labels) in enumerate(dataloader):
        print('Batch: {}'.format(i))
        print(labels)
        print(images.shape)

        npimg = np.moveaxis(images[0].numpy(), source=0, destination=-1)
        plt.figure()
        plt.imshow(npimg)
        plt.title('Target: {}'.format(labels[0]))

        if i > 15:
            break
    plt.show()