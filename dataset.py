import cv2
import torch
import os
import numpy as np
import random
import math
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor

def tissue_segmentation(wsi, use_otsu=False):
    # This method segments the tissue from the background of a WSI, ignoring holes.
    # It was taken from https://github.com/mahmoodlab/CLAM
    # Input: wsi
    # Output: cv2 Contours object
    def _filter_contours(contours, hierarchy, min_area=255 * 255, max_n_holes=8):
        """
            Filter contours by: area.
        """
        filtered = []

        # find indices of foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        all_holes = []

        # loop through foreground contour indices
        for cont_idx in hierarchy_1:
            # actual contour
            cont = contours[cont_idx]
            # indices of holes contained in this contour (children of parent contour)
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            # take contour area (includes holes)
            a = cv2.contourArea(cont)
            # calculate the contour area of each hole
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            # actual area of foreground contour region
            a = a - np.array(hole_areas).sum()
            # print(a)
            # self.displayContours(img.copy(), cont)
            if a == 0: continue
            if min_area < a:
                filtered.append(cont_idx)
                all_holes.append(holes)

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]

        hole_contours = []

        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids]
            unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
            # take max_n_holes largest holes by area
            unfilered_holes = unfilered_holes[:max_n_holes]
            filtered_holes = []

            # filter these holes
            for hole in unfilered_holes:
                if cv2.contourArea(hole) > min_area:
                    filtered_holes.append(hole)

            hole_contours.append(filtered_holes)

        return foreground_contours, hole_contours

    # wsi = cv2.imread(wsi_path)
    img_hsv = cv2.cvtColor(wsi, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_med = cv2.medianBlur(img_hsv[:, :, 1], 7)  # Apply median blurring

    # Thresholding
    if use_otsu:
        _, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, img_otsu = cv2.threshold(img_med, 8, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours

    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
    return _filter_contours(contours, hierarchy)  # Necessary for filtering out artifacts

class WSIAEDataset(Dataset):
    # Use this class when training an autoencoder
    # This class is used to load a list of WSIs and extract patches from them.
    # It uses the tissue segmentation method from above to ignore background.
    # Arguments:
    #  wsi_paths: list of paths to the WSIs
    #  patch_size: size of the patches to extract (a number, the patches will be square)
    #  overlap: overlap between patches (in pixels)
    def __init__(self,
                 wsi_paths,
                 patch_size,
                 overlap,
                 device):
        self.paths = wsi_paths

        self.current_wsi = 0
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = ToTensor()

        self.all_patches = []
        for i, url in enumerate(self.paths):
            if not os.path.exists(url):
                raise FileNotFoundError(url)
            self._load_wsi(url)
            coords = self._get_patches()
            xyw = [(x, y, i) for (x, y) in coords]
            self.all_patches = self.all_patches + xyw

    def _load_wsi(self, path):
        self.wsi = imread(path)
        self.shape = self.wsi.shape
        if self.shape[-1] > 3:
            print('Found more than three channels, keeping primary three')
            self.wsi = self.wsi[:, :, :3]

    def _get_patches(self):
        foreground_contours, _ = tissue_segmentation(self.wsi)
        patch_coords = []
        for contour in foreground_contours:
            start_x, start_y, w, h = cv2.boundingRect(contour)
            img_h, img_w = self.shape[:2]
            stop_y = min(start_y + h, img_h - self.patch_size + 1)
            stop_x = min(start_x + w, img_w - self.patch_size + 1)

            if stop_x < start_x or stop_y < start_y:
                continue

            step_size_x = math.floor(
                (stop_x - start_x) / math.ceil((stop_x - start_x) / (self.patch_size - self.overlap)))
            step_size_y = math.floor(
                (stop_y - start_y) / math.ceil((stop_y - start_y) / (self.patch_size - self.overlap)))

            x_range = np.arange(start_x, stop_x, step=step_size_x)
            y_range = np.arange(start_y, stop_y, step=step_size_y)
            x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
            coords = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

            for pt in coords:
                cent = pt + int(self.patch_size / 2)
                if cv2.pointPolygonTest(contour, tuple(np.array(cent).astype(float)),
                                        measureDist=False) > -1:  # check that point is within contour
                    patch_coords.append(pt)

        return patch_coords

    def __len__(self):
        return len(self.all_patches)

    def __getitem__(self, index):
        (x, y, wsi) = self.all_patches[index]
        if wsi != self.current_wsi:
            self._load_wsi(self.paths[wsi])
            self.current_wsi = wsi

        patch = self.wsi[y:y + self.patch_size, x:x + self.patch_size, :]
        
        patch = self.transform(patch).to(self.device)
        return patch

# class WSIDataset(Dataset):
#     def __init__(self, path, patch_size):
#         if not os.path.exists(path):
#             raise FileNotFoundError(path)
#         self.path = path
#         self.patch_size = patch_size

#     def _load_image(self):
#         self.wsi = imread(self.path)

# class ImageDataset(Dataset):
#     def __init__(self, path):
#         self.path = path
#         self.file_names = [file for file in os.listdir(self.path) if file[-3:] == 'png']
    
#     def __getitem__(self, idx):
#         path = self.path + f'/patch{idx}.png'
#         patch = imread(path).astype(np.float32)/255
#         patch = torch.from_numpy(patch[:,:,:3])
#         patch = patch.reshape(3, 200, 200)
#         return patch
    
#     def show(self, idx):
#         patch = self[idx].reshape(200, 200, 3).detach().numpy()
#         imshow(patch)
#         plt.show()

#     def __len__(self):
#         return len(self.file_names)
    
#     def rescale(self, m, n):
#         for i in range(len(self)):
#             patch = self[i].reshape(200, 200, 3).detach().numpy()
#             patch = patch[:m, :n, :]
#             imsave(self.path + '/' + self.file_names[i], patch)

