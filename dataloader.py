from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
import json
import h5py
from PIL import Image

from logger import logger


class ShapenetDataset(Dataset):
    def __init__(self, dataset_path, snapshot_path, transforms=None, only: list = None, mode: str = 'easy'):
        self.dataset_path = dataset_path
        self.snapshot_path = snapshot_path
        self.transforms = transforms
        self.only = only
        self.mode = mode
        self.source_name, self.source_file, self.source_points = self.read_dataset()
        self.image_path, self.gt_points, self.length = self.convert2realpath()

    def read_dataset(self):
        logger.debug('==================================')
        logger.debug(f'Read Dataset')
        all_name, all_file, all_gt_points = np.empty([0]), np.empty([0]), np.empty([0, 2048, 3])
        with open(self.dataset_path, 'r') as f:
            train_files = f.readlines()
            np_train_files = np.char.split(train_files, sep='\n')
        read_bar = tqdm(np_train_files, unit='file', leave=True)
        for key, value in enumerate(read_bar):
            index = value[0].split('.')[0]
            train_name_json = Path(self.dataset_path).parent.parent.joinpath(f"{index}_id2name.json")
            train_file_json = Path(self.dataset_path).parent.parent.joinpath(f"{index}_id2file.json")
            train_h5 = Path(self.dataset_path).parent.parent.joinpath(f"{index}.h5")

            with open(train_name_json) as f:
                name = np.array(json.load(f))

            if self.only is not None:
                target_cls = np.isin(name, self.only)
            else:
                target_cls = tuple()

            name = name[target_cls]

            with open(train_file_json) as f:
                file = np.array(json.load(f))[target_cls]

            h5 = h5py.File(train_h5, 'r')['data'][:][target_cls]

            read_bar.set_description(f'File [{key + 1}/{len(np_train_files)}]')
            read_bar.set_postfix_str(f'Count: {len(name)}')

            all_name = np.r_[all_name, name]
            all_file = np.r_[all_file, file]
            all_gt_points = np.r_[all_gt_points, h5]
        logger.debug(f'Count Sum (Point Model): {len(all_name)}')
        logger.debug(f'Ground Truth Size (MB): {all_gt_points.itemsize * all_gt_points.size / 1024 / 1024}')
        return all_name, all_file, all_gt_points

    def convert2realpath(self):
        logger.debug('==================================')
        logger.debug(f'Convert Dataset')
        images, points = np.empty([0]), np.empty([0], dtype=int)
        root_path = Path(self.snapshot_path)
        convert_bar = tqdm(self.source_file, unit='file', leave=True)
        for key, value in enumerate(convert_bar):
            dir_path = value.split('.npy')[0]
            current_path = str(root_path.joinpath(f'{dir_path}/{self.mode}'))
            if not Path(current_path).exists():
                continue
            image_count = sum(1 for element in Path(current_path).iterdir() if element.suffix == '.png')
            scope = np.array(list(map('{:02d}.png'.format, np.arange(image_count))))
            png_files = np.char.add(f'{str(current_path)}/', scope)

            images = np.r_[images, png_files]
            points = np.r_[points, np.full(png_files.shape, key, dtype=int)]

            convert_bar.set_description(f'File [{key + 1}/{len(self.source_file)}]')
            convert_bar.set_postfix_str(f'Image Count: {image_count}')
        logger.debug(f'Total Images: {len(images)}')
        return images, points, len(images)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        gt_point = self.source_points[self.gt_points[index]]
        return image, gt_point, index

    def __len__(self):
        return self.length
