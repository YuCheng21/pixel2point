from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
import json
import h5py
from PIL import Image
from kmeans_pytorch import kmeans
import torch

from lib.logger import logger


class ShapenetDataset(Dataset):
    def __init__(self, dataset_path, snapshot_path, transforms=None,
                 only: list = None, mode: str = 'easy', remake: bool = True):
        self.dataset_path = dataset_path
        self.snapshot_path = snapshot_path
        self.transforms = transforms
        self.only = only
        self.mode = mode

        self.data_type = self.dataset_path.stem.split("_files")[0]

        for key, value in enumerate(self.only):
            if self.check_final_h5(value) is True and remake is False:
                source_name, source_file, source_points, \
                    image_path, gt_points, image = self.read_final_h5(value)
            else:
                source_name, source_file, source_points = self.read_dataset([value])
                image_path, gt_points, length = self.convert2realpath(source_file)
                image = self.open_image(image_path)

                source_points = self.points_normalization(source_points)
                # source_points = self.points_classification(source_points)
                
                self.save_final_h5(value, source_name, source_file, source_points, image_path, gt_points, image)

            if key == 0:
                self.source_name = source_name
                self.source_file = source_file
                self.source_points = source_points
                self.image_path = image_path
                self.gt_points = gt_points
                self.length = len(image_path)
                self.image = image
            else:
                self.source_name = np.concatenate((self.source_name, source_name), 0)
                self.source_file = np.concatenate((self.source_file, source_file), 0)
                self.source_points = np.concatenate((self.source_points, source_points), 0)
                self.image_path = np.concatenate((self.image_path, image_path), 0)
                self.gt_points = np.concatenate((self.gt_points, gt_points), 0)
                self.length = self.length + len(image_path)
                self.image = np.concatenate((self.image, image), 0)
        
        if self.length == 0:
            raise Exception('no data found')
        logger.debug(f'Dataset Type: {self.data_type}, Point Model: {len(self.source_points)}, Images: {self.length}')

    def check_final_h5(self, filename):
        h5_name = f'shapenet_{filename}_{self.data_type}_{self.mode}.h5'
        h5_path = self.snapshot_path.parent.joinpath('h5')
        h5_path.mkdir(parents=True, exist_ok=True)
        return h5_path.joinpath(f'{h5_name}').exists()

    def read_final_h5(self, filename):
        h5_name = f'shapenet_{filename}_{self.data_type}_{self.mode}.h5'
        h5_path = self.snapshot_path.parent.joinpath(f'h5/{h5_name}')
        with h5py.File(h5_path, 'r') as f:
            return [
                f['source_name'][:], f['source_file'][:], f['source_points'][:],
                f['image_path'][:], f['gt_points'][:],
                f['image'][:]
            ]

    def save_final_h5(self, filename, source_name, source_file, source_points, image_path, gt_points, image):
        h5_name = f'shapenet_{filename}_{self.data_type}_{self.mode}.h5'
        h5_path = self.snapshot_path.parent.joinpath(f'h5/{h5_name}')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('source_name', data=source_name.tolist())
            f.create_dataset('source_file', data=source_file.tolist())
            f.create_dataset('source_points', data=source_points)
            f.create_dataset('image_path', data=image_path.tolist())
            f.create_dataset('gt_points', data=gt_points)
            f.create_dataset('image', data=image)

    def read_dataset(self, only):
        all_name, all_file, all_gt_points = np.empty([0]), np.empty([0]), np.empty([0, 2048, 3], dtype=np.float32)
        with open(self.dataset_path, 'r') as f:
            files = f.readlines()
            np_files = np.char.split(files, sep='\n')
        read_bar = tqdm(np_files, unit='file', leave=True, colour='#DA7E7E')
        for key, value in enumerate(read_bar):
            index = value[0].split('.')[0]
            prefix_path = Path(self.dataset_path).parent.parent.joinpath(f"{Path(index).parent.name}")
            name_json = prefix_path.joinpath(f"{Path(index).name}_id2name.json")
            file_json = prefix_path.joinpath(f"{Path(index).name}_id2file.json")
            h5 = prefix_path.joinpath(f"{Path(index).name}.h5")

            with open(name_json) as f:
                name = np.array(json.load(f))

            if only is not None:
                target_cls = np.isin(name, only)
            else:
                target_cls = tuple()

            name = name[target_cls]

            with open(file_json) as f:
                file = np.array(json.load(f))[target_cls]

            with h5py.File(h5, 'r') as f:
                h5_data = f['data'][:][target_cls]

            read_bar.set_description(f'Read Dataset (File) [{key + 1}/{len(np_files)}]')
            read_bar.set_postfix_str(f'Count: {len(name)}')

            all_name = np.concatenate((all_name, name), 0)
            all_file = np.concatenate((all_file, file), 0)
            all_gt_points = np.concatenate((all_gt_points, h5_data), 0)
        return all_name, all_file, all_gt_points
    
    def points_normalization(self, source_points):
        size = source_points.shape[0]
        n_points = torch.from_numpy(source_points).view(size, -1)
        n_points = n_points - n_points.min(1, keepdim=True)[0]
        n_points = n_points / n_points.max(1, keepdim=True)[0]
        n_points = n_points.view(size, -1, 3)
        return n_points.numpy()
    
    def points_classification(self, source_points):
        cluster = torch.empty([0, 2048])
        for key, value in enumerate(source_points):
            cluster_ids_x, _ = kmeans(X=torch.from_numpy(value), num_clusters=3, distance='euclidean', tol=1e-4, device='cuda')
            cluster = torch.cat((cluster, cluster_ids_x.view(1, -1)), 0)
        return torch.cat((torch.from_numpy(source_points), cluster.unsqueeze(2)), 2)

    def convert2realpath(self, source_file):
        images, points = np.empty([0]), np.empty([0], dtype=int)
        root_path = Path(self.snapshot_path)
        convert_bar = tqdm(source_file, unit='file', leave=True, colour='#DAB97E')
        for key, value in enumerate(convert_bar):
            dir_path = value.split('.npy')[0]
            current_path = str(root_path.joinpath(f'{dir_path}/{self.mode}'))
            if not Path(current_path).exists():
                continue
            image_count = sum(1 for element in Path(current_path).iterdir() if element.suffix == '.png')
            scope = np.array(list(map('{:02d}.png'.format, np.arange(image_count))))
            png_files = np.char.add(f'{str(current_path)}/', scope)

            images = np.concatenate((images, png_files), 0)
            points = np.concatenate((points, np.full(png_files.shape, key, dtype=int)), 0)

            convert_bar.set_description(f'Convert Dataset (Point Model) [{key + 1}/{len(source_file)}]')
            convert_bar.set_postfix_str(f'Image Count: {image_count}')
        return images, points, len(images)

    def open_image(self, image_path):
        image = []
        open_bar = tqdm(image_path, unit='file', leave=True, colour='#DADA7E')
        for key, value in enumerate(open_bar):
            image += [Image.open(value).copy()]
            image[-1] = image[-1].convert("RGB")
            if self.transforms is not None:
                image[-1] = self.transforms(image[-1]).numpy()
            open_bar.set_description(f'Open Image (File) [{key + 1}/{len(image_path)}]')
        return image

    def __getitem__(self, index):
        return self.image[index], self.source_points[self.gt_points[index]], index

    def __len__(self):
        return self.length
