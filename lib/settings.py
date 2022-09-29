from pydantic import BaseSettings, validator
from typing import Optional
from enum import Enum
from pathlib import Path
from datetime import datetime


class ModeEnum(str, Enum):
    easy = 'easy'
    hard = 'hard'


class Settings(BaseSettings):
    snapshot_path: Path = r"/root/pixel2point/dataset/image"
    output_path: Path = f"./output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    only: list[str] = ["chair"]
    mode: list[ModeEnum] = ["easy"]
    dataset_remake: list[bool] = [True]
    num_workers: list[int] = [0]
    pin_memory: list[bool] = [False]
    reproducibility: list[bool] = [False]
    seed: list[int] = [0]
    use_amp: list[bool] = [True]
    batch_size: list[int] = [32]
    shuffle: list[bool] = [True]
    resize: tuple[int, int] = (128, 128)
    device: list[str] = ["cuda"]
    save_model: list[bool] = [True]
    save_result: list[bool] = [True]

    initial_point: list[int] = [0]
    
    train_dataset_path: Path = r"/root/pixel2point/dataset/shapenetcorev2_hdf5_2048/train_files.txt"
    val_dataset_path: Path = r"/root/pixel2point/dataset/shapenetcorev2_hdf5_2048/val_files.txt"
    epoch: list[int] = [10]
    learning_rate: list[float] = [5e-5]
    
    test_dataset_path: Path = r"/root/pixel2point/dataset/shapenetcorev2_hdf5_2048/test_files.txt"
    model_path: Path = r"/root/pixel2point/model/2022-09-15_11-32-03_param.pt"


    @validator('only')
    @classmethod
    def only_must_in_shapenet_55_cls(cls, only):
        shapenet_55_cls = [
            'airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus',
            'cabinet', 'camera', 'can', 'cap', 'car', 'cellphone', 'chair', 'clock', 'dishwasher', 'earphone', 'faucet',
            'file', 'guitar', 'helmet', 'jar', 'keyboard', 'knife', 'lamp', 'laptop', 'mailbox', 'microphone',
            'microwave', 'monitor', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer',
            'remote_control', 'rifle', 'rocket', 'skateboard', 'sofa', 'speaker', 'stove', 'table', 'telephone',
            'tin_can', 'tower', 'train', 'vessel', 'washer'
        ]
        for element in only:
            if element not in shapenet_55_cls:
                raise ValueError(f'must be in {shapenet_55_cls}')
        return only

    class Config:
        env_file = '.env'


if __name__ == '__main__':
    settings = Settings()
    print(settings.dict())
