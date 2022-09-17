from pydantic import BaseSettings, validator
from enum import Enum
from pathlib import Path


class ModeEnum(str, Enum):
    easy = 'easy'
    hard = 'hard'


class Settings(BaseSettings):
    snapshot_path: Path = r"/root/pixel2point/dataset/image"
    only: list[str] = ["chair"]
    mode: ModeEnum = "easy"

    @validator('only')
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


class Training(Settings):
    train_dataset_path: Path = r"/root/pixel2point/dataset/shapenetcorev2_hdf5_2048/train_files.txt"


class Testing(Settings):
    test_dataset_path: Path = r"/root/pixel2point/dataset/shapenetcorev2_hdf5_2048/test_files.txt"
    model_path: Path = r"/root/pixel2point/model/2022-09-15_11-32-03_param.pt"


if __name__ == '__main__':
    settings = Testing()
    print(settings.dict())