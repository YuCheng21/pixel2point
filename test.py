import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from logger import logger, console_logger, file_logger
from dataloader import ShapenetDataset
from model import Pixel2Point
from utils import show_3d, show_img

test_dataset_path = r"/root/pixel2point/dataset/shapenetcorev2_hdf5_2048/test_files.txt"
snapshot_path = r"/root/pixel2point/dataset/image"
model_path = r"/root/pixel2point/model/2022-09-15_11-32-03_param.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.debug('==================================')
logger.debug(f"Using {device} device")
logger.debug('==================================')


if __name__ == '__main__':
    console_logger()
    file_logger()

    hyper_param_batch = 32  # 32

    transforms_test = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    test_dataset = ShapenetDataset(
        dataset_path=test_dataset_path,
        snapshot_path=snapshot_path,
        transforms=transforms_test,
        only=['chair'],
        mode='easy'
    )
    test_loader = DataLoader(test_dataset, batch_size=hyper_param_batch, shuffle=True)
    logger.debug('==================================')
    logger.debug('DataLoader OK')
    pixel2point = Pixel2Point().to(device)
    checkpoint = torch.load(model_path)
    pixel2point.load_state_dict(checkpoint['model_state_dict'])
    logger.debug(summary(pixel2point))
    logger.debug('Model OK')
    logger.debug('==================================')
    logger.debug('Start Testing')

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    outputs_path = Path(f"./outputs/test/{current_time}")
    outputs_path.mkdir(parents=True, exist_ok=True)

    # Test the model
    pixel2point.eval()
    with torch.no_grad():
        losses = []
        pbar = tqdm(test_loader, unit='batch', leave=True)
        for i_batch, (pred, gt) in enumerate(pbar):
            pred = pred.to(device)
            gt = gt.to(device)

            outputs = pixel2point(pred)
            outputs = outputs.view((gt.shape[0], -1, 3)).type(torch.float64)
            loss, _ = chamfer_distance(outputs, gt)
            losses.append(loss.item())

            pbar.set_description(f'Testing')
            pbar.set_postfix(loss=loss.item())

            if i_batch % 500 == 0:
                show_img(pred[0], mode='file', path=outputs_path.joinpath(f'pred_{i_batch}.html'))
                show_3d(outputs[0], mode='file', path=outputs_path.joinpath(f'outputs_{i_batch}.html'))
                show_3d(gt[0], mode='file', path=outputs_path.joinpath(f'gt_{i_batch}.html'))

    logger.info(f'Loss {np.mean(losses)}')
    with open(outputs_path.joinpath('info.txt'), 'w') as f:
        f.write(f'model path: {model_path}\n')
        f.write(f'mean loss: {np.mean(losses)}\n')
