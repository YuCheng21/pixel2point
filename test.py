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
from utils import show_3d, show_img, set_seed, seed_worker
from settings import Testing

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug('==================================')
    logger.debug(f"Using {device} device")
    logger.debug('==================================')

    console_logger()
    file_logger()
    settings = Testing()
    set_seed(settings.seed)

    hyper_param_batch = 32  # 32

    # Prepare the dataset
    transforms_test = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    test_dataset = ShapenetDataset(
        dataset_path=settings.test_dataset_path,
        snapshot_path=settings.snapshot_path,
        transforms=transforms_test,
        only=settings.only,
        mode=settings.mode
    )
    generator = torch.Generator()
    generator.manual_seed(settings.seed)
    test_loader = DataLoader(dataset=test_dataset, batch_size=hyper_param_batch, shuffle=True,
                             num_workers=8, worker_init_fn=seed_worker, generator=generator)
    logger.debug('==================================')
    logger.debug(f'Dataset only: {test_dataset.only}')
    logger.debug(f'Dataset mode: {test_dataset.mode}')
    logger.debug(f'Dataset transforms: {test_dataset.transforms}')
    logger.debug('DataLoader OK')

    # Prepare the model
    pixel2point = Pixel2Point().to(device)
    checkpoint = torch.load(settings.model_path)
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
        for i_batch, (pred, gt, index) in enumerate(pbar):
            pred = pred.to(device)
            gt = gt.to(device)

            outputs = pixel2point(pred)
            outputs = outputs.view((gt.shape[0], -1, 3)).type(torch.float64)
            loss, _ = chamfer_distance(outputs, gt)
            losses.append(loss.item())

            pbar.set_description(f'Testing')
            pbar.set_postfix(loss=loss.item())

            # Save the output result
            # - Each epoch save 4 result
            if i_batch % np.floor(test_dataset.length / hyper_param_batch / 4).astype(int) == 0:
                show_img(pred[0], mode='file', path=outputs_path.joinpath(f'pred_{i_batch}.html'))
                show_3d(outputs[0], mode='file', path=outputs_path.joinpath(f'outputs_{i_batch}.html'))
                show_3d(gt[0], mode='file', path=outputs_path.joinpath(f'gt_{i_batch}.html'))

    # Save testing information
    logger.info(f'Loss {np.mean(losses)}')
    with open(outputs_path.joinpath('info.txt'), 'w') as f:
        f.write(f'model path: {settings.model_path}\n')
        f.write(f'mean loss: {np.mean(losses)}\n')
        f.write(f'only: {test_dataset.only}\n')
        f.write(f'mode: {test_dataset.mode}\n')
        f.write(f'transforms: {test_dataset.transforms}\n')
