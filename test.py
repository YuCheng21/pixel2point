import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from pathlib import Path

from logger import logger, console_logger, file_logger
from dataloader import ShapenetDataset
from model import Pixel2Point
from utils import save_result, set_seed, seed_worker
from settings import Testing


def run(output_path=None, only=None):
    settings = Testing()
    set_seed(settings.seed)

    if output_path is not None:
        settings.output_path = Path(output_path)
        settings.model_path = settings.output_path.joinpath(f'model/{only}_param.pt')

    device = settings.device
    logger.debug('==================================')
    logger.debug(f"Using {device} device")

    # Prepare the dataset
    transforms_test = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(settings.resize),
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
    test_loader = DataLoader(dataset=test_dataset, batch_size=settings.batch_size, shuffle=True,
                             num_workers=settings.num_workers, worker_init_fn=seed_worker, generator=generator)
    logger.debug('==================================')
    logger.debug('DataLoader OK')

    # Prepare the model
    pixel2point = Pixel2Point().to(device)
    checkpoint = torch.load(settings.model_path)
    pixel2point.load_state_dict(checkpoint['model_state_dict'])
    logger.debug(summary(pixel2point, verbose=0))
    logger.debug('Model OK')
    logger.debug('==================================')
    logger.debug('Start Testing')

    test_path = settings.output_path.joinpath('test')
    test_path.mkdir(parents=True, exist_ok=True)

    # Test the model
    pixel2point.eval()
    with torch.no_grad():
        losses = []
        pbar = tqdm(test_loader, unit='batch', leave=True)
        for i_batch, (pred, gt, index) in enumerate(pbar):
            pred = pred.to(device)
            gt = gt.to(device)

            output = pixel2point(pred)
            output = output.view((gt.shape[0], -1, 3)).type(torch.float64)
            loss, _ = chamfer_distance(output, gt)
            losses.append(loss.item())

            pbar.set_description(f'Testing')
            pbar.set_postfix(loss=loss.item())

            # Save the output result
            # - Each epoch save 4 result
            if i_batch % np.floor(test_dataset.length / settings.batch_size / 4).astype(int) == 0:
                save_result(pred[0], output[0], gt[0], test_path, i_batch)

    # Save testing information
    logger.info(f'Loss {np.mean(losses)}')
    with open(test_path.joinpath('info.txt'), 'w') as f:
        f.write(f'model path: {settings.model_path}\n')
        f.write(f'mean loss: {np.mean(losses)}\n')
        f.write(f'only: {test_dataset.only}\n')
        f.write(f'mode: {test_dataset.mode}\n')
        f.write(f'transforms: {test_dataset.transforms}\n')


if __name__ == '__main__':
    console_logger()
    file_logger()

    run()
