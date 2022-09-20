import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

from logger import logger, console_logger, file_logger
from dataloader import ShapenetDataset
from model import Pixel2Point
from utils import show_3d, show_img, set_seed, seed_worker
from settings import Training

if __name__ == '__main__':
    console_logger()
    file_logger()
    settings = Training()
    set_seed(settings.seed)

    device = settings.device
    logger.debug('==================================')
    logger.debug(f"Using {device} device")
    logger.debug('==================================')

    # Prepare the dataset
    transforms_train = transforms.Compose([
        transforms.Grayscale(1),  # 0~255 -> 0~1, channel 3/4 -> 1
        transforms.Resize(settings.resize),
        transforms.RandomRotation(10.),
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5)
    ])
    train_dataset = ShapenetDataset(
        dataset_path=settings.train_dataset_path,
        snapshot_path=settings.snapshot_path,
        transforms=transforms_train,
        only=settings.only,
        mode=settings.mode
    )
    if train_dataset.length == 0:
        raise Exception('no training data')
    generator = torch.Generator()
    generator.manual_seed(settings.seed)
    train_loader = DataLoader(dataset=train_dataset, batch_size=settings.batch_size, shuffle=True,
                              num_workers=settings.num_workers, worker_init_fn=seed_worker, generator=generator)
    logger.debug('==================================')
    logger.debug(f'Dataset only: {train_dataset.only}')
    logger.debug(f'Dataset mode: {train_dataset.mode}')
    logger.debug(f'Dataset transforms: {train_dataset.transforms}')
    logger.debug('DataLoader OK')

    # Prepare the model
    pixel2point = Pixel2Point().to(device)
    optimizer = torch.optim.Adam(pixel2point.parameters(), lr=settings.learning_rate)
    logger.debug(summary(pixel2point))
    logger.debug('Model OK')
    logger.debug('==================================')
    logger.debug('Start Training')

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    outputs_path = Path(f"./outputs/train/{current_time}")
    outputs_path.mkdir(parents=True, exist_ok=True)
    sample_path = outputs_path.joinpath('sample')
    sample_path.mkdir(parents=True, exist_ok=True)
    sample = None
    e_losses = []

    # Train the model
    for e in range(settings.epoch):
        losses = []
        pbar = tqdm(train_loader, unit='batch', leave=True)
        for i_batch, (pred, gt, index) in enumerate(pbar):
            pred = pred.to(device)
            gt = gt.to(device)

            # Forward pass
            outputs = pixel2point(pred)
            outputs = outputs.view((gt.shape[0], -1, 3)).type(torch.float64)
            loss, _ = chamfer_distance(outputs, gt)
            losses.append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f'Epoch [{e + 1}/{settings.epoch}]')
            pbar.set_postfix(loss=loss.item())

            # Save the output result
            # - Watch the first object of the dataset change at each epoch
            if i_batch == 0 and e == 0:
                sample = index[0]
                show_img(pred[0], mode='file', path=sample_path.joinpath(f'sample_pred_{e}_{i_batch}.html'))
                show_3d(gt[0], mode='file', path=sample_path.joinpath(f'sample_gt_{e}_{i_batch}.html'))
                show_3d(outputs[0], mode='file', path=sample_path.joinpath(f'sample_outputs_{e}_{i_batch}.html'))
            else:
                if sample in index:
                    show_3d(outputs[index.tolist().index(sample)],
                            mode='file', path=sample_path.joinpath(f'sample_outputs_{e}_{i_batch}.html'))
            # - Each epoch save 4 result
            if i_batch % np.floor(train_dataset.length / settings.batch_size / 4).astype(int) == 0:
                show_img(pred[0], mode='file', path=outputs_path.joinpath(f'pred_{e}_{i_batch}.html'))
                show_3d(outputs[0], mode='file', path=outputs_path.joinpath(f'outputs_{e}_{i_batch}.html'))
                show_3d(gt[0], mode='file', path=outputs_path.joinpath(f'gt_{e}_{i_batch}.html'))
        e_losses.append(np.mean(losses))

    # Save model
    model_path = Path("./model")
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': settings.epoch,
        'model_state_dict': pixel2point.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': chamfer_distance,
    }, model_path.joinpath(f"{current_time}_param.pt"))

    # Save loss value
    fig_loss = go.Figure(data=go.Scatter(x=e_losses, y=np.arange(len(e_losses))))
    fig_loss.write_html(outputs_path.joinpath(f"loss.html"))

    # Save training information
    with open(outputs_path.joinpath('info.txt'), 'w') as f:
        f.write(f'only: {train_dataset.only}\n')
        f.write(f'mode: {train_dataset.mode}\n')
        f.write(f'transforms: {train_dataset.transforms}\n')
