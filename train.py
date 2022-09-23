import numpy as np
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

from dataloader import ShapenetDataset
from logger import logger, console_logger, file_logger
from model import Pixel2Point
from settings import Training
from utils import show_3d, save_result, set_seed, seed_worker
from loss import chamfer_distance


def run():
    settings = Training()
    if settings.reproducibility is True:
        set_seed(settings.seed)

    device = settings.device
    logger.debug('==================================')
    logger.debug(f"Using {device} device")

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
    logger.debug('DataLoader OK')

    # Prepare the model
    pixel2point = Pixel2Point().to(device)
    optimizer = torch.optim.Adam(pixel2point.parameters(), lr=settings.learning_rate)
    logger.debug(summary(pixel2point, verbose=0))
    logger.debug('Model OK')
    logger.debug('==================================')
    logger.debug('Start Training')

    settings.output_path = settings.output_path.parent.joinpath(f'{settings.output_path.name}_{settings.only}')
    train_path = settings.output_path.joinpath('train')
    train_path.mkdir(parents=True, exist_ok=True)

    sample_path = train_path.joinpath('sample')
    sample_path.mkdir(parents=True, exist_ok=True)
    sample = None

    # Train the model
    e_losses = []
    for e in range(settings.epoch):
        pixel2point.train()
        losses = []
        pbar = tqdm(train_loader, unit='batch', leave=True)
        for i_batch, (pred, gt, index) in enumerate(pbar):
            pred = pred.to(device=device)
            gt = gt.to(device=device)

            # Forward pass
            output = pixel2point.forward(pred)
            output = output.type_as(gt).view(gt.shape[0], -1, 3)
            loss, _ = chamfer_distance(output, gt)
            losses.append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()

            pbar.set_description(f'Epoch [{e + 1}/{settings.epoch}]')
            pbar.set_postfix(loss=loss.item())

            # Save the output result
            if settings.save_result is True:
                # Watch the first object of the dataset change at each epoch
                if i_batch == 0 and e == 0:
                    sample = index[0]
                    save_result(pred[0], output[0], gt[0], sample_path, f"{e}_{i_batch}")
                else:
                    if sample in index:
                        show_3d(output[index.tolist().index(sample)],
                                mode='file', path=sample_path.joinpath(f'output_{e}_{i_batch}.html'))
                # Each epoch save 4 result
                if i_batch % np.floor(train_dataset.length / settings.batch_size / 4).astype(int) == 0:
                    save_result(pred[0], output[0], gt[0], train_path, f"{e}_{i_batch}")
        e_losses.append(np.mean(losses))

    # Save model
    if settings.save_model is True:
        model_path = settings.output_path.joinpath('model')
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': settings.epoch,
            'model_state_dict': pixel2point.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': chamfer_distance,
        }, model_path.joinpath(f"{settings.only}_param.pt"))

    # Save loss value
    fig_loss = go.Figure(data=go.Scatter(x=e_losses, y=np.arange(len(e_losses))))
    fig_loss.write_html(train_path.joinpath(f"loss.html"))

    # Save training information
    with open(train_path.joinpath('info.txt'), 'w') as f:
        f.write(f'only: {train_dataset.only}\n')
        f.write(f'mode: {train_dataset.mode}\n')
        f.write(f'transforms: {train_dataset.transforms}\n')

    return settings.output_path, settings.only


if __name__ == '__main__':
    console_logger()
    file_logger()

    output_path, only = run()
