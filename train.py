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
from utils import show_3d, show_img

train_dataset_path = r"/root/pixel2point/dataset/shapenetcorev2_hdf5_2048/train_files.txt"
snapshot_path = r"/root/pixel2point/dataset/image"

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.debug('==================================')
logger.debug(f"Using {device} device")
logger.debug('==================================')


if __name__ == '__main__':
    console_logger()
    file_logger()

    hyper_param_batch = 32  # 32
    hyper_param_epoch = 10
    hyper_param_learning_rate = 5e-5

    transforms_train = transforms.Compose([
        transforms.Grayscale(1),  # 0~255 -> 0~1, channel 3/4 -> 1
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10.),
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5)
    ])
    train_dataset = ShapenetDataset(
        dataset_path=train_dataset_path,
        snapshot_path=snapshot_path,
        transforms=transforms_train,
        only=['chair'],
        mode='easy'
    )
    train_loader = DataLoader(train_dataset, batch_size=hyper_param_batch, shuffle=True)
    logger.debug('==================================')
    logger.debug('DataLoader OK')
    pixel2point = Pixel2Point().to(device)
    optimizer = torch.optim.Adam(pixel2point.parameters(), lr=hyper_param_learning_rate)
    logger.debug(summary(pixel2point))
    logger.debug('Model OK')
    logger.debug('==================================')
    logger.debug('Start Training')

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    outputs_path = Path(f"./outputs/train/{current_time}")
    outputs_path.mkdir(parents=True, exist_ok=True)
    e_losses = []
    for e in range(hyper_param_epoch):
        losses = []
        pbar = tqdm(train_loader, unit='batch', leave=True)
        for i_batch, (pred, gt) in enumerate(pbar):
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

            pbar.set_description(f'Epoch [{e + 1}/{hyper_param_epoch}]')
            pbar.set_postfix(loss=loss.item())

            if i_batch % 1500 == 0:
                show_img(pred[0], mode='file', path=outputs_path.joinpath(f'pred_{e}_{i_batch}.html'))
                show_3d(outputs[0], mode='file', path=outputs_path.joinpath(f'outputs_{e}_{i_batch}.html'))
                show_3d(gt[0], mode='file', path=outputs_path.joinpath(f'gt_{e}_{i_batch}.html'))

        e_losses.append(np.mean(losses))

    model_path = Path("./model")
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': hyper_param_epoch,
        'model_state_dict': pixel2point.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': chamfer_distance,
    }, model_path.joinpath(f"{current_time}_param.pt"))

    fig_loss = go.Figure(data=go.Scatter(x=e_losses, y=np.arange(len(e_losses))))
    fig_loss.write_html(outputs_path.joinpath(f"loss.html"))
