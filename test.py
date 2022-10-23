import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from lib.logger import logger, console_logger, file_logger
from lib.dataloader import ShapenetDataset
from lib.model import Pixel2Point
from lib.utils import env_init, dataloader_init, show_3d, show_result, save_multiple_images
from lib.settings import Settings


if __name__ == '__main__':
    console_logger()
    file_logger()

    settings = Settings()
    device = settings.device[0]
    logger.debug(f"Using {device} device")

    env_init(settings.reproducibility[0], settings.seed[0])
    worker_init_fn, generator = dataloader_init(settings.loader_reproducibility[0], settings.seed[0])

    preprocess = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(settings.resize[0]),
        transforms.ToTensor()
    ])
    test_dataset = ShapenetDataset(
        dataset_path=settings.test_dataset_path, snapshot_path=settings.snapshot_path,
        transforms=preprocess, only=settings.only[0],
        mode=settings.mode[0], remake=settings.dataset_remake[0]
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=settings.batch_size[0], shuffle=settings.shuffle[0],
        num_workers=settings.num_workers[0], pin_memory=settings.pin_memory[0],
        worker_init_fn=worker_init_fn, generator=generator
    )

    pixel2point = Pixel2Point(initial_point=settings.initial_point[0]).to(device)
    checkpoint = torch.load(settings.model_path)
    pixel2point.load_state_dict(checkpoint['model_state_dict'])
    loss_function = checkpoint['criterion']

    plotly_path = settings.output_path.joinpath('plotly')
    plotly_path.mkdir(parents=True, exist_ok=True)
    show_3d(pixel2point.initial_point, path=plotly_path.joinpath('initial_point.html'))

    loss_test = 0
    pixel2point.train(mode=False)
    loss_function.train_param(mode=False)
    with torch.no_grad():
        test_bar = tqdm(test_loader, unit='batch', leave=True, colour='#9E7EDA')
        for i_batch, (pred, gt, index) in enumerate(test_bar):
            pred = pred.to(device)
            gt = gt.to(device)

            output = pixel2point.forward(pred)
            output = output.type_as(gt).view((gt.shape[0], -1, 3))
            loss, _ = loss_function.forward(output, gt)

            loss_test += loss.item()
            test_bar.set_description(f'Testing')
            test_bar.set_postfix(loss=loss.item())

            if i_batch == 5:
                save_multiple_images(pred[:None].permute(0, 2, 3, 1).detach().cpu(), plotly_path.joinpath('imgs.png'))
                watch_index = 48
                show_result(pred[watch_index], output[watch_index], gt[watch_index], plotly_path, watch_index)

    logger.debug(f'Testing Loss: {loss_test / len(test_loader)}')
