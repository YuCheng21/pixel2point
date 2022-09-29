import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from lib.logger import logger, console_logger, file_logger
from lib.dataloader import ShapenetDataset
from lib.model import Pixel2Point
from lib.utils import env_init
from lib.settings import Settings


if __name__ == '__main__':
    console_logger()
    file_logger()

    settings = Settings()
    device = settings.device[0]
    logger.debug(f"Using {device} device")

    worker_init_fn, generator = env_init(settings.reproducibility[0], settings.seed[0])

    preprocess = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(settings.resize),
        transforms.ToTensor()
    ])
    test_dataset = ShapenetDataset(
        dataset_path=settings.test_dataset_path, snapshot_path=settings.snapshot_path,
        transforms=preprocess, only=settings.only,
        mode=settings.mode, remake=settings.dataset_remake
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=settings.batch_size, shuffle=settings.shuffle, 
        num_workers=settings.num_workers, pin_memory=settings.pin_memory,
        worker_init_fn=worker_init_fn, generator=generator
    )

    pixel2point = Pixel2Point().to(device)
    checkpoint = torch.load(settings.model_path)
    pixel2point.load_state_dict(checkpoint['model_state_dict'])
    loss_function = checkpoint['criterion']

    loss_test = 0
    pixel2point.train(mode=False)
    with torch.no_grad():
        test_bar = tqdm(test_loader, unit='batch', leave=True, colour='#9E7EDA')
        for i_batch, (pred, gt, index) in enumerate(test_bar):
            pred = pred.to(device)
            gt = gt.to(device)

            output = pixel2point.forward(pred)
            output = output.type_as(gt).view((gt.shape[0], -1, 3))
            loss, _ = loss_function(output, gt)
            
            loss_test += loss.item()
            test_bar.set_description(f'Testing')
            test_bar.set_postfix(loss=loss.item())
