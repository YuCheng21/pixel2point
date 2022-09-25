import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from writer import summary_writer, profile, mesh_dict, text_string, d32rgb, d42rgb

from logger import logger, console_logger, file_logger
from dataloader import ShapenetDataset
from model import Pixel2Point
from utils import set_seed, seed_worker
from settings import Testing


if __name__ == '__main__':
    console_logger()
    file_logger()
    settings = Testing()
    if settings.reproducibility is True:
        set_seed(settings.seed)
        generator = torch.Generator()
        generator.manual_seed(settings.seed)
        worker_init_fn = seed_worker
    else:
        generator = None
        worker_init_fn = None

    device = settings.device
    logger.debug(f"Using {device} device")

    writer = summary_writer(comment='test')
    prof = profile(dir_name=writer.logdir)
    prof.start()

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
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=settings.batch_size,
        shuffle=True, num_workers=settings.num_workers,
        worker_init_fn=seed_worker, generator=generator
    )

    pixel2point = Pixel2Point().to(device)
    checkpoint = torch.load(settings.model_path)
    pixel2point.load_state_dict(checkpoint['model_state_dict'])
    logger.debug(summary(pixel2point, verbose=0))

    pixel2point.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader, unit='batch', leave=True)
        for i_batch, (pred, gt, index) in enumerate(pbar):
            pred = pred.to(device)
            gt = gt.to(device)

            # Forward propagation
            output = pixel2point.forward(pred)
            output = output.type_as(gt).view((gt.shape[0], -1, 3))
            loss, _ = chamfer_distance(output, gt)
            
            # Update progress bar
            pbar.set_description(f'Testing')
            pbar.set_postfix(loss=loss.item())
    
    prof.stop()
    writer.close()
