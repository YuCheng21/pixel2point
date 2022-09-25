import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

from writer import summary_writer, profile, mesh_dict, text_string, d32rgb, d42rgb
from dataloader import ShapenetDataset
from logger import logger, console_logger, file_logger
from model import Pixel2Point
from settings import Training
from utils import set_seed, seed_worker
from loss import chamfer_distance


if __name__ == '__main__':
    console_logger()
    file_logger()
    settings = Training()
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

    writer = summary_writer(comment='train')
    prof = profile(dir_name=writer.logdir)

    transforms_train = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(settings.resize),
        # transforms.RandomRotation(10.),
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5)
    ])
    transforms_val = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(settings.resize),
        transforms.ToTensor()
    ])
    train_dataset = ShapenetDataset(
        dataset_path=settings.train_dataset_path,
        snapshot_path=settings.snapshot_path,
        transforms=transforms_train,
        only=settings.only,
        mode=settings.mode
    )
    val_dataset = ShapenetDataset(
        dataset_path=settings.val_dataset_path,
        snapshot_path=settings.snapshot_path,
        transforms=transforms_val,
        only=settings.only,
        mode=settings.mode
    )
    if train_dataset.length == 0:
        raise Exception('no training data')

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=settings.batch_size,
        shuffle=True, num_workers=settings.num_workers,
        worker_init_fn=worker_init_fn, generator=generator
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=settings.batch_size,
        shuffle=True, num_workers=settings.num_workers,
        worker_init_fn=seed_worker, generator=generator
    )

    pixel2point = Pixel2Point(initial_point=settings.initial_point).to(device)
    logger.debug(summary(pixel2point, verbose=0))

    optimizer = torch.optim.Adam(pixel2point.parameters(), lr=settings.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    writer.add_graph(pixel2point, torch.rand([settings.batch_size, 1] + list(settings.resize)).to(device))
    writer.add_text('Setting Parameter', text_string=f"<pre>{text_string(settings, transforms_train)}", global_step=0)

    sample = None
    for i_epoch in range(settings.epoch):
        train_loss = 0
        prof.start()
        pixel2point.train()
        pbar = tqdm(train_loader, unit='batch', leave=True)
        for i_batch, (pred, gt, index) in enumerate(pbar):
            global_step = i_epoch * len(train_loader) + i_batch + 1
            pred = pred.to(device=device)
            gt = gt.to(device=device)

            # Forward propagation
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=settings.use_amp):
                output = pixel2point.forward(pred)
                output = output.type_as(gt).view(gt.shape[0], -1, 3)
                loss, _ = chamfer_distance(output, gt)

            # Backward propagation and optimize
            optimizer.zero_grad()
            # torch.autograd.backward(loss)
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            prof.step()

            # Update progress bar
            pbar.set_description(f'Epoch [{i_epoch + 1}/{settings.epoch}]')
            pbar.set_postfix(loss=loss.item())

            # Logging information
            train_loss += loss.item()
            if i_batch == 0:
                writer.add_images('Input/train', d42rgb(pred), global_step)
                writer.add_mesh(f'Output_{i_epoch}_Train', output,
                                config_dict=mesh_dict(output), global_step=global_step)
                writer.add_mesh('Ground_Truth_Train', gt,
                                config_dict=mesh_dict(gt), global_step=global_step)
                writer.add_histogram('Layer1_Conv/weight', pixel2point.layer1[0].weight, global_step=global_step)
                writer.add_histogram('Layer1_Conv/bais', pixel2point.layer1[0].bias, global_step=global_step)
                writer.add_histogram('Layer2_Conv/weight', pixel2point.layer2[0].weight, global_step=global_step)
                writer.add_histogram('Layer2_Conv/bais', pixel2point.layer2[0].bias, global_step=global_step)
                writer.add_histogram('Layer3_Conv/weight', pixel2point.layer3[0].weight, global_step=global_step)
                writer.add_histogram('Layer3_Conv/bais', pixel2point.layer3[0].bias, global_step=global_step)
                writer.add_histogram('Layer4_Conv/weight', pixel2point.layer4[0].weight, global_step=global_step)
                writer.add_histogram('Layer4_Conv/bais', pixel2point.layer4[0].bias, global_step=global_step)
                writer.add_histogram('Layer5_Conv/weight', pixel2point.layer5[0].weight, global_step=global_step)
                writer.add_histogram('Layer5_Conv/bais', pixel2point.layer5[0].bias, global_step=global_step)
                writer.add_histogram('Layer6_Conv/weight', pixel2point.layer6[0].weight, global_step=global_step)
                writer.add_histogram('Layer6_Conv/bais', pixel2point.layer6[0].bias, global_step=global_step)
                writer.add_histogram('Layer7_Conv/weight', pixel2point.layer7[0].weight, global_step=global_step)
                writer.add_histogram('Layer7_Conv/bais', pixel2point.layer7[0].bias, global_step=global_step)
                writer.add_histogram('fc1/weight', pixel2point.fc1[0].weight, global_step=global_step)
                writer.add_histogram('fc1/bais', pixel2point.fc1[0].bias, global_step=global_step)
                writer.add_histogram('fc2/weight', pixel2point.fc2[0].weight, global_step=global_step)
                writer.add_histogram('fc2/bais', pixel2point.fc2[0].bias, global_step=global_step)
                writer.add_histogram('fc3/weight', pixel2point.fc3[0].weight, global_step=global_step)
                writer.add_histogram('fc3/bais', pixel2point.fc3[0].bias, global_step=global_step)
                writer.add_histogram('fc4/weight', pixel2point.fc4.weight, global_step=global_step)
                writer.add_histogram('fc4/bais', pixel2point.fc4.bias, global_step=global_step)

            if global_step == 1:
                target_index = 0
                sample = index[target_index]
                writer.add_mesh(f'Initial_Point: {settings.initial_point}', pixel2point.initial_point.unsqueeze(0),
                                config_dict=mesh_dict(pixel2point.initial_point), global_step=global_step)

            if sample in index:
                target_index = index.tolist().index(sample)
                writer.add_image('Global_0_Input', d32rgb(pred[target_index]), global_step)
                writer.add_mesh(f'Global_0_Output/{i_epoch}', output[target_index].unsqueeze(0),
                                config_dict=mesh_dict(output), global_step=global_step)
                writer.add_mesh('Global_0_Ground_Truth', gt[target_index].unsqueeze(0),
                                config_dict=mesh_dict(gt), global_step=global_step)

        prof.stop()
        val_loss = 0
        pixel2point.eval()
        with torch.no_grad():
            pbar = tqdm(val_loader, unit='batch', leave=True)
            for i_batch, (pred, gt, index) in enumerate(pbar):
                pred = pred.to(device)
                gt = gt.to(device)

                # Forward propagation
                output = pixel2point.forward(pred)
                output = output.type_as(gt).view((gt.shape[0], -1, 3))
                loss, _ = chamfer_distance(output, gt)
                val_loss += loss.item()

                # Update progress bar
                pbar.set_description(f'Validating')
                pbar.set_postfix(loss=loss.item())

                # Logging information
                if i_batch == 0:
                    writer.add_images('Input/val', d42rgb(pred), global_step)
                    writer.add_mesh(f'Output_{i_epoch}_Val', output,
                                    config_dict=mesh_dict(output), global_step=global_step)
                    writer.add_mesh('Ground_Truth_Val', gt,
                                    config_dict=mesh_dict(gt), global_step=global_step)
        writer.add_hparams(
            {'learning_rate': settings.learning_rate, 'batch_size': settings.batch_size},
            {'Loss/train': train_loss / len(train_loader), 'Loss/val': val_loss / len(val_loader)},
            './',
            global_step=global_step
        )
    writer.close()

    # Save model
    if settings.save_model is True:
        model_path = settings.output_path.joinpath('model')
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': settings.epoch,
            'model_state_dict': pixel2point.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'criterion': chamfer_distance,
        }, model_path.joinpath(f"{settings.only}_param.pt"))
