import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm
from itertools import product
from pydantic import create_model
from json import dumps

from lib.writer import summary_writer, profile, mesh_dict, d32rgb, d42rgb
from lib.dataloader import ShapenetDataset
from lib.logger import logger, console_logger, file_logger
from lib.model import Pixel2Point
from lib.settings import Settings
from lib.utils import env_init
from lib.loss import chamfer_distance


class MyProcess():
    def __init__(self):
        self.settings = Settings()
        self.device = self.settings.device[0]
        logger.debug(f"Using {self.device} device")
        self.parameters = self.settings.dict(exclude={
            'snapshot_path', 'output_path', 'model_path',
            'train_dataset_path', 'val_dataset_path', 'test_dataset_path',
        })

    def train_loop(self):
        self.loss_train = 0
        self.pixel2point.train(mode=True)
        self.prof.start()
        train_bar = tqdm(self.loader_train, unit='batch', leave=True, colour='#B8DA7E')
        for i_batch, (self.pred, self.gt, index) in enumerate(train_bar):
            self.global_step = self.i_epoch
            self.pred = self.pred.to(self.device)
            self.gt = self.gt.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.hparam.use_amp):
                self.output = self.pixel2point.forward(self.pred)
                self.output = self.output.type_as(self.gt).view(self.gt.shape[0], -1, 3)
                loss, _ = self.loss_function(self.output, self.gt)

            self.optimizer.zero_grad()
            # torch.autograd.backward(loss)
            self.scaler.scale(loss).backward()
            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.prof.step

            self.loss_train += loss.item()
            train_bar.set_description(f'Epoch [{self.i_epoch}/{self.hparam.epoch}]')
            train_bar.set_postfix(loss=loss.item())
        self.prof.stop()

    def validation_loop(self):
        self.loss_val = 0
        self.pixel2point.train(mode=False)
        with torch.no_grad():
            val_bar = tqdm(self.loader_validation, unit='batch', leave=True, colour='#7EA9DA')
            for i_batch, (self.pred, self.gt, index) in enumerate(val_bar):
                self.pred = self.pred.to(self.device)
                self.gt = self.gt.to(self.device)

                self.output = self.pixel2point.forward(self.pred)
                self.output = self.output.type_as(self.gt).view(self.gt.shape[0], -1, 3)
                loss, _ = self.loss_function(self.output, self.gt)

                self.loss_val += loss.item()
                val_bar.set_description(f'Validating')
                val_bar.set_postfix(loss=loss.item())

    def test_loop(self):
        self.loss_test = 0
        self.pixel2point.train(mode=False)
        with torch.no_grad():
            test_bar = tqdm(self.loader_test, unit='batch', leave=True, colour='#9E7EDA')
            for i_batch, (self.pred, self.gt, index) in enumerate(test_bar):
                self.pred = self.pred.to(self.device)
                self.gt = self.gt.to(self.device)

                self.output = self.pixel2point.forward(self.pred)
                self.output = self.output.type_as(self.gt).view((self.gt.shape[0], -1, 3))
                loss, _ = self.loss_function(self.output, self.gt)

                self.loss_test += loss.item()
                test_bar.set_description(f'Testing')
                test_bar.set_postfix(loss=loss.item())

    def transform_config(self):
        preprocess = []
        if 'grayscale' in self.hparam.preprocess:
            preprocess += [transforms.Grayscale(1)]
        if 'resize' in self.hparam.preprocess:
            preprocess += [transforms.Resize(self.hparam.resize)]
        if 'totensor' in self.hparam.preprocess:
            preprocess += [transforms.ToTensor]
        return transforms.Compose(preprocess)

    def save_model(self, key, data):
        model_path = self.settings.output_path.joinpath('model')
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': self.hparam.epoch,
            'model_state_dict': self.pixel2point.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'criterion': self.loss_function,
        }, model_path.joinpath(f"{key}_param.pt"))
        with open(model_path.joinpath('hparam.txt'), 'a') as f:
            f.write(f'{key}, {data}\n')

    def save_weight(self):
        self.writer.add_histogram('Layer1_Conv/weight', self.pixel2point.layer1[0].weight, global_step=self.global_step)
        self.writer.add_histogram('Layer1_Conv/bais', self.pixel2point.layer1[0].bias, global_step=self.global_step)
        self.writer.add_histogram('Layer2_Conv/weight', self.pixel2point.layer2[0].weight, global_step=self.global_step)
        self.writer.add_histogram('Layer2_Conv/bais', self.pixel2point.layer2[0].bias, global_step=self.global_step)
        self.writer.add_histogram('Layer3_Conv/weight', self.pixel2point.layer3[0].weight, global_step=self.global_step)
        self.writer.add_histogram('Layer3_Conv/bais', self.pixel2point.layer3[0].bias, global_step=self.global_step)
        self.writer.add_histogram('Layer4_Conv/weight', self.pixel2point.layer4[0].weight, global_step=self.global_step)
        self.writer.add_histogram('Layer4_Conv/bais', self.pixel2point.layer4[0].bias, global_step=self.global_step)
        self.writer.add_histogram('Layer5_Conv/weight', self.pixel2point.layer5[0].weight, global_step=self.global_step)
        self.writer.add_histogram('Layer5_Conv/bais', self.pixel2point.layer5[0].bias, global_step=self.global_step)
        self.writer.add_histogram('Layer6_Conv/weight', self.pixel2point.layer6[0].weight, global_step=self.global_step)
        self.writer.add_histogram('Layer6_Conv/bais', self.pixel2point.layer6[0].bias, global_step=self.global_step)
        self.writer.add_histogram('Layer7_Conv/weight', self.pixel2point.layer7[0].weight, global_step=self.global_step)
        self.writer.add_histogram('Layer7_Conv/bais', self.pixel2point.layer7[0].bias, global_step=self.global_step)
        self.writer.add_histogram('fc1/weight', self.pixel2point.fc1[0].weight, global_step=self.global_step)
        self.writer.add_histogram('fc1/bais', self.pixel2point.fc1[0].bias, global_step=self.global_step)
        self.writer.add_histogram('fc2/weight', self.pixel2point.fc2[0].weight, global_step=self.global_step)
        self.writer.add_histogram('fc2/bais', self.pixel2point.fc2[0].bias, global_step=self.global_step)
        self.writer.add_histogram('fc3/weight', self.pixel2point.fc3[0].weight, global_step=self.global_step)
        self.writer.add_histogram('fc3/bais', self.pixel2point.fc3[0].bias, global_step=self.global_step)
        self.writer.add_histogram('fc4/weight', self.pixel2point.fc4.weight, global_step=self.global_step)
        self.writer.add_histogram('fc4/bais', self.pixel2point.fc4.bias, global_step=self.global_step)

    def save_mesh(self, tag, coordinate, global_step):
        self.writer.add_mesh(tag, coordinate, config_dict=mesh_dict(coordinate), global_step=global_step)

    def save_result(self, data_type):
        self.writer.add_images(f'Input/{data_type}', d42rgb(self.pred), self.global_step)
        self.save_mesh(f'Output/{data_type}', self.output, self.global_step)
        self.save_mesh(f'GT/{data_type}', self.gt, self.global_step)

    def run(self):
        for key, data in enumerate(product(*[v for v in self.parameters.values()])):
            self.writer = summary_writer(comment='main')
            self.prof = profile(dir_name=self.writer.logdir)

            hparam_dict = dict(zip(list(self.parameters.keys()), data))
            self.hparam = create_model('HyperParameter', **hparam_dict)()
            logger.debug(f"Hyper Parameter: {dumps(hparam_dict, indent=2)}")
            self.writer.add_text('Hyper Parameter', text_string=f"<pre>{dumps(hparam_dict, indent=2)}", global_step=0)

            self.worker_init_fn, self.generator = env_init(self.hparam.reproducibility, self.hparam.seed)

            self.preprocess = self.transform_config()
            self.dataset_train = ShapenetDataset(
                dataset_path=self.settings.train_dataset_path, snapshot_path=self.settings.snapshot_path,
                transforms=self.preprocess, only=self.hparam.only,
                mode=self.hparam.mode, remake=self.hparam.dataset_remake
            )
            self.dataset_validation = ShapenetDataset(
                dataset_path=self.settings.val_dataset_path, snapshot_path=self.settings.snapshot_path,
                transforms=self.preprocess, only=self.hparam.only,
                mode=self.hparam.mode, remake=self.hparam.dataset_remake
            )
            self.dataset_test = ShapenetDataset(
                dataset_path=self.settings.test_dataset_path, snapshot_path=self.settings.snapshot_path,
                transforms=self.preprocess, only=self.hparam.only,
                mode=self.hparam.mode, remake=self.hparam.dataset_remake
            )
            self.loader_train = DataLoader(
                dataset=self.dataset_train, batch_size=self.hparam.batch_size, shuffle=self.hparam.shuffle,
                num_workers=self.hparam.num_workers, pin_memory=self.hparam.pin_memory,
                worker_init_fn=self.worker_init_fn, generator=self.generator
            )
            self.loader_validation = DataLoader(
                dataset=self.dataset_validation, batch_size=self.hparam.batch_size, shuffle=self.hparam.shuffle,
                num_workers=self.hparam.num_workers, pin_memory=self.hparam.pin_memory,
                worker_init_fn=self.worker_init_fn, generator=self.generator
            )
            self.loader_test = DataLoader(
                dataset=self.dataset_test, batch_size=self.hparam.batch_size, shuffle=self.hparam.shuffle,
                num_workers=self.hparam.num_workers, pin_memory=self.hparam.pin_memory,
                worker_init_fn=self.worker_init_fn, generator=self.generator
            )
            self.pixel2point = Pixel2Point(initial_point=self.hparam.initial_point).to(self.device)
            input_size = [self.hparam.batch_size, 1] + self.hparam.resize
            logger.debug(
                summary(self.pixel2point, input_size, col_names=["input_size", "output_size", "num_params"], verbose=0)
            )
            self.writer.add_graph(self.pixel2point, torch.rand(input_size).to(self.device))
            self.save_mesh('Initial_Point', self.pixel2point.initial_point.unsqueeze(0), global_step=0)

            self.loss_function = chamfer_distance
            self.optimizer = torch.optim.Adam(self.pixel2point.parameters(), lr=self.hparam.learning_rate)
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.hparam.use_amp)

            for self.i_epoch in range(self.hparam.epoch):
                self.train_loop()
                self.validation_loop()
                self.save_result('validation')
                # self.save_weight()

                self.writer.add_hparams(
                    {
                        'learning_rate': self.hparam.learning_rate,
                        'batch_size': self.hparam.batch_size,
                        'initial_point': self.hparam.initial_point
                    },
                    {
                        'Loss/train': self.loss_train / len(self.loader_train),
                        'Loss/validation': self.loss_val / len(self.loader_test)
                    },
                    './',
                    global_step=self.global_step
                )
            if self.hparam.save_model is True:
                self.save_model(key, hparam_dict)

        self.writer.close()


if __name__ == '__main__':
    console_logger()
    file_logger()

    my_process = MyProcess()
    my_process.run()
