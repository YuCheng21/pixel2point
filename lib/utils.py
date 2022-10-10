import os
import random
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from math import ceil


def env_init(reproducibility, seed):
    if reproducibility is True:
        # URL: https://pytorch.org/docs/stable/notes/randomness.html
        # PyTorch random number generator
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Python
        random.seed(seed)
        # Random number generators in other libraries
        np.random.seed(seed)
        # ensures that CUDA selects the same algorithm each time an application is run,
        # that algorithm itself may be nondeterministic
        torch.backends.cudnn.benchmark = False
        # Avoiding nondeterministic algorithm
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        # CUDA Results reproducibility (CUDA 11.7.1)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ['PYTHONHASHSEED'] = str(seed)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
    
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True



def dataloader_init(loader_reproducility, seed):
    if loader_reproducility is True:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = seed_worker
    else:
        generator = None
        worker_init_fn = None

    return worker_init_fn, generator


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def show_3d(data, mode='file', path=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"):
    if torch.is_tensor(data):
        if data.requires_grad:
            data = data.detach()
        if not data.device == 'cpu':
            data = data.to('cpu')
        data = data.numpy()
    marker = {'size': 2, 'opacity': 0.8, }
    if data.shape[1] == 4:
        marker['color'] = data[:, 3]
    fig = go.Figure(
        data=[go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode='markers',
            marker=marker
        )],
        layout=go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0}, scene=dict(aspectmode='data'))
    )
    if mode == 'browser':
        fig.show(renderer='browser')
    elif mode == 'file':
        fig.write_html(path)


def show_img(data, mode='browser', path='file.html'):
    if torch.is_tensor(data):
        if not data.device == 'cpu':
            data = data.to('cpu')
        data = data.numpy().squeeze(0)
    img = px.imshow(data, color_continuous_scale='gray')
    if mode == 'browser':
        img.show(renderer='browser')
    elif mode == 'file':
        img.write_html(path)


def show_result(pred, output, gt, save_path, index):
    show_img(pred, mode='file', path=save_path.joinpath(f'pred_{index}.html'))
    show_3d(output, mode='file', path=save_path.joinpath(f'outputs_{index}.html'))
    show_3d(gt, mode='file', path=save_path.joinpath(f'gt_{index}.html'))


def save_multiple_images(images, path, columns=8):
    rows = ceil(len(images) / columns)
    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(2*columns, 3*rows))
    for i, axi in enumerate(ax.flat):
        try:
            target = images[i-1]
        except:
            break
        axi.imshow(target)
        axi.set_title(f'Index:{i}')
    plt.savefig(path)
