import os
import random
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def set_seed(seed):
    # URL: https://pytorch.org/docs/stable/notes/randomness.html
    # PyTorch random number generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Python
    random.seed(seed)
    # Random number generators in other libraries
    np.random.seed(seed)
    # ensures that CUDA selects the same algorithm each time an application is run,
    # that algorithm itself may be nondeterministic
    torch.backends.cudnn.benchmark = False
    # Avoiding nondeterministic algorithm
    torch.use_deterministic_algorithms(True)
    # CUDA Results reproducibility (CUDA 11.7.1)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def show_3d(data, mode='browser', path='file.html'):
    if torch.is_tensor(data):
        if data.requires_grad:
            data = data.detach()
        if not data.device == 'cpu':
            data = data.to('cpu')
        data = data.numpy()
    fig = go.Figure(
        data=[go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode='markers',
            marker={'size': 2, 'opacity': 0.8, }
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
