import torch
import plotly.graph_objects as go
import plotly.express as px


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
        layout=go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
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
