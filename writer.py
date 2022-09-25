
import torch
from torchvision import transforms
from tensorboardX import SummaryWriter
from json import dumps


d42rgb = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x)
d32rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)


def summary_writer(comment='main'):
    return SummaryWriter(comment=f'-{comment}')


def profile(dir_name=None):
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=dir_name),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    )


def text_string(*args):
    return dumps({
        **(args[0].dict(exclude={
            'snapshot_path', 'train_dataset_path', 'output_path',
            'test_dataset_path', 'model_path', 'val_dataset_path'
        })),
        **{
            'transforms': str(args[1].transforms)
        },
    }, indent=2)


def mesh_dict(data):
    return {
        'camera': {
            'cls': 'PerspectiveCamera',
            'fov': 50,
            'aspect': 1,
            'near': 0.01,
        },
        'material': {
            'cls': 'PointsMaterial',
            'size': round((torch.max(data).item() - torch.min(data).item()) / 50, 4),
            'depthTest': True,
            'sizeAttenuation': True,
            'transparent': True,
            'opacity': 0.8,
        },
    }
