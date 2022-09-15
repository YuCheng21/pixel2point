A Pytorch implementation of the paper: Pixel2point: 3D Object Reconstruction From a Single Image Using CNN and Initial Sphere

- [Pixel2point: 3D Object Reconstruction From a Single Image Using CNN and Initial Sphere | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9305196)

## Requirement

``` bash
conda env create -f ./environment.yml
conda activate pixel2point
```

## Dataset

- Shapenet Point cloud (shapenetcorev2_hdf5_2048.zip): [antao97/PointCloudDatasets](https://github.com/antao97/PointCloudDatasets)
- Shapenet renderer (image.tar): [Xharlie/ShapenetRender_more_variation](https://github.com/Xharlie/ShapenetRender_more_variation)

> Need to modify the dataset path in the `train.py` and `test.py`

## Training

``` bash
python train.py
```

After starting the training, the output results are in `./outputs/train`

## Testing

```bash
python test.py
```

The test result is in `./outputs/test`