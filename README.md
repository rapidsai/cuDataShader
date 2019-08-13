# cuDatashader

cuDatashader is a port of Datashader to CUDA technology allowing GPU acceleration of the rendering. It allows faster rendering of complex visualizations containing billions of points. It can also be used for realtime applications or as a fast high resolution video renderer.

cuDatashader is still in its infancy and might contain some bugs. Only some of Datashader's features are currently implemented.

## Implemented features

- Core : Canvas taking points or lines
- Aggregations/Reductions : Any, Count, Max, Sum
- Transfer Functions : Shade, Spread, Dynspread, Stack
- Edge Bundling : connect_edges and fdeb_bundle for GPU FDEB Edge Bundling

## How to use it?

Please refer to the documentation of Datashader : [http://datashader.org/user_guide/index.html](http://datashader.org/user_guide/index.html). The cuDatashader features work the same way as their original counterparts except that the functions take cuDF DataFrames instead of Pandas DataFrames.

## Installation

If you want to install it in an isolated environment:
```sh
$ conda create -n test_env
$ source activate test_env
```

Install with pip:
```sh
# While in the root folder
$ pip install -e . # For default Python
$ pip3 install -e . # For Python3
```

Or directly from Python:
```sh
# While in the root folder
$ python setup.py install # For default Python
$ python3 setup.py install # For Python3
```