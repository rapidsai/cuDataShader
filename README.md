# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; cuDataShader</div>

The [RAPIDS](https://rapids.ai) cuDataShader - GPU Accelerated

![cuDataShaderGeo](https://github.com/rapidsai/cuDataShader/blob/master/files/cuDataShader.png)

cuDataShader is a port of PyViz's [Datashader](http://datashader.org/) using GPU CUDA technology to enable up to 50x acceleration in visualization render times. Started as an intern project, cuDatashader is still in its infancy and is not feature complete with Datashader. Plans are in the works to actively work with the PyViz group to implement or extend Datashader with further GPU acceleration. cuDataShader is a chart component of the RAPIDS [cuXfilter](https://github.com/rapidsai/cuxfilter) library.


## Implemented features

- Core : Canvas taking points or lines
- Aggregations/Reductions : Any, Count, Max, Sum
- Transfer Functions : Shade, Spread, Dynspread, Stack
- Edge Bundling : connect_edges and fdeb_bundle for GPU FDEB Edge Bundling

## Quick Start

The fastest way to get started is to use cuDataShader as part of the RAPIDS [cuXfilter](https://github.com/rapidsai/cuxfilter) library. 

Otherwise, please refer to the documentation of Datashader [http://datashader.org/user_guide/index.html](http://datashader.org/user_guide/index.html) for usage guides. The cuDatashader features API is built to mirror the original, except that the functions take cuDF DataFrames instead of Pandas DataFrames.

## Install cuDataShader

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

## Contributing Guide

Review the [CONTRIBUTING.md](https://github.com/rapidsai/cuDataShader/blob/master/CONTRIBUTING.md) file for information on how to contribute code and issues to the project.


