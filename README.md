# HMC  
Heterogeneous Memories Communication

[中文](README_zh.md)

## How to Build?
> You can automatically build the project by running bash `build.sh`

1. Create a build directory  
```
mkdir build && cd build
```

2. Generate the Makefile  
```
cmake ..
```

Supported parameters:  
```
-DBUILD_STATIC_LIB=ON # Enable static library compilation
-DBUILD_PYTHON_MOD=ON # Enable python interface
```

3. Build  
```
make
```

## Environment Dependencies
1. Compute libraries and drivers: CUDA/DTK/CNRT, etc.
2. Glog  
    - `sudo apt-get install libgoogle-glog-dev`
3. Gtest if build tests
    - `sudo apt-get install libgtest-dev`

```
HIP clang cmath error:
sudo apt install libstdc++-12-devs g++-12
```

## Build Python Package  
The project packages its core functionalities for Python using pybind11.

To build the Python package, you first need to pull the pybind11 library to support module building:  
- In the project root directory, execute `git submodule update --init --recursive`

Then, enable Python module support and rebuild the project:  
- Use the cmake command with the `-DBUILD_PYTHON_MOD=ON` flag, and rebuild the entire project following the previous steps.
- The Python environment must have the build library installed: `pip install build`
- Build the wheel package: `python -m build`
- Install the wheel package: `pip install dist/xxx.whl`