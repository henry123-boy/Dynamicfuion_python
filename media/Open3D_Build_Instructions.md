# Open3D Build Instructions

## Ubuntu / MacOS ##

### Configuring CMake

From the source/repository root, make & enter a build folder:
```shell
mkdir build && cd build
```
Run CMake with the following arguments:
```shell
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA_MODULE=ON ..
```
Or, if you'd like to install to a custom folder {install_folder}:

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA_MODULE=ON -DCMAKE_INSTALL_PREFIX={install_folder}..
```

**Note:** if you choose to install to a custom folder, you'll later have to provide the location of the `Open3DConfig.cmake` file to the CMake script of the NNRT project, by adding `-DOpen3D_DIR={install folder}/lib/cmake` to CMake arguments.

Alternatively, the `cmake-gui ..`command can be used to graphically configure the parameters. We highly recommend using that with the "Grouped" and "Advanced" options checked if you're new to CMake or are trying to resolve any CMake configuration errors.

### Ubuntu CMake CPPABI_LIBRARY Error ###
If you're on Ubuntu, you might get a configuration error that looks like this:
```
Could not find CPPABI_LIBRARY using the following names: c++abi
```
This is a known issue documented [here](https://github.com/isl-org/Open3D/issues/2559). You can resolve it by running the following in the shell:
```shell
sudo apt install libc++-[YOUR ACTIVE GCC VERSION]-dev libc++abi-[YOUR ACTIVE GCC VERSION]-dev
```

### Building & Installing

You'll want to build the Open3D C++ library _and_ the Python `pip` package. Modify `-j4` to reflect however many cores you would like to use for the build.

Build the C++ library with:
```shell
make -j4
```
Install with:
```shell
make install
```
-or (Ubuntu only)-
```shell
checkinstall --exclude=/home
```
Prepend with `sudo` as needed. `checkinstall` allows you to package the library for easy removal (relevant for system-wide installations), and can be installed using:
```shell
sudo apt install checkinstall
```

Build and install the pip package with:
```shell
cmake --build . --target install_pip_package -- -j4
```

## Windows ##

1) Make a `build` folder inside the downloaded project root.
2) Run the `CMake (GUI)` app: 
   1) Specify the absolute path to the `build` folder
   2) Specify the absolute path to the source folder (downloaded source / checked-out repo root).
   3) Hit `Configure`, choose the Visual Studio version and x64 architecture, and resolve errors (if any) in the configuration.
   4) Check `BUILD_CUDA_MODULE` on.
   5) Hit `Generate` and make sure there are no new errors in the configuration process.
3) Open up the resulting solution file in Visual Studio. Build the `INSTALL` and `install_pip_package` targets.

## Additional Details ##

For additional details, please refer to 
[official Open3D compilation instructions](http://www.open3d.org/docs/release/compilation.html#id3). Note that we are not using virtualenv for any of the packages described here, so if you choose to use `virtualenv`, please make the necessary adjustments to your specific installation process.

