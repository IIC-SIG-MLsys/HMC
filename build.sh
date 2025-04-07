rm -rf build
# build
mkdir -p build && cd build
cmake ..
make -j16