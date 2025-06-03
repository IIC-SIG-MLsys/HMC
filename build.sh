rm -rf build
# build
mkdir -p build && cd build
cmake ..  -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC_LIB=ON
make -j
