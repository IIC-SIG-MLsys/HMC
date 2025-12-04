rm -rf build
# build
mkdir -p build && cd build
cmake ..  -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC_LIB=ON -DBUILD_PYTHON_MOD=ON
make -j
cd -

pip3 install build
python3 -m build
pip install dist/hmc-*.whl --force-reinstall