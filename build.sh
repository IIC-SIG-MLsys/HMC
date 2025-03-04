rm -rf build
cd src/utils/protobuf/ && rm hddt.pb.cc && rm hddt.pb.h && protoc --cpp_out=. hddt.proto && cd -
mkdir -p build && cd build
cmake ..
make -j16