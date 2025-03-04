rm -rf build
# re generate protobuf files
cd src/utils/protobuf/
rm hddt.pb.cc
rm hddt.pb.h
protoc --cpp_out=. hddt.proto
cd -

# build
mkdir -p build && cd build
cmake ..
make -j16