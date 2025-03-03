apt install protobuf-compiler

protoc --python_out=. hddt.proto
protoc --cpp_out=. hddt.proto
