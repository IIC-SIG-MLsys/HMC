/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "hddt.h"
#include "mem.h"

#include "status.h"
#include <functional>
#include <memory>
#include <pybind11/functional.h> // 用于 std::function 支持
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 用于封装内存缓冲区的包装类
struct PyBufferWrapper {
  void *ptr;                           // 内存指针
  size_t size;                         // 分配的字节数
  std::function<void(void *)> deleter; // 当对象析构时调用释放内存

  PyBufferWrapper(void *ptr, size_t size, std::function<void(void *)> deleter)
      : ptr(ptr), size(size), deleter(deleter) {}

  ~PyBufferWrapper() {
    if (ptr && deleter) {
      deleter(ptr);
    }
  }
};

PYBIND11_MODULE(hddt, m) {
  // 内存类型枚举绑定
  py::enum_<hddt::MemoryType>(m, "MemoryType")
      .value("DEFAULT", hddt::MemoryType::DEFAULT)
      .value("CPU", hddt::MemoryType::CPU)
      .value("NVIDIA_GPU", hddt::MemoryType::NVIDIA_GPU)
      .value("AMD_GPU", hddt::MemoryType::AMD_GPU)
      .value("CAMBRICON_MLU", hddt::MemoryType::CAMBRICON_MLU)
      .export_values();

  // Memory 类绑定
  py::class_<hddt::Memory>(m, "Memory")
      .def(py::init<int, hddt::MemoryType>(), py::arg("device_id"),
           py::arg("mem_type") = hddt::MemoryType::DEFAULT)
      .def("init", &hddt::Memory::init)
      .def("free", &hddt::Memory::free)
      .def("createMemoryClass", &hddt::Memory::createMemoryClass)
      .def("copy_host_to_device", &hddt::Memory::copy_host_to_device)
      .def("copy_device_to_host", &hddt::Memory::copy_device_to_host)
      .def("copy_device_to_device", &hddt::Memory::copy_device_to_device)
      // 修改 allocate_buffer：分配内存后封装为 PyBufferWrapper 对象返回
      .def(
          "allocate_buffer",
          [](hddt::Memory &self, size_t size) {
            void *ptr = nullptr;
            auto status = self.allocate_buffer(&ptr, size);
            if (status != hddt::status_t::SUCCESS)
              return py::make_tuple(status, py::none());
            // 注意：此处要求 Memory 对象 self
            // 必须在返回的缓冲区对象生命周期内保持有效
            auto buffer_wrapper = new PyBufferWrapper(
                ptr, size, [&self](void *p) { self.free_buffer(p); });
            return py::make_tuple(status, py::cast(buffer_wrapper));
          },
          py::arg("size"))
      // 同理，修改 allocate_peerable_buffer
      .def(
          "allocate_peerable_buffer",
          [](hddt::Memory &self, size_t size) {
            void *ptr = nullptr;
            auto status = self.allocate_peerable_buffer(&ptr, size);
            if (status != hddt::status_t::SUCCESS)
              return py::make_tuple(status, py::none());
            auto buffer_wrapper = new PyBufferWrapper(
                ptr, size, [&self](void *p) { self.free_buffer(p); });
            return py::make_tuple(status, py::cast(buffer_wrapper));
          },
          py::arg("size"))
      .def("free_buffer", &hddt::Memory::free_buffer)
      .def("set_DeviceId_and_MemoryType",
           &hddt::Memory::set_DeviceId_and_MemoryType)
      .def("get_MemoryType", &hddt::Memory::get_MemoryType)
      .def("get_init_Status", &hddt::Memory::get_init_Status)
      .def("get_DeviceId", &hddt::Memory::get_DeviceId);

  // 状态枚举绑定
  py::enum_<hddt::status_t>(m, "status_t")
      .value("SUCCESS", hddt::status_t::SUCCESS)
      .value("ERROR", hddt::status_t::ERROR)
      .value("UNSUPPORT", hddt::status_t::UNSUPPORT)
      .value("INVALID_CONFIG", hddt::status_t::INVALID_CONFIG)
      .value("NOT_FOUND", hddt::status_t::NOT_FOUND)
      .export_values();

  // 通信类型枚举绑定
  py::enum_<hddt::ConnType>(m, "ConnType")
      .value("RDMA", hddt::ConnType::RDMA)
      .value("UCX", hddt::ConnType::UCX)
      .export_values();

  // ConnBuffer 类绑定
  py::class_<hddt::ConnBuffer, std::shared_ptr<hddt::ConnBuffer>>(m,
                                                                  "ConnBuffer")
      // 构造函数绑定
      .def(py::init<int, size_t, hddt::MemoryType>(), py::arg("device_id"),
           py::arg("buffer_size"),
           py::arg("mem_type") = hddt::MemoryType::DEFAULT,
           "Create ConnBuffer with specified parameters")

      // 暴露只读属性
      // 属性绑定
      .def_property_readonly(
          "ptr",
          [](const hddt::ConnBuffer &self) {
            return reinterpret_cast<uintptr_t>(self.ptr);
          },
          "Buffer pointer address")
      .def_property_readonly(
          "buffer_size",
          [](const hddt::ConnBuffer &self) { return self.buffer_size; },
          "Allocated buffer size in bytes")

      // CPU 数据传输
      .def(
          "writeFromCpu",
          [](hddt::ConnBuffer &self, py::buffer src, size_t size, size_t bias) {
            py::buffer_info info = src.request();
            // 安全检查
            if (size + bias > self.buffer_size) {
              throw std::runtime_error(
                  "Buffer overflow: size + bias exceeds buffer capacity");
            }
            if (info.size * info.itemsize < size) {
              throw std::runtime_error("Source buffer too small");
            }
            return self.writeFromCpu(info.ptr, size, bias);
          },
          py::arg("src"), py::arg("size"), py::arg("bias") = 0,
          "Copy data from CPU memory to buffer\n"
          "Args:\n"
          "    src (buffer): Python buffer object (bytes/bytearray/numpy "
          "array)\n"
          "    size (int): Number of bytes to copy\n"
          "    bias (int): Offset in buffer (default 0)")

      .def(
          "readToCpu",
          [](hddt::ConnBuffer &self, py::buffer dest, size_t size,
             size_t bias) {
            py::buffer_info info = dest.request();

            // 安全检查
            if (size + bias > self.buffer_size) {
              throw std::runtime_error(
                  "Buffer overflow: size + bias exceeds buffer capacity");
            }
            if (info.size * info.itemsize < size) {
              throw std::runtime_error("Destination buffer too small");
            }
            if (info.readonly) {
              throw std::runtime_error(
                  "Destination buffer must be writeable (e.g. use bytearray)");
            }

            return self.readToCpu(info.ptr, size, bias);
          },
          py::arg("dest"), py::arg("size"), py::arg("bias") = 0,
          "Copy data from buffer to CPU memory\n"
          "Args:\n"
          "    dest (buffer): Writeable Python buffer (bytearray/numpy array)\n"
          "    size (int): Number of bytes to copy\n"
          "    bias (int): Offset in buffer (default 0)")

      // GPU 数据传输（假设使用设备指针地址）
      .def(
          "writeFromGpu",
          [](hddt::ConnBuffer &self, uintptr_t src_ptr, size_t size,
             size_t bias) {
            if (size + bias > self.buffer_size) {
              throw std::runtime_error(
                  "Buffer overflow: size + bias exceeds buffer capacity");
            }
            return self.writeFromGpu(reinterpret_cast<void *>(src_ptr), size,
                                     bias);
          },
          py::arg("src_ptr"), py::arg("size"), py::arg("bias") = 0,
          "Copy data from GPU memory to buffer\n"
          "Args:\n"
          "    src_ptr (int): GPU memory address (e.g. from PyCUDA)\n"
          "    size (int): Number of bytes to copy\n"
          "    bias (int): Offset in buffer (default 0)")

      .def(
          "readToGpu",
          [](hddt::ConnBuffer &self, uintptr_t dest_ptr, size_t size,
             size_t bias) {
            if (size + bias > self.buffer_size) {
              throw std::runtime_error(
                  "Buffer overflow: size + bias exceeds buffer capacity");
            }
            return self.readToGpu(reinterpret_cast<void *>(dest_ptr), size,
                                  bias);
          },
          py::arg("dest_ptr"), py::arg("size"), py::arg("bias") = 0,
          "Copy data from buffer to GPU memory\n"
          "Args:\n"
          "    dest_ptr (int): GPU memory address (e.g. from PyCUDA)\n"
          "    size (int): Number of bytes to copy\n"
          "    bias (int): Offset in buffer (default 0)");

  // Communicator 类绑定
  py::class_<hddt::Communicator>(m, "Communicator")
      .def(py::init<std::shared_ptr<hddt::ConnBuffer>>(),
           "Constructor for Communicator", py::arg("buffer"))
      .def("writeTo", &hddt::Communicator::writeTo, "Send data to remote",
           py::arg("node_rank"), py::arg("ptr_bias"), py::arg("size"),
           py::arg("connType"))
      .def("readFrom", &hddt::Communicator::readFrom, "read data from remote",
           py::arg("node_rank"), py::arg("ptr_bias"), py::arg("size"),
           py::arg("connType"))
      .def("connectTo", &hddt::Communicator::connectTo,
           "Connect to a new communicator", py::arg("node_rank"),
           py::arg("connType"))
      .def("initServer", &hddt::Communicator::initServer, "Start a new Server",
           py::arg("ip"), py::arg("node_rank"), py::arg("connType"))
      .def("addNewRankAddr", &hddt::Communicator::addNewRankAddr,
           "Add new rank addr", py::arg("rank"), py::arg("ip"), py::arg("port"))
      .def("delRankAddr", &hddt::Communicator::delRankAddr,
           "Remove a rank addr", py::arg("rank"));

  // 为 PyBufferWrapper 类绑定并实现 buffer 协议，使其可直接转换为 memoryview
  py::class_<PyBufferWrapper>(m, "Buffer", py::buffer_protocol())
      .def_buffer([](PyBufferWrapper &b) -> py::buffer_info {
        return py::buffer_info(
            b.ptr,                 // 内存起始地址
            sizeof(unsigned char), // 单个元素大小
            py::format_descriptor<unsigned char>::format(), // 元素格式（此处以
                                                            // byte 为例）
            1,                      // 维度数
            {b.size},               // 每一维的大小
            {sizeof(unsigned char)} // 每一维的步长
        );
      });

  // 导出内存支持查询函数
  m.def("memory_supported", &hddt::memory_supported,
        "Get the supported memory type");
}

/*
# 如何使用 memory ?
import hddt

# 创建 Memory 对象（例如使用 CPU 内存，device_id 为 0）
memory = hddt.Memory(0, hddt.MemoryType.CPU)

# 分配 1024 字节内存，返回 tuple(status, buffer)
status, buf = memory.allocate_buffer(1024)

if status != hddt.status_t.SUCCESS:
    raise Exception("内存分配失败！")
else:
    # buf 是 hddt.Buffer 类型，支持 buffer 协议，可转换为 memoryview
    mv = memoryview(buf)
    print("分配成功，内存大小：", len(mv))

    # 示例：修改第一个字节
    mv[0] = 42
*/
