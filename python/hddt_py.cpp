/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "hddt.h"
#include "mem.h"
#include "status.h"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>  // 用于 std::function 支持
#include <functional>
#include <memory>

namespace py = pybind11;

// 用于封装内存缓冲区的包装类
struct PyBufferWrapper {
    void* ptr;              // 内存指针
    size_t size;            // 分配的字节数
    std::function<void(void*)> deleter; // 当对象析构时调用释放内存

    PyBufferWrapper(void* ptr, size_t size, std::function<void(void*)> deleter)
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
      .def(py::init<int, hddt::MemoryType>(),
           py::arg("device_id"), py::arg("mem_type") = hddt::MemoryType::DEFAULT)
      .def("init", &hddt::Memory::init)
      .def("free", &hddt::Memory::free)
      .def("createMemoryClass", &hddt::Memory::createMemoryClass)
      .def("copy_host_to_device", &hddt::Memory::copy_host_to_device)
      .def("copy_device_to_host", &hddt::Memory::copy_device_to_host)
      .def("copy_device_to_device", &hddt::Memory::copy_device_to_device)
      // 修改 allocate_buffer：分配内存后封装为 PyBufferWrapper 对象返回
      .def("allocate_buffer", [](hddt::Memory &self, size_t size) {
           void* ptr = nullptr;
           auto status = self.allocate_buffer(&ptr, size);
           if (status != hddt::status_t::SUCCESS)
             return py::make_tuple(status, py::none());
           // 注意：此处要求 Memory 对象 self 必须在返回的缓冲区对象生命周期内保持有效
           auto buffer_wrapper = new PyBufferWrapper(ptr, size, [&self](void* p) {
                 self.free_buffer(p);
           });
           return py::make_tuple(status, py::cast(buffer_wrapper));
      }, py::arg("size"))
      // 同理，修改 allocate_peerable_buffer
      .def("allocate_peerable_buffer", [](hddt::Memory &self, size_t size) {
           void* ptr = nullptr;
           auto status = self.allocate_peerable_buffer(&ptr, size);
           if (status != hddt::status_t::SUCCESS)
             return py::make_tuple(status, py::none());
           auto buffer_wrapper = new PyBufferWrapper(ptr, size, [&self](void* p) {
                 self.free_buffer(p);
           });
           return py::make_tuple(status, py::cast(buffer_wrapper));
      }, py::arg("size"))
      .def("free_buffer", &hddt::Memory::free_buffer)
      .def("set_DeviceId_and_MemoryType", &hddt::Memory::set_DeviceId_and_MemoryType)
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
  py::class_<hddt::ConnBuffer>(m, "ConnBuffer")
      .def(py::init<int, size_t, hddt::MemoryType>(),
           "Constructor for ConnBuffer",
           py::arg("device_id"), py::arg("buffer_size"),
           py::arg("mem_type") = hddt::MemoryType::DEFAULT);

  // Communicator 类绑定
  py::class_<hddt::Communicator>(m, "Communicator")
      .def(py::init<std::shared_ptr<hddt::ConnBuffer>>(),
           "Constructor for Communicator",
           py::arg("buffer"))
      .def("sendData", &hddt::Communicator::sendData,
           "Send data to remote",
           py::arg("node_rank"), py::arg("ptr_bias"), py::arg("size"))
      .def("recvData", [](hddt::Communicator &self, uint32_t node_rank) {
           size_t flag = 0;
           auto status = self.recvData(node_rank, &flag);
           return py::make_tuple(status, flag);
      }, "Receive data from remote", py::arg("node_rank"))
      .def("connectTo", &hddt::Communicator::connectTo, "Connect to a new communicator",
           py::arg("node_rank"), py::arg("connType"))
      .def("initServer", &hddt::Communicator::initServer, "Start a new Server",
           py::arg("ip"), py::arg("node_rank"), py::arg("connType"))
      .def("addNewRankAddr", &hddt::Communicator::addNewRankAddr, "Add new rank addr",
           py::arg("rank"), py::arg("ip"), py::arg("port"))
      .def("delRankAddr", &hddt::Communicator::delRankAddr, "Remove a rank addr",
           py::arg("rank"));

  // 为 PyBufferWrapper 类绑定并实现 buffer 协议，使其可直接转换为 memoryview
  py::class_<PyBufferWrapper>(m, "Buffer", py::buffer_protocol())
      .def_buffer([](PyBufferWrapper &b) -> py::buffer_info {
           return py::buffer_info(
               b.ptr,                              // 内存起始地址
               sizeof(unsigned char),              // 单个元素大小
               py::format_descriptor<unsigned char>::format(), // 元素格式（此处以 byte 为例）
               1,                                  // 维度数
               { b.size },                         // 每一维的大小
               { sizeof(unsigned char) }           // 每一维的步长
           );
      });
      
  // 导出内存支持查询函数
  m.def("memory_supported", &hddt::memory_supported, "Get the supported memory type");
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