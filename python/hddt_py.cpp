#include "hddt.h"
#include "p2p.h"
#include "mem.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(hddt, m) {
  // 对mem的抽象
   py::enum_<hddt::MemoryType>(m, "MemoryType")
        .value("DEFAULT", hddt::MemoryType::DEFAULT)
        .value("CPU", hddt::MemoryType::CPU)
        .value("NVIDIA_GPU", hddt::MemoryType::NVIDIA_GPU)
        .value("AMD_GPU", hddt::MemoryType::AMD_GPU)
        .value("CAMBRICON_MLU", hddt::MemoryType::CAMBRICON_MLU)
        // 可以继续添加更多的值，如果需要的话
        .export_values();

    
  
   py::class_<hddt::Memory>(m, "Memory")
    .def(py::init<int, hddt::MemoryType>(),  // 构造函数
         py::arg("device_id"), py::arg("mem_type") = hddt::MemoryType::DEFAULT)
    .def("init", &hddt::Memory::init)  // 初始化
    .def("free", &hddt::Memory::free)  // 释放
    .def("createMemoryClass", &hddt::Memory::createMemoryClass)  // 创建 MemoryBase 类实例
    .def("copy_host_to_device", &hddt::Memory::copy_host_to_device)  // 主机到设备复制
    .def("copy_device_to_host", &hddt::Memory::copy_device_to_host)  // 设备到主机复制
    .def("copy_device_to_device", &hddt::Memory::copy_device_to_device)  // 设备间复制
    .def("allocate_buffer", 
     [](hddt::Memory &self, py::buffer buf, size_t size) {
         // 通过pybind11的缓冲区管理
         py::buffer_info buf_info = buf.request();
         void* addr = buf_info.ptr;  // 获取缓冲区指针

         // 创建一个指向 void* 的指针，传递给 C++ 方法
         void** addr_ptr = reinterpret_cast<void**>(&addr);

         // 调用C++的allocate_buffer方法
         return self.allocate_buffer(addr_ptr, size);
     },
     py::arg("addr"), py::arg("size"))  // 分配缓冲区
    .def("free_buffer", &hddt::Memory::free_buffer)  // 释放缓冲区
    .def("set_DeviceId_and_MemoryType", &hddt::Memory::set_DeviceId_and_MemoryType)  // 设置设备ID和内存类型
    .def("get_MemoryType", &hddt::Memory::get_MemoryType)  // 获取内存类型
    .def("get_init_Status", &hddt::Memory::get_init_Status)  // 获取初始化状态
    .def("get_DeviceId", &hddt::Memory::get_DeviceId);  // 获取设备ID



  // 对通信接口的抽象
  py::enum_<hddt::status_t>(m, "status_t")
        .value("SUCCESS", hddt::status_t::SUCCESS)
        .value("ERROR", hddt::status_t::ERROR)
        .value("UNSUPPORT", hddt::status_t::UNSUPPORT)
        .export_values();

   py::enum_<hddt::CommunicatorType>(m, "CommunicatorType")
      .value("DEFAULT", hddt::CommunicatorType::DEFAULT)
      .value("RDMA", hddt::CommunicatorType::RDMA)
      .value("TCP", hddt::CommunicatorType::TCP)
      .export_values();  

    py::class_<hddt::Communicator>(m, "Communicator")
        .def("Send", &hddt::Communicator::Send)
        .def("Recv", &hddt::Communicator::Recv)
        .def("Start", &hddt::Communicator::Start)
        .def("Close", &hddt::Communicator::Close);
  

    py::class_<hddt::TCPCommunicator, hddt::Communicator>(m, "TCPCommunicator")
    .def(py::init<hddt::Memory *, bool, bool, std::string, uint16_t, std::string, uint16_t, int, int>(), 
         "Constructor for TCPCommunicator",
         py::arg("mem_op"), py::arg("is_server") = false, 
         py::arg("is_client") = false, py::arg("client_ip") = "",
         py::arg("client_port") = 0, py::arg("server_ip") = "",
         py::arg("server_port") = 0, py::arg("retry_times") = 10,
         py::arg("retry_delay_time") = 1000)
    .def("Send", &hddt::TCPCommunicator::Send, 
         "Send data to remote (TCP)",
         py::arg("input_buffer"), py::arg("send_flags"))  // 修改 Send 方法
    .def("Recv", &hddt::TCPCommunicator::Recv, 
         "Receive data from remote (TCP)",
         py::arg("output_buffer"), py::arg("buffer_size"), py::arg("recv_flags"))  // 修改 Recv 方法
    .def("Start", &hddt::TCPCommunicator::Start, "Start the TCP communicator")
    .def("Close", &hddt::TCPCommunicator::Close, "Close the TCP communicator");
    

    py::class_<hddt::RDMACommunicator, hddt::Communicator>(m, "RDMACommunicator")
    .def(py::init<hddt::Memory *, bool, bool, std::string, uint16_t, std::string, uint16_t, int, int>(),
         "Constructor for RDMACommunicator",
         py::arg("mem_op"),
         py::arg("is_server") = false,
         py::arg("is_client") = false,
         py::arg("client_ip") = "0.0.0.0",  // 默认值
         py::arg("client_port") = RDMA_DEFAULT_PORT,
         py::arg("server_ip") = "0.0.0.0",  // 默认值
         py::arg("server_port") = RDMA_DEFAULT_PORT,
         py::arg("retry_times") = 10,
         py::arg("retry_delay_time") = 1000)
    .def("Send", &hddt::RDMACommunicator::Send,
         "Send data to remote (RDMA)",
         py::arg("input_buffer"), py::arg("send_flags"))  // 参数顺序和数量要匹配
    .def("Recv", &hddt::RDMACommunicator::Recv,
         "Receive data from remote (RDMA)",
         py::arg("output_buffer"), py::arg("buffer_size"), py::arg("recv_flags"))  // 需要传递三个参数
    .def("Start", &hddt::RDMACommunicator::Start, "Start the RDMA communicator")
    .def("Close", &hddt::RDMACommunicator::Close, "Close the RDMA communicator");




  // export function
  // 对mem接口的抽象
  m.def("memory_supported", &hddt::memory_supported, "Get the supported memory type");

  // 对通信接口的抽象
  m.def("CreateCommunicator", &hddt::CreateCommunicator,
          py::arg("mem_op"), 
          py::arg("comm_type") = hddt::CommunicatorType::DEFAULT, 
          py::arg("is_server") = false, 
          py::arg("is_client") = false, 
          py::arg("client_ip") = "", 
          py::arg("client_port") = 0,
          py::arg("server_ip") = "", 
          py::arg("server_port") = 0,
          py::arg("retry_times") = 10, 
          py::arg("retry_delay_time") = 1000,
          "Create a communicator instance");

}