/**
 * @copyright
 * Copyright (c) 2025, SDU spgroup Holding Limited
 * All rights reserved.
 */
#include "hmc.h"
#include "mem.h"
#include "status.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct PyBufferWrapper {
  void *ptr;
  size_t size;
  std::function<void(void *)> deleter;

  PyBufferWrapper(void *ptr, size_t size, std::function<void(void *)> deleter)
      : ptr(ptr), size(size), deleter(std::move(deleter)) {}

  ~PyBufferWrapper() {
    if (ptr && deleter) deleter(ptr);
  }
};

PYBIND11_MODULE(hmc, m) {
  py::enum_<hmc::MemoryType>(m, "MemoryType")
      .value("DEFAULT", hmc::MemoryType::DEFAULT)
      .value("CPU", hmc::MemoryType::CPU)
      .value("NVIDIA_GPU", hmc::MemoryType::NVIDIA_GPU)
      .value("AMD_GPU", hmc::MemoryType::AMD_GPU)
      .value("CAMBRICON_MLU", hmc::MemoryType::CAMBRICON_MLU)
      .value("MOORE_GPU", hmc::MemoryType::MOORE_GPU)
      .export_values();

  py::class_<hmc::Memory>(m, "Memory")
      .def(py::init<int, hmc::MemoryType>(), py::arg("device_id"),
           py::arg("mem_type") = hmc::MemoryType::DEFAULT)
      .def("init", &hmc::Memory::init)
      .def("free", &hmc::Memory::free)
      .def("createMemoryClass", &hmc::Memory::createMemoryClass)
      .def("copyHostToDevice", &hmc::Memory::copyHostToDevice)
      .def("copyDeviceToHost", &hmc::Memory::copyDeviceToHost)
      .def("copyDeviceToDevice", &hmc::Memory::copyDeviceToDevice)
      .def(
          "allocateBuffer",
          [](hmc::Memory &self, size_t size) {
            void *ptr = nullptr;
            auto status = self.allocateBuffer(&ptr, size);
            if (status != hmc::status_t::SUCCESS)
              return py::make_tuple(status, py::none());
            auto buffer_wrapper = new PyBufferWrapper(
                ptr, size, [&self](void *p) { self.freeBuffer(p); });
            return py::make_tuple(status, py::cast(buffer_wrapper));
          },
          py::arg("size"))
      .def(
          "allocatePeerableBuffer",
          [](hmc::Memory &self, size_t size) {
            void *ptr = nullptr;
            auto status = self.allocatePeerableBuffer(&ptr, size);
            if (status != hmc::status_t::SUCCESS)
              return py::make_tuple(status, py::none());
            auto buffer_wrapper = new PyBufferWrapper(
                ptr, size, [&self](void *p) { self.freeBuffer(p); });
            return py::make_tuple(status, py::cast(buffer_wrapper));
          },
          py::arg("size"))
      .def("freeBuffer", &hmc::Memory::freeBuffer)
      .def("setDeviceIdAndMemoryType", &hmc::Memory::setDeviceIdAndMemoryType)
      .def("getMemoryType", &hmc::Memory::getMemoryType)
      .def("getInitStatus", &hmc::Memory::getInitStatus)
      .def("getDeviceId", &hmc::Memory::getDeviceId);

  py::enum_<hmc::status_t>(m, "status_t")
      .value("SUCCESS", hmc::status_t::SUCCESS)
      .value("ERROR", hmc::status_t::ERROR)
      .value("UNSUPPORT", hmc::status_t::UNSUPPORT)
      .value("INVALID_CONFIG", hmc::status_t::INVALID_CONFIG)
      .value("NOT_FOUND", hmc::status_t::NOT_FOUND)
      .value("TIMEOUT", hmc::status_t::TIMEOUT)
      .export_values();

  py::enum_<hmc::ConnType>(m, "ConnType")
      .value("RDMA", hmc::ConnType::RDMA)
      .value("UCX", hmc::ConnType::UCX)
      .export_values();

  py::class_<hmc::ConnBuffer, std::shared_ptr<hmc::ConnBuffer>>(m, "ConnBuffer")
      .def(py::init<int, size_t, hmc::MemoryType>(), py::arg("device_id"),
           py::arg("buffer_size"),
           py::arg("mem_type") = hmc::MemoryType::DEFAULT)
      .def_property_readonly("ptr",
                             [](const hmc::ConnBuffer &self) {
                               return reinterpret_cast<uintptr_t>(self.ptr);
                             })
      .def_property_readonly("buffer_size",
                             [](const hmc::ConnBuffer &self) {
                               return self.buffer_size;
                             })

      .def(
          "writeFromCpu",
          [](hmc::ConnBuffer &self, py::buffer src, size_t size, size_t bias) {
            py::buffer_info info = src.request();
            if (size + bias > self.buffer_size)
              throw std::runtime_error("Buffer overflow");
            if (static_cast<size_t>(info.size * info.itemsize) < size)
              throw std::runtime_error("Source buffer too small");
            return self.writeFromCpu(info.ptr, size, bias);
          },
          py::arg("src"), py::arg("size"), py::arg("bias") = 0)

      .def(
          "readToCpu",
          [](hmc::ConnBuffer &self, py::buffer dest, size_t size, size_t bias) {
            py::buffer_info info = dest.request();
            if (size + bias > self.buffer_size)
              throw std::runtime_error("Buffer overflow");
            if (static_cast<size_t>(info.size * info.itemsize) < size)
              throw std::runtime_error("Destination buffer too small");
            if (info.readonly)
              throw std::runtime_error("Destination buffer must be writeable");
            return self.readToCpu(info.ptr, size, bias);
          },
          py::arg("dest"), py::arg("size"), py::arg("bias") = 0)

      .def(
          "writeFromGpu",
          [](hmc::ConnBuffer &self, uintptr_t src_ptr, size_t size, size_t bias) {
            if (size + bias > self.buffer_size)
              throw std::runtime_error("Buffer overflow");
            return self.writeFromGpu(reinterpret_cast<void *>(src_ptr), size,
                                     bias);
          },
          py::arg("src_ptr"), py::arg("size"), py::arg("bias") = 0)

      .def(
          "readToGpu",
          [](hmc::ConnBuffer &self, uintptr_t dest_ptr, size_t size,
             size_t bias) {
            if (size + bias > self.buffer_size)
              throw std::runtime_error("Buffer overflow");
            return self.readToGpu(reinterpret_cast<void *>(dest_ptr), size,
                                  bias);
          },
          py::arg("dest_ptr"), py::arg("size"), py::arg("bias") = 0)

      .def(
          "copyWithin",
          [](hmc::ConnBuffer &self, size_t dst_bias, size_t src_bias,
             size_t size) {
            if (dst_bias + size > self.buffer_size ||
                src_bias + size > self.buffer_size)
              throw std::runtime_error("Buffer overflow");
            return self.copyWithin(dst_bias, src_bias, size);
          },
          py::arg("dst_bias"), py::arg("src_bias"), py::arg("size"));

  // ---------------- Communicator ctrl API bindings ----------------
  py::enum_<hmc::Communicator::CtrlTransport>(m, "CtrlTransport")
      .value("TCP", hmc::Communicator::CtrlTransport::TCP)
      .value("UDS", hmc::Communicator::CtrlTransport::UDS)
      .export_values();

  py::class_<hmc::Communicator::CtrlLink>(m, "CtrlLink")
      .def(py::init<>())
      .def_readwrite("transport", &hmc::Communicator::CtrlLink::transport)
      .def_readwrite("ip", &hmc::Communicator::CtrlLink::ip)
      .def_readwrite("port", &hmc::Communicator::CtrlLink::port)
      .def_readwrite("uds_path", &hmc::Communicator::CtrlLink::uds_path);

  py::class_<hmc::Communicator, std::shared_ptr<hmc::Communicator>>(
      m, "Communicator")
      .def(py::init<std::shared_ptr<hmc::ConnBuffer>, size_t>(),
           py::arg("buffer"), py::arg("num_chs") = 1)
      .def(py::init([](py::object buf, size_t num_chs) {
             auto sp = buf.cast<std::shared_ptr<hmc::ConnBuffer>>();
             return std::make_shared<hmc::Communicator>(sp, num_chs);
           }),
           py::arg("buffer"), py::arg("num_chs") = 1)

      // ================= core ops (ip + port) =================
      .def("put", &hmc::Communicator::put,
           py::arg("ip"), py::arg("port"),
           py::arg("local_off"), py::arg("remote_off"), py::arg("size"),
           py::arg("connType") = hmc::ConnType::RDMA)

      .def("get", &hmc::Communicator::get,
           py::arg("ip"), py::arg("port"),
           py::arg("local_off"), py::arg("remote_off"), py::arg("size"),
           py::arg("connType") = hmc::ConnType::RDMA)

      // NB ops: return (status, wr_id) to Python
      .def("putNB",
           [](hmc::Communicator &self,
              const std::string &ip, uint16_t port,
              size_t local_off, size_t remote_off, size_t size,
              hmc::ConnType connType) {
             uint64_t id = 0;
             auto st = self.putNB(ip, port, local_off, remote_off, size, &id,
                                  connType);
             return py::make_tuple(st, id);
           },
           py::arg("ip"), py::arg("port"),
           py::arg("local_off"), py::arg("remote_off"), py::arg("size"),
           py::arg("connType") = hmc::ConnType::RDMA)

      .def("getNB",
           [](hmc::Communicator &self,
              const std::string &ip, uint16_t port,
              size_t local_off, size_t remote_off, size_t size,
              hmc::ConnType connType) {
             uint64_t id = 0;
             auto st = self.getNB(ip, port, local_off, remote_off, size, &id,
                                  connType);
             return py::make_tuple(st, id);
           },
           py::arg("ip"), py::arg("port"),
           py::arg("local_off"), py::arg("remote_off"), py::arg("size"),
           py::arg("connType") = hmc::ConnType::RDMA)

      .def("wait", py::overload_cast<uint64_t>(&hmc::Communicator::wait),
           py::arg("wr_id"))
      .def("wait",
           py::overload_cast<const std::vector<uint64_t> &>(
               &hmc::Communicator::wait),
           py::arg("wr_ids"))

      // ================= high-level RDMA-only =================
      // sendDataTo now is (ip, port, send_buf, ...)
      .def("sendDataTo", &hmc::Communicator::sendDataTo,
           py::arg("ip"), py::arg("port"),
           py::arg("send_buf"), py::arg("buf_size"), py::arg("buf_type"),
           py::arg("connType") = hmc::ConnType::RDMA)

      // recvDataFrom remains (ip, recv_buf, ...) per your new class definition
      .def("recvDataFrom", &hmc::Communicator::recvDataFrom,
           py::arg("ip"),
           py::arg("recv_buf"), py::arg("buf_size"), py::arg("buf_type"),
           py::arg("flag"),
           py::arg("connType") = hmc::ConnType::RDMA)

      // ================= ctrl: by rank (CtrlId) =================
      .def("ctrlSend",
           [](hmc::Communicator &self, hmc::Communicator::CtrlId peer,
              uint64_t tag) { return self.ctrlSend(peer, tag); },
           py::arg("peer"), py::arg("tag"))

      .def("ctrlRecv",
           [](hmc::Communicator &self, hmc::Communicator::CtrlId peer) {
             uint64_t tag = 0;
             auto st = self.ctrlRecv(peer, &tag);
             return py::make_tuple(st, tag);
           },
           py::arg("peer"))

      // ================= ctrl server/client mgmt =================
      .def("initCtrlServer", &hmc::Communicator::initCtrlServer,
           py::arg("bind_ip"), py::arg("tcp_port"), py::arg("uds_path") = "")
      .def("closeCtrl", &hmc::Communicator::closeCtrl)
      .def("connectCtrl", &hmc::Communicator::connectCtrl,
           py::arg("peer_id"), py::arg("self_id"), py::arg("link"))
      .def("closeCtrlPeer", &hmc::Communicator::closeCtrlPeer,
           py::arg("peer_id"))
      .def_static("udsPathFor", &hmc::Communicator::udsPathFor,
                  py::arg("dir"), py::arg("peer_id"))

      // ================= connect/init server =================
      .def("initServer", &hmc::Communicator::initServer,
           py::arg("bind_ip"),
           py::arg("data_port"),
           py::arg("ctrl_tcp_port"),
           py::arg("ctrl_uds_path"),
           py::arg("serverType") = hmc::ConnType::RDMA)

      .def("connectTo", &hmc::Communicator::connectTo,
           py::arg("peer_id"),
           py::arg("self_id"),
           py::arg("peer_ip"),
           py::arg("data_port"),
           py::arg("ctrl_link"),
           py::arg("connType") = hmc::ConnType::RDMA)

      .def("closeServer", &hmc::Communicator::closeServer)

      // disconnect/check: now (ip, port, connType)
      .def("disConnect", &hmc::Communicator::disConnect,
           py::arg("ip"), py::arg("port"),
           py::arg("connType") = hmc::ConnType::RDMA)

      .def("checkConn", &hmc::Communicator::checkConn,
           py::arg("ip"), py::arg("port"),
           py::arg("connType") = hmc::ConnType::RDMA);

  py::class_<PyBufferWrapper>(m, "Buffer", py::buffer_protocol())
      .def_buffer([](PyBufferWrapper &b) -> py::buffer_info {
        return py::buffer_info(
            b.ptr,
            sizeof(unsigned char),
            py::format_descriptor<unsigned char>::format(),
            1,
            {b.size},
            {sizeof(unsigned char)});
      });

  m.def("memory_supported", &hmc::memory_supported,
        "Get the supported memory type");
}
