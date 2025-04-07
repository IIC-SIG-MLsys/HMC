#include <gtest/gtest.h>
#include <hmc.h>
#include <mem.h>

namespace hmc {

class DeviceMemoryTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    auto device_mem_ops = new Memory(0);
    if (device_mem_ops->getMemoryType() == MemoryType::CPU) {
      GTEST_SKIP(); // 跳过本测试用例类中的所有测试
    }
    delete device_mem_ops; // 释放资源
  }

  void SetUp() override {
    memory_init_test = new Memory(0);
    device_mem_ops = new Memory(0); // 初始化 device_mem_ops
  }

  void TearDown() override {
    if (memory_init_test) {
      EXPECT_EQ(memory_init_test->free(), status_t::SUCCESS);
      delete memory_init_test;
    }
    if (device_mem_ops) {
      delete device_mem_ops; // 确保清理资源
    }
  }

  Memory *memory_init_test = nullptr;
  Memory *device_mem_ops = nullptr; // 新增成员变量
};

// 测试初始化状态为成功且类型不为CPU
TEST_F(DeviceMemoryTest, InitWithMemoryTypeDevice) {
  EXPECT_EQ(memory_init_test->getInitStatus(), status_t::SUCCESS);
  EXPECT_NE(memory_init_test->getMemoryType(), MemoryType::CPU);
}

// 测试初始化状态为成功且类型为CPU
TEST_F(DeviceMemoryTest, InitWithMemoryTypeCPU) {
  MemoryType mem_init_type = MemoryType::CPU;
  memory_init_test = new Memory(0, mem_init_type);
  EXPECT_EQ(memory_init_test->getInitStatus(), status_t::SUCCESS);
  EXPECT_EQ(memory_init_test->getMemoryType(), MemoryType::CPU);
}

// 测试不支持的内存类型（AMD_GPU）
TEST_F(DeviceMemoryTest, InitWithUnsupportedMemoryType_AMD_GPU) {
  MemoryType mem_init_type = MemoryType::AMD_GPU;
  EXPECT_THROW(new Memory(0, mem_init_type), std::runtime_error);
}

// 测试host_to_buffer和buffer_to_host方法的正常路径
TEST_F(DeviceMemoryTest, CopyHostToBuffer_HappyPath) {
  char source[] = "Hello, World!";
  char *src;
  char des[20] = {0};

  ASSERT_EQ(device_mem_ops->allocateBuffer((void **)&src, 20),
            status_t::SUCCESS);
  ASSERT_EQ(device_mem_ops->copyHostToDevice(src, source, 20),
            status_t::SUCCESS);
  ASSERT_EQ(device_mem_ops->copyDeviceToHost(des, src, 20), status_t::SUCCESS);
  EXPECT_STREQ(des, source);
}

// 测试从空源复制到目标缓冲区
TEST_F(DeviceMemoryTest, CopyBufferToBuffer_NullSource) {
  void *dest;
  size_t bufferSize = 1024;

  ASSERT_EQ(device_mem_ops->allocateBuffer(&dest, bufferSize),
            status_t::SUCCESS);
  EXPECT_EQ(device_mem_ops->copyDeviceToDevice(dest, nullptr, bufferSize),
            status_t::ERROR);
  EXPECT_EQ(device_mem_ops->freeBuffer(dest), status_t::SUCCESS);
}

// 测试设置新的设备ID和内存类型（GPU到CPU）
TEST_F(DeviceMemoryTest, SetNewIdMemoryType_DeviceGPUtoCPU) {
  MemoryType mem_init_type = MemoryType::DEFAULT;
  memory_init_test = new Memory(0, mem_init_type);
  EXPECT_EQ(memory_init_test->getInitStatus(), status_t::SUCCESS);
  EXPECT_EQ(memory_init_test->getDeviceId(), 0);
  EXPECT_NE(memory_init_test->getMemoryType(), MemoryType::CPU);

  memory_init_test->setDeviceIdAndMemoryType(1, MemoryType::CPU);
  EXPECT_EQ(memory_init_test->getDeviceId(), 1);
  EXPECT_EQ(memory_init_test->getMemoryType(), MemoryType::CPU);
}

// 测试设置新的设备ID和不支持的内存类型（NVIDIA GPU到AMD GPU）
TEST_F(DeviceMemoryTest, SetNewIdMemoryType_NvidiaGPUtoAMDGPU_NotSupported) {
  MemoryType mem_init_type = MemoryType::DEFAULT;
  memory_init_test = new Memory(0, mem_init_type);
  EXPECT_EQ(memory_init_test->getInitStatus(), status_t::SUCCESS);
  EXPECT_NE(memory_init_test->getMemoryType(), MemoryType::CPU);

  EXPECT_THROW(
      memory_init_test->setDeviceIdAndMemoryType(1, MemoryType::AMD_GPU),
      std::runtime_error);
}

} // namespace hmc