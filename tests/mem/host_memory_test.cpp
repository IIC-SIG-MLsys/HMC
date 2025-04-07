#include <gtest/gtest.h>
#include <hmc.h>
#include <mem.h>

namespace hmc {

class HostMemoryTest : public ::testing::Test {
protected:
  Memory *host_mem_ops;

  void SetUp() override { host_mem_ops = new Memory(0, MemoryType::CPU); }

  void TearDown() override { delete host_mem_ops; }
};

// 测试allocateBuffer和freeBuffer方法的正常路径
TEST_F(HostMemoryTest, AllocateAndFreeBuffer) {
  void *addr;
  size_t size = 1024;
  EXPECT_EQ(host_mem_ops->allocateBuffer(&addr, size), status_t::SUCCESS);
  ASSERT_NE(addr, nullptr); // 确保分配成功
  EXPECT_EQ(host_mem_ops->freeBuffer(addr), status_t::SUCCESS);
}

// 测试freeBuffer方法的异常路径（传入nullptr）
TEST_F(HostMemoryTest, FreeBuffer_Nullptr) {
  void *addr = nullptr;
  EXPECT_EQ(host_mem_ops->freeBuffer(addr), status_t::ERROR);
}

// 测试copyHostToDevice方法的正常路径
TEST_F(HostMemoryTest, CopyHostToDevice_HappyPath) {
  char source[] = "Hello, World!";
  char destination[20] = {0};
  EXPECT_EQ(host_mem_ops->copyHostToDevice(destination, source, sizeof(source)),
            status_t::SUCCESS);
  EXPECT_STREQ(destination, "Hello, World!");
}

// 测试copyHostToDevice方法的异常路径（传入nullptr）
TEST_F(HostMemoryTest, CopyHostToDevice_Nullptr) {
  char source[] = "Hello, World!";
  EXPECT_EQ(host_mem_ops->copyHostToDevice(nullptr, source, sizeof(source)),
            status_t::ERROR);
  char destination[20] = {0};
  EXPECT_EQ(
      host_mem_ops->copyHostToDevice(destination, nullptr, sizeof(source)),
      status_t::ERROR);
}

// 测试copyDeviceToHost方法的正常路径
TEST_F(HostMemoryTest, CopyDeviceToHost_HappyPath) {
  char source[] = "Hello, World!";
  char destination[20] = {0};
  EXPECT_EQ(host_mem_ops->copyDeviceToHost(destination, source, sizeof(source)),
            status_t::SUCCESS);
  EXPECT_STREQ(destination, "Hello, World!");
}

// 测试copyDeviceToHost方法的异常路径（传入nullptr）
TEST_F(HostMemoryTest, CopyDeviceToHost_Nullptr) {
  char source[] = "Hello, World!";
  EXPECT_EQ(host_mem_ops->copyDeviceToHost(nullptr, source, sizeof(source)),
            status_t::ERROR);
  char destination[20] = {0};
  EXPECT_EQ(
      host_mem_ops->copyDeviceToHost(destination, nullptr, sizeof(source)),
      status_t::ERROR);
}

// 测试copyDeviceToDevice方法的正常路径
TEST_F(HostMemoryTest, CopyDeviceToDevice_HappyPath) {
  char source[] = "Hello, World!";
  char destination[20] = {0};
  EXPECT_EQ(
      host_mem_ops->copyDeviceToDevice(destination, source, sizeof(source)),
      status_t::SUCCESS);
  EXPECT_STREQ(destination, "Hello, World!");
}

// 测试copyDeviceToDevice方法的异常路径（传入nullptr）
TEST_F(HostMemoryTest, CopyDeviceToDevice_Nullptr) {
  char source[] = "Hello, World!";
  EXPECT_EQ(host_mem_ops->copyDeviceToDevice(nullptr, source, sizeof(source)),
            status_t::ERROR);
  char destination[20] = {0};
  EXPECT_EQ(
      host_mem_ops->copyDeviceToDevice(destination, nullptr, sizeof(source)),
      status_t::ERROR);
}

} // namespace hmc