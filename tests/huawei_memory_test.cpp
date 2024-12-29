#include <gtest/gtest.h>
#include <hddt.h>
#include <mem.h>

namespace hddt {

class HuaweiMemoryTest : public ::testing::Test {
protected:
  static HuaweiMemory *huawei_memory;

  static void SetUpTestSuite() {
    huawei_memory = new HuaweiMemory(1, MemoryType::HUAWEI_ASCEND_NPU);
    huawei_memory->init(); // 初始化 Huawei 内存
  }

  static void TearDownTestSuite() {
    huawei_memory->free(); // 释放资源
    delete huawei_memory;
    huawei_memory = nullptr;
  }
};

HuaweiMemory *HuaweiMemoryTest::huawei_memory = nullptr;

// 正常路径：测试 allocate_buffer 和 free_buffer
TEST_F(HuaweiMemoryTest, AllocateAndFreeBuffer_HappyPath) {
  void *addr = nullptr;
  size_t size = 1024;

  // 测试分配
  EXPECT_EQ(huawei_memory->allocate_buffer(&addr, size), status_t::SUCCESS);
  EXPECT_NE(addr, nullptr);

  // 测试释放
  EXPECT_EQ(huawei_memory->free_buffer(addr), status_t::SUCCESS);
}

// 异常路径：free_buffer 时传入空指针
TEST_F(HuaweiMemoryTest, FreeBuffer_Nullptr) {
  void *addr = nullptr;
  EXPECT_EQ(huawei_memory->free_buffer(addr), status_t::ERROR);
}

// 正常路径：测试 copy_host_to_device 和 copy_device_to_host
TEST_F(HuaweiMemoryTest, CopyHostToBuffer_And_BufferToHost_HappyPath) {
  char source[] = "Test Data";
  char *device_buffer = nullptr;
  size_t size = sizeof(source);

  // 分配设备缓冲区
  EXPECT_EQ(huawei_memory->allocate_buffer((void **)&device_buffer, size),
            status_t::SUCCESS);

  // 从主机到设备
  EXPECT_EQ(huawei_memory->copy_host_to_device(device_buffer, source, size),
            status_t::SUCCESS);

  // 从设备到主机
  char destination[size];
  EXPECT_EQ(
      huawei_memory->copy_device_to_host(destination, device_buffer, size),
      status_t::SUCCESS);
  EXPECT_STREQ(destination, source);

  // 释放设备缓冲区
  huawei_memory->free_buffer(device_buffer);
}

// 异常路径：copy_host_to_device 时目标指针为空
TEST_F(HuaweiMemoryTest, CopyHostToBuffer_NullDest) {
  char source[] = "Test Data";
  EXPECT_EQ(huawei_memory->copy_host_to_device(nullptr, source, sizeof(source)),
            status_t::ERROR);
}

// 异常路径：copy_device_to_host 时源指针为空
TEST_F(HuaweiMemoryTest, CopyBufferToHost_NullSrc) {
  char destination[20] = {0};
  EXPECT_EQ(huawei_memory->copy_device_to_host(destination, nullptr,
                                               sizeof(destination)),
            status_t::ERROR);
}

// 正常路径：测试 copy_device_to_device
TEST_F(HuaweiMemoryTest, CopyDeviceToDevice_HappyPath) {
  size_t bufferSize = 1024;
  void *src = nullptr;
  void *dest = nullptr;

  // 分配设备缓冲区
  EXPECT_EQ(huawei_memory->allocate_buffer(&src, bufferSize),
            status_t::SUCCESS);
  EXPECT_EQ(huawei_memory->allocate_buffer(&dest, bufferSize),
            status_t::SUCCESS);

  // 模拟数据并拷贝
  char sourceData[bufferSize] = {0};
  for (size_t i = 0; i < bufferSize; ++i) {
    sourceData[i] = static_cast<char>(i % 256);
  }

  EXPECT_EQ(huawei_memory->copy_host_to_device(src, sourceData, bufferSize),
            status_t::SUCCESS);
  EXPECT_EQ(huawei_memory->copy_device_to_device(dest, src, bufferSize),
            status_t::SUCCESS);

  // 释放设备缓冲区
  huawei_memory->free_buffer(src);
  huawei_memory->free_buffer(dest);
}

// 异常路径：copy_device_to_device 时目标指针为空
TEST_F(HuaweiMemoryTest, CopyDeviceToDevice_NullDest) {
  void *src = nullptr;
  size_t bufferSize = 1024;

  EXPECT_EQ(huawei_memory->allocate_buffer(&src, bufferSize),
            status_t::SUCCESS);
  EXPECT_EQ(huawei_memory->copy_device_to_device(nullptr, src, bufferSize),
            status_t::ERROR);

  huawei_memory->free_buffer(src);
}

// 异常路径：copy_device_to_device 时源指针为空
TEST_F(HuaweiMemoryTest, CopyDeviceToDevice_NullSrc) {
  void *dest = nullptr;
  size_t bufferSize = 1024;

  EXPECT_EQ(huawei_memory->allocate_buffer(&dest, bufferSize),
            status_t::SUCCESS);
  EXPECT_EQ(huawei_memory->copy_device_to_device(dest, nullptr, bufferSize),
            status_t::ERROR);

  huawei_memory->free_buffer(dest);
}

} // namespace hddt
