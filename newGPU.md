# 如何新增GPU支持

## 构建系统
1. 在`cmake/`目录下新增对应动态库、头文件和编译器
    - 例如：nvidia gpu 需要找到 cuda的头文件、动态库和nvcc编译器
    - 将变量导出，供根目录的CmakeLists.txt内使用，例如：FindCuda导出了 CUDA_INCLUDE_DIRS CUDA_LIBRARIES CUDA_NVCC_EXECUTABLE

2. 在根目录 CmakeLists.txt 内
    - include(FindXXX)
    - 增加相应的GPU后端的链接分支：例如：摩尔线程：MUSA_FOUND
        - add_definitions(-DENABLE_MUSA) // 增加一个宏定义
        - include_directories(${MUSA_INCLUDE_DIRS}) // 头文件
        - target_link_libraries(${target_name} PUBLIC ${MUSA_LIBRARIES}) // 链接相应动态库

3. 在 apps/uhm_app/CmakeLists.txt tests/CmakeLists.txt 也加入相应的库头文件链接逻辑。

## 代码修改

1. include/hmc.h 头文件入口，增加相应分支，
```
#ifdef ENABLE_XXX
#include xxx
#endif
```

2. include/mem.h 增加显存类型
```
enum class MemoryType {
  DEFAULT,       ///< Automatically determined by the system
  CPU,           ///< Host system memory
  NVIDIA_GPU,    ///< NVIDIA CUDA GPU memory
  AMD_GPU,       ///< AMD ROCm GPU memory
  CAMBRICON_MLU, ///< Cambricon MLU accelerator memory
  MOORE_GPU      ///< Moore Threads GPU memory
  XXXX_GPU       /// NEW <-
};
```

3. 在src/memories/hddt_memory_cpp, host_memory.cpp 相应判断gpu类型的地方，参考MUSA的方式增加新gpu的相关类型指定

4. 对于和cuda不兼容的gpu，在src/memories/mem_type.h实现对应类型的XXXMemory类，新建cpp文件使用相应计算库提供的显存操作接口实现类；
对于和cuda兼容的gpu，在Memory::createMemoryClass函数内，case该类型gpu的时候，也创建CudaMemory。

5. 修改cuda_memory.cpp，改#ifdef ENABLE_CUDA 为 #if defined(ENABLE_CUDA) || defined(ENABLE_XXX)，这样直接复用cuda的逻辑

6. 在gpu_interface.cpp和gpu_interface.h实现相应gpu的逻辑，如果和cuda兼容，则可以类似5直接修改cuda的判断条件，增加新gpu的ENABLE_XXX, 复用现有cuda逻辑。