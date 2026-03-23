# 编译器
NVCC = nvcc
CXX = g++

# 编译选项
CUDA_ARCH = -arch=sm_70
NVCC_FLAGS = -O3 -std=c++14
CXX_FLAGS = -O3 -std=c++14

# 包含路径
INCLUDES = -I/public/software/compiler/dtk-25.04/cuda/cuda/include
LIBS = -L/public/software/compiler/dtk-25.04/cuda/cuda/lib64 -lcudart -lcublas

# 目标
TARGET = TensorCore
SRCS = TensorCore.cu
OBJS = $(SRCS:.cu=.o)
OBJS := $(OBJS:.cpp=.o)

# 默认目标
all: $(TARGET)

# 链接
$(TARGET): $(OBJS)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# CUDA 文件编译规则
%.o: %.cu
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# C++ 文件编译规则
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# 清理
clean:
	rm -f $(TARGET) $(OBJS)

# 运行
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
