.PHONY: all run clean

NVCC = nvcc

BUILD_DIR = build

CXXFLAGS = -O3 -std=c++20
CUDAFLAGS = -arch=sm_86
INCLUDES = -Iinclude
LDLIBS = -lcurand

SOURCES := $(shell find src -type f \( -name "*.cpp" -o -name "*.cu" \))

TARGET = $(BUILD_DIR)/matmul

all:
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) $(INCLUDES) $(SOURCES) $(LDLIBS) -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
