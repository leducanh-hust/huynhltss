# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

# Directories
SRC_DIR = src
BUILD_DIR = build
KERNEL_DIR = kernels

# AMD APP SDK Paths
OPENCL_INC = "C:/OpenCL-SDK/include"
OPENCL_LIB = "C:/OpenCL-SDK/lib"

# Linker Flags
LDFLAGS = -L$(OPENCL_LIB) -lOpenCL

# Files
SRC = $(SRC_DIR)/radix.cpp
EXE = $(BUILD_DIR)/radix.exe

# Pass Kernel File Path as a Macro
KERNEL_FILE = $(KERNEL_DIR)/radix.cl
CPPFLAGS = -DKERNEL_FILE_PATH=\"$(KERNEL_FILE)\" -I$(OPENCL_INC)

# Default target
all: $(EXE)

# Build executable
$(EXE): $(SRC)
	@echo Building project...
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(SRC) -o $@ $(LDFLAGS)
	@echo Build successful!

# Run Program
run: all
	@echo Running program...
	@$(BUILD_DIR)/radix.exe
	@echo Run successfully!

# Clean build files
clean:
	@if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
	@echo Cleaned build files!
