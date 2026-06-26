ifneq ($(wildcard /opt/maca/include/mctlass/mctlass.h),)
BACKEND ?= maca
else
BACKEND ?= cuda
endif

CUTLASS_DIR ?=
ARCH ?= 80
BUILD_DIR ?= build
TARGET ?= $(BUILD_DIR)/$(BACKEND)_gemm_demo
MCPTI_LIST_TARGET ?= $(BUILD_DIR)/mcpti_list
GAP1_C500_TARGET ?= $(BUILD_DIR)/gap1_c500

SRC := demos/cutlass_gemm.cu
MCPTI_LIST_SRC := tools/mcpti_list.cpp
GAP1_C500_SRC := demos/gap1_c500.cu

NVCCFLAGS ?= -O2 -std=c++17 --expt-relaxed-constexpr
GENCODE := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

INCLUDES :=
LDFLAGS :=

ifeq ($(BACKEND),maca)
MACA_PATH ?= /opt/maca
MCTLASS_DIR ?= $(MACA_PATH)
NVCC ?= $(MACA_PATH)/tools/cu-bridge/bin/cucc
INCLUDES += -I$(MCTLASS_DIR)/include -I$(MCTLASS_DIR)/include/mcr
LDFLAGS += -L$(MACA_PATH)/lib -lmcpti
NVCCFLAGS += -DUSE_MCTLASS=1 -Wno-macro-redefined -Wno-sometimes-uninitialized -Wno-maca-compat -Wno-return-type
else ifeq ($(BACKEND),cuda)
NVCC ?= nvcc
ifneq ($(strip $(CUTLASS_DIR)),)
  ifneq ($(wildcard $(CUTLASS_DIR)/include/cutlass/cutlass.h),)
    INCLUDES += -I$(CUTLASS_DIR)/include
  else
    INCLUDES += -I$(CUTLASS_DIR)
  endif
endif
else
$(error Unsupported BACKEND=$(BACKEND). Use BACKEND=maca or BACKEND=cuda)
endif

.PHONY: all run gap1 gap1-c500 run-gap1-c500 mcpti-list clean

all: $(TARGET)

$(TARGET): $(SRC) Makefile
	@mkdir -p $(dir $@)
	CUBRIDGE_HOME=$(abspath $(BUILD_DIR)) $(NVCC) $(NVCCFLAGS) $(GENCODE) $(INCLUDES) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

gap1:
	python3 tools/gap1_benchmarks.py

gap1-c500: $(GAP1_C500_TARGET)

run-gap1-c500: $(GAP1_C500_TARGET)
	./$(GAP1_C500_TARGET)

$(GAP1_C500_TARGET): $(GAP1_C500_SRC) Makefile
	@mkdir -p $(dir $@)
	CUBRIDGE_HOME=$(abspath $(BUILD_DIR)) $(MACA_PATH)/tools/cu-bridge/bin/cucc -O2 -std=c++17 --expt-relaxed-constexpr $(GENCODE) -Wno-macro-redefined -Wno-sometimes-uninitialized -Wno-maca-compat -Wno-return-type -I$(MACA_PATH)/include -I$(MACA_PATH)/include/mcr $< -o $@ -L$(MACA_PATH)/lib -lmcpti

mcpti-list: $(MCPTI_LIST_TARGET)

$(MCPTI_LIST_TARGET): $(MCPTI_LIST_SRC) Makefile
	@mkdir -p $(dir $@)
	$(CXX) -O2 -std=c++17 -I/opt/maca/include -I/opt/maca/include/mcr $< -o $@ -L/opt/maca/lib -lmcruntime -lmcpti

clean:
	rm -rf $(BUILD_DIR)
