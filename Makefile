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

SRC := demos/cutlass_gemm.cu
MCPTI_LIST_SRC := tools/mcpti_list.cpp

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

.PHONY: all run mcpti-list clean

all: $(TARGET)

$(TARGET): $(SRC) Makefile
	@mkdir -p $(dir $@)
	CUBRIDGE_HOME=$(abspath $(BUILD_DIR)) $(NVCC) $(NVCCFLAGS) $(GENCODE) $(INCLUDES) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

mcpti-list: $(MCPTI_LIST_TARGET)

$(MCPTI_LIST_TARGET): $(MCPTI_LIST_SRC) Makefile
	@mkdir -p $(dir $@)
	$(CXX) -O2 -std=c++17 -I/opt/maca/include -I/opt/maca/include/mcr $< -o $@ -L/opt/maca/lib -lmcruntime -lmcpti

clean:
	rm -rf $(BUILD_DIR)
