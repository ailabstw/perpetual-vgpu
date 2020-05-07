# Copyright 2020 Hung-Hsin Chen, LSA Lab, National Tsing Hua University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# path to CUDA installation
CUDA_PATH ?= /usr/local/cuda

SMS ?= 30 35 37 50 52 60 61 70 72 75
GENCODE_FLAGS += $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))

NVCC ?= $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

CUDA_LDFLAGS += -lcuda -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib64/stubs
LDFLAGS += -ldl -lrt

CXXFLAGS += -std=c++11 -fPIC
CXXFLAGS += -O2

all: libumhook.so.1

umhook.o: umhook.cpp
	$(NVCC) -m64 --compiler-options "$(CXXFLAGS)" $(GENCODE_FLAGS) -o $@ -c $<

libumhook.so.1: umhook.o
	$(EXEC) $(NVCC) -shared -m64 $(GENCODE_FLAGS) -o $@ $+ $(CUDA_LDFLAGS) $(LDFLAGS)

clean:
	rm -f *.o *.so.1
