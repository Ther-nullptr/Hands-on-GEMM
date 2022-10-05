CUDA_HOME ?= /opt/cuda
CU=$(CUDA_HOME)/bin/nvcc
CC=clang++
LIBS=-lcublas
CPP_SOURCE=./src/cpp
HPP_SOURCE=./src/cpp/include
CUDA_SOURCE=./src/cuda
TEST_SOURCE=./test
INCLUDE_DIR=-I./src/cuda/include -I./src/cpp/include -I./src/cuda/
BUILD=./build
BIN=./bin
MAIN_SOURCE=./benchmark
STD=c++17
FLAGS=-gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_80,code=sm_80
OPTI=-O3  
DEBUG=--debug -g -G -O0
PTXAS_FLAGS=--ptxas-options=-v --expt-relaxed-constexpr

Wno=-Xcudafe "--diag_suppress=declared_but_not_referenced" -Wno-deprecated-gpu-targets

$(BUILD)/%.o: $(CPP_SOURCE)/%.cpp 
	$(CC) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%.o: $(HPP_SOURCE)/%.hpp 
	$(CC) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%.o: $(TEST_SOURCE)/%.cpp
	$(CC) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%.o: $(TEST_SOURCE)/%.cu
	$(CU) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno) $(PTXAS_FLAGS)

$(BUILD)/%.o: $(CUDA_SOURCE)/%.cu
	$(CU) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno) $(PTXAS_FLAGS)

$(BUILD)/%.o: $(MAIN_SOURCE)/%.cu $(DEP)
	$(CU) -std=$(STD) $(OPTI) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno) $(PTXAS_FLAGS)

$(BUILD)/%-d.o: $(CPP_SOURCE)/%.cpp 
	$(CC) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%-d.o: $(HPP_SOURCE)/%.hpp 
	$(CC) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%-d.o: $(TEST_SOURCE)/%.cpp
	$(CC) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%-d.o: $(TEST_SOURCE)/%.cu
	$(CU) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno) $(PTXAS_FLAGS)

$(BUILD)/%-d.o: $(CUDA_SOURCE)/%.cu
	$(CU) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno) $(PTXAS_FLAGS)

$(BUILD)/%-d.o: $(MAIN_SOURCE)/%.cu $(DEP)
	$(CU) -std=$(STD) $(DEBUG) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS) $(Wno) $(PTXAS_FLAGS)

benchmark_%: $(BUILD)/benchmark.o $(BUILD)/%_gemm.o
	mkdir -p $(BIN)
	$(CU) $^ -std=$(STD) $(OPTI) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)
	# sh ${SCRIPT_SOURCE}/$@.sh

i8benchmark: $(BUILD)/i8-benchmark.o $(BUILD)/i8_gemm.o
	mkdir -p $(BIN)
	$(CU) $^ -std=$(STD) $(OPTI) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)
	# sh ${SCRIPT_SOURCE}/$@.sh

test_%: $(BUILD)/test.o $(BUILD)/%_gemm.o
	$(CU) $^ -std=$(STD) -o $(BIN)/$@ -g $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

test_%-d: $(BUILD)/test-d.o $(BUILD)/%_gemm-d.o
	$(CU) $^ -std=$(STD) -o $(BIN)/$@ -g $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

i8gemm-test: $(BUILD)/i8gemm-test.o $(BUILD)/i8_gemm.o
	$(CU) $^ -std=$(STD) -o $(BIN)/$@ -g $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

i8gemm-test-d: $(BUILD)/i8gemm-test-d.o $(BUILD)/i8_gemm-d.o
	$(CU) $^ -std=$(STD) -o $(BIN)/$@ -g $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

.PHONY: clean
clean:
	rm $(BUILD)/*