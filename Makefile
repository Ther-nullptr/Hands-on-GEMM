CUDA_HOME ?= /usr/local/cuda
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
FLAGS=-gencode=arch=compute_80,code=sm_80
OPTI=-O3  
DEBUG=--debug -g -G -O0
PTXAS_FLAGS=--ptxas-options=-v --expt-relaxed-constexpr -lineinfo
Wno=-Xcudafe "--diag_suppress=declared_but_not_referenced" -Wno-deprecated-gpu-targets

# for profiling
EXE=./bin/profile_1_naive_template
CMD_OPT=128
ARCH=sm_80
KERNEL=matrixMul
NCU=/opt/nvidia/nsight-compute/2023.1.0/ncu

# metrics for roofline
NCU_FLAG := --profile-from-start=ON --kernel-name $(KERNEL) # --launch-skip 2 --launch-count 1 
# NCU_FLAG := --profile-from-start=ON 
metrics_gpu = sm__cycles_elapsed.max,gpc__cycles_elapsed.avg.per_second,
metrics_compute = smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.avg.peak_sustained_elapsed,breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed,
metrics_memory = breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,
metrics_shared = smsp__sass_data_bytes_mem_shared_op_ld,smsp__sass_data_bytes_mem_shared_op_ldgsts,smsp__sass_data_bytes_mem_shared_op_ldgsts_cache_bypass,smsp__sass_data_bytes_mem_shared_op_ldgsts_cache_access,smsp__sass_data_bytes_mem_shared_op_ldsm,smsp__sass_data_bytes_mem_shared_op_st,smsp__sass_data_bytes_mem_shared,smsp__sass_data_bytes_mem_shared.avg.peak_sustained,smsp__sass_data_bytes_mem_shared.max.peak_sustained,smsp__sass_data_bytes_mem_shared.avg.peak_sustained_elapsed,smsp__sass_data_bytes_mem_shared.avg.per_cycle_elapsed,smsp__sass_data_bytes_mem_shared.sum.pct_of_peak_sustained_elapsed,smsp__sass_data_bytes_mem_shared.sum.pct_of_peak_sustained_active,sm__sass_data_bytes_mem_shared_op_ldsm.avg.pct_of_peak_sustained_active,sm__sass_data_bytes_mem_shared_op_ldsm.avg.pct_of_peak_sustained_elapsed,sm__sass_data_bytes_mem_shared_op_ldsm.avg.peak_sustained,sm__sass_data_bytes_mem_shared_op_ldsm.avg.peak_sustained_active,sm__sass_data_bytes_mem_shared_op_ldsm.avg.peak_sustained_elapsed,sm__sass_data_bytes_mem_shared_op_ldsm.avg.per_cycle_active,sm__sass_data_bytes_mem_shared_op_ldsm.avg.per_cycle_elapsed,sm__sass_data_bytes_mem_shared_op_ld.avg.per_cycle_active,
metrics_issue = 
metrics := $(metrics_gpu)$(metrics_compute)$(metrics_memory)
# metrics_roofline += breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed,breakdown:smsp__sass_data_bytes_mem_shared.avg.pct_of_peak_sustained_elapsed
# metrics ?= gpc__cycles_elapsed.avg,sm__cycles_elapsed.sum,smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum,lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum.per_second,dram__sectors_read.sum,dram__sectors_write.sum,dram__bytes_read.sum
# metrics for shader pipeline
# metrics := smsp__cycles_active.sum,smsp__pipe_alu_cycles_active.sum,smsp__pipe_fma_cycles_active.sum,smsp__pipe_fp64_cycles_active.sum,smsp__pipe_tensor_cycles_active.sum,smsp__pipe_tensor_op_dmma_cycles_active.sum,smsp__pipe_tensor_op_hmma_cycles_active.sum,smsp__pipe_tensor_op_imma_cycles_active.sum,smsp__pipe_shared_cycles_active.sum

# roofline_section ?= --section LaunchStats --section Occupancy --section SpeedOfLight_RooflineChart --section SpeedOfLight_HierarchicalTensorRooflineChart --section MemoryWorkloadAnalysis_Chart --section ComputeWorkloadAnalysis
roofline_section ?= --section SpeedOfLight --section LaunchStats --section SpeedOfLight_HierarchicalTensorRooflineChart --section Occupancy --section MemoryWorkloadAnalysis_Chart --section ComputeWorkloadAnalysis

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

debug_%: $(BUILD)/benchmark_debug.o $(BUILD)/%_gemm.o
	mkdir -p $(BIN)
	$(CU) $^ -std=$(STD) $(OPTI) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)
	# sh ${SCRIPT_SOURCE}/$@.sh

profile_%: $(BUILD)/benchmark_profile.o $(BUILD)/%_gemm.o
	mkdir -p $(BIN)
	$(CU) $^ -std=$(STD) $(OPTI) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

sb_%: $(BUILD)/single-benchmark.o $(BUILD)/%_gemm.o
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
	$(CU) $^ -std=$(STD) $(DEBUG) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

i8gemm-test: $(BUILD)/i8gemm-test.o $(BUILD)/i8_gemm.o
	$(CU) $^ -std=$(STD) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

i8gemm-test-d: $(BUILD)/i8gemm-test-d.o $(BUILD)/i8_gemm-d.o
	$(CU) $^ -std=$(STD) -o $(BIN)/$@ -g $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

i8: $(BUILD)/i8.o $(BUILD)/i8gemm.o
	$(CU) $^ -std=$(STD) $(OPTI) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

i8-d: $(BUILD)/i8-d.o $(BUILD)/i8gemm-d.o
	$(CU) $^ -std=$(STD) $(DEBUG) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

i8_%: $(BUILD)/i8.o $(BUILD)/i8%_gemm.o
	mkdir -p $(BIN)
	$(CU) $^ -std=$(STD) $(OPTI) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)
	# sh ${SCRIPT_SOURCE}/$@.sh

i8-test: $(BUILD)/i8-test.o $(BUILD)/i8gemm.o
	$(CU) $^ -std=$(STD) $(OPTI) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

i8-test-d: $(BUILD)/i8-test-d.o $(BUILD)/i8gemm-d.o
	$(CU) $^ -std=$(STD) $(DEBUG) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

i8-test_%: $(BUILD)/i8-test.o $(BUILD)/i8%_gemm.o
	$(CU) $^ -std=$(STD) $(OPTI) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

i8-test_%-d: $(BUILD)/i8-test-d.o $(BUILD)/i8%_gemm-d.o
	$(CU) $^ -std=$(STD) $(DEBUG) -o $(BIN)/$@ $(LIBS) $(FLAGS) $(Wno) $(PTXAS_FLAGS)

mma_ptx: $(MAIN_SOURCE)/mma_ptx.cu
	$(CU) -std=$(STD) $(OPTI) $(INCLUDE_DIR) $^ -o $(BIN)/$@  $(FLAGS) $(Wno) $(PTXAS_FLAGS)

run:
	./$(EXE) $(CMD_OPT)

ncu_metrics:
	$(NCU) $(NCU_FLAG) --target-processes all --metrics $(metrics) ./$(EXE) $(CMD_OPT) | tee nsight-compute.csv

ncu_roofline:
	$(NCU) $(NCU_FLAG) --target-processes all $(roofline_section) -o roofline -f ./$(EXE) $(CMD_OPT) 

.PHONY: clean
clean:
	rm $(BUILD)/*