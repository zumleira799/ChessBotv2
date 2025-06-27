NVCC = nvcc
AS = as

NVCC_FLAGS = -rdc=true -g
AS_FLAGS = 

TARGET = main
ASSEMBLY_FILE = fileHandler.s
ASSEMBLY_OBJECT = fileH.o
CUDA_FILE = main.cu
CUDA_OBJECT = main.o

all: $(TARGET)

$(ASSEMBLY_OBJECT): $(ASSEMBLY_FILE)
	$(AS) $(AS_FLAGS) -o $@ $<


$(CUDA_OBJECT): $(CUDA_FILE)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(TARGET): $(ASSEMBLY_OBJECT) $(CUDA_OBJECT)
	$(NVCC) -Xlinker=-no-pie -g -o $@ $^

clean:
	rm -f $(TARGET) $(ASSEMBLY_OBJECT) $(CUDA_OBJECT)