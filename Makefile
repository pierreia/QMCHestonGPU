# Specify the compiler
NVCC = nvcc

# Specify the include directories
INCLUDES = -I./

# Specify the compiler flags
CFLAGS = -std=c++11

# Specify the CUDA architecture
ARCH = -arch=sm_60

# Specify the linker flags
LDFLAGS = -lcudart -lcurand

# Specify the object files
OBJS = kernel.o main.o product.o

# Specify the target executable
TARGET = HestonMC

# Build the target executable
$(TARGET): $(OBJS)
	$(NVCC) $(ARCH) $(LDFLAGS) $(OBJS) -o $(TARGET)

# Build the object files
kernel.o: kernel.cu kernel.h
	$(NVCC) $(ARCH) $(INCLUDES) $(CFLAGS) -c kernel.cu -o kernel.o

product.o: product.cu product.h
	$(NVCC) $(ARCH) $(INCLUDES) $(CFLAGS) -c product.cu -o product.o

main.o: main.cu kernel.h
	$(NVCC) $(ARCH) $(INCLUDES) $(CFLAGS) -c main.cu -o main.o

# Remove object files and the target executable
clean:
	rm -f $(OBJS) $(TARGET)
