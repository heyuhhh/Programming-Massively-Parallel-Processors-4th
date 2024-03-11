CC = nvcc
CFLAGS = -I "../Ch11 - Prefix sum (scan)" -I "../Ch12 - Merge" -I ../utils -rdc=true
LDFLAGS =

TARGET = radixSort.out

# 定义对象文件，直接列出，不通过自动推导
OBJS = main.o sort.o merge_kernel.o sort_kernel.o scan.o scan_kernel.o

all: $(TARGET)

# 编译.cpp文件
main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp -o main.o

sort.o: sort.cu
	$(CC) $(CFLAGS) -c sort.cu -o sort.o

# 对于路径中包含空格的文件，确保在命令中正确引用
merge_kernel.o:
	$(CC) $(CFLAGS) -c "../Ch12 - Merge/merge_kernel.cu" -o merge_kernel.o

scan.o:
	$(CC) $(CFLAGS) -c "../Ch11 - Prefix sum (scan) /scan.cu" -o scan.o

scan_kernel.o:
	$(CC) $(CFLAGS) -c "../Ch11 - Prefix sum (scan) /scan_kernel.cu" -o scan_kernel.o

sort_kernel.o: sort_kernel.cu
	$(CC) $(CFLAGS) -c sort_kernel.cu -o sort_kernel.o

$(echo $(OBJS))

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean