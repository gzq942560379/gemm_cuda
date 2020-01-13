CC = nvcc

CFLAGS += -w 
CFLAGS += -std=c++11 
CFLAGS += -O2
# CFLAGS += -gencode arch=compute_60,code=sm_60
CFLAGS += -m64 
CFLAGS += -ccbin g++
# CFLAGS += --ptxas-options=-v

# CFLAGS += -g -G
# CFLAGS += -DDEBUG 

INC = -I./include
LIB = -L. -L./lib -lcublas 

ROOT = .

LIB_DIR = $(ROOT)/lib
INC_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin
ASM_DIR = $(ROOT)/asm
OBJ_DIR = $(ROOT)/obj
KERNEL_DIR = $(ROOT)/kernel

SRC_DIR = $(shell find src -type d)
TEST_DIR = $(ROOT)/test

vpath %.cpp $(SRC_DIR)
vpath %.cpp $(TEST_DIR)
vpath %.cu $(SRC_DIR)
vpath %.cu $(KERNEL_DIR)

KERNEL_SRC = $(wildcard $(KERNEL_DIR)/*.cu)
KERNEL_OBJ = $(patsubst %.cu, $(OBJ_DIR)/%.o, $(notdir $(KERNEL_SRC)))
KERNEL_ASM = $(patsubst %.cu, $(ASM_DIR)/%.S, $(notdir $(KERNEL_SRC)))

CPP_SRC += $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.cpp) )
CU_SRC += $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.cu) )
CPP_OBJ += $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(CPP_SRC)))
CU_OBJ += $(patsubst %.cu, $(OBJ_DIR)/%.o, $(notdir $(CU_SRC)))

OBJ = $(CPP_OBJ) $(CU_OBJ)
ASM = $(KERNEL_ASM)

TEST_SRC = $(foreach d,$(TEST_DIR), $(wildcard $(d)/*.cpp) )
TEST_OBJ = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(TEST_SRC)))


BIN = $(patsubst %.cu, $(BIN_DIR)/%, $(notdir $(KERNEL_SRC)) )

.PHONY : clean all

all : bin

bin : $(BIN) asm

asm : $(ASM) 

$(BIN) : $(BIN_DIR)/% : $(OBJ_DIR)/matMul.o $(OBJ_DIR)/%.o $(OBJ) 
	$(CC) -o $@ $^ $(CFLAGS) $(LIB)

$(CPP_OBJ) : $(OBJ_DIR)/%.o : %.cpp
	$(CC) -c -o $@ $^ $(CFLAGS) $(INC)

$(CU_OBJ) : $(OBJ_DIR)/%.o : %.cu
	$(CC) -c -o $@ $^ $(CFLAGS) $(INC)

$(KERNEL_OBJ) : $(OBJ_DIR)/%.o : %.cu
	$(CC) -c -o $@ $^ $(CFLAGS) $(INC)

$(TEST_OBJ) : $(OBJ_DIR)/%.o : %.cpp
	$(CC) -c -o $@ $^ $(CFLAGS) $(INC)

$(KERNEL_ASM) : $(ASM_DIR)/%.S : %.cu
	$(CC) -ptx -o $@ $^ $(CFLAGS) $(INC)

clean:
	rm -rf $(OBJ_DIR)/*.o
	rm -rf $(BIN_DIR)/*
	rm -rf $(ASM_DIR)/*.S
