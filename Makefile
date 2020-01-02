CC = nvcc

CFLAGS += -w 
CFLAGS += -std=c++11 
CFLAGS += -O0
CFLAGS += -gencode arch=compute_60,code=sm_60
CFLAGS += -m64 
CFLAGS += -ccbin g++
CFLAGS += -g -G
# CFLAGS += -DDEBUG 

INC = -I./include
LIB = -L. -L./lib -lcublas 

ROOT = .

LIB_DIR = $(ROOT)/lib
INC_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin
ASM_DIR = $(ROOT)/asm
OBJ_DIR = $(ROOT)/obj

SRC_DIR = $(shell find src -type d)
TEST_DIR = $(ROOT)/test

vpath %.cpp $(SRC_DIR)
vpath %.cu $(SRC_DIR)
vpath %.cpp $(TEST_DIR)

CPP_SRC += $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.cpp) )
CU_SRC += $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.cu) )
CPP_OBJ += $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(CPP_SRC)))
CU_OBJ += $(patsubst %.cu, $(OBJ_DIR)/%.o, $(notdir $(CU_SRC)))

CPP_ASM += $(patsubst %.cpp, $(ASM_DIR)/%.S, $(notdir $(CPP_SRC)))
CU_ASM += $(patsubst %.cu, $(ASM_DIR)/%.S, $(notdir $(CU_SRC)))

OBJ = $(CPP_OBJ) $(CU_OBJ)
ASM = $(CPP_ASM) $(CU_ASM)

TEST_SRC = $(foreach d,$(TEST_DIR), $(wildcard $(d)/*.cpp) )
TEST_OBJ = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(TEST_SRC)))
BIN = $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TEST_SRC) )

$(info $(BIN))

.PHONY : clean all

all : bin

bin : $(BIN) asm

asm : $(ASM) 

$(BIN) : $(BIN_DIR)/% : $(OBJ_DIR)/%.o $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIB)

$(CPP_OBJ) : $(OBJ_DIR)/%.o : %.cpp
	$(CC) -c -o $@ $^ $(CFLAGS) $(INC)

$(CU_OBJ) : $(OBJ_DIR)/%.o : %.cu
	$(CC) -c -o $@ $^ $(CFLAGS) $(INC)

$(CPP_ASM) : $(ASM_DIR)/%.S : %.cpp
	$(CC) -ptx -o $@ $^ $(CFLAGS) $(INC)

$(CU_ASM) : $(ASM_DIR)/%.S : %.cu
	$(CC) -ptx -o $@ $^ $(CFLAGS) $(INC)

$(TEST_OBJ) : $(OBJ_DIR)/%.o : %.cpp
	$(CC) -c -o $@ $^ $(CFLAGS) $(INC)

clean:
	rm -rf $(OBJ_DIR)/*.o
	rm -rf $(BIN_DIR)/*
	rm -rf $(ASM_DIR)/*.S
