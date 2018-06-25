CU_FILES := $(wildcard src/*.cu)
CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)) $(notdir $(CPP_FILES:.cpp=.o)))
$(info OBJ_FILES is $(OBJ_FILES))
OBJDIR = obj
INCL = common/inc
LIBS = -lglfw -lGL -lGLEW



CC = g++
CFLAGS = --std=c++11

NVCC = nvcc
NVFLAGS = -arch=sm_30 --std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D__STRING_ANSI__ 
#vmc: $(OBJ_FILES) 
#	$(NVCC) $(NVFLAGS) -o vmc $(OBJDIR)/main.o $(LIBS)

#obj/%.o: $(CU_FILES)
#	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

# - lpng when you use it later 

# vmc: $(OBJ_FILES)
# 	$(NVCC) $(NVFLAGS) -o $@ $(OBJ_FILES) $(LIBS) $(INCL)

# run_gl_test: obj/main.o obj/SoftmaxInitializer.o
# 	$(NVCC) $(NVFLAGS) -o $@ $(OBJ_FILES) $(LIBS) -I$(INCL) -lpng

run_softmax: $(OBJ_FILES)
	$(NVCC) $(NVFLAGS) -o $@ $(OBJ_FILES) $(LIBS) -I$(INCL) -lpng

obj/main.o: src/main.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS) -I$(INCL)

obj/GLInstance.o: src/GLInstance.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

obj/SoftmaxInitializer.o: src/SoftmaxInitializer.cu src/SoftmaxInitializer.h
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS) -I$(INCL)

obj/SoftmaxRegression.o: src/SoftmaxRegression.cu src/SoftmaxRegression.h
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

obj/DivLogLikelihood_Kernel.o: src/DivLogLikelihood_Kernel.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

obj/CalculateProbability_Kernel.o: src/CalculateProbability_Kernel.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

obj/Color_Kernel.o: src/Color_Kernel.cu 
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)