# Default compiler (can be overridden when invoking make)
COMPILER = g++

# Compiler flags for g++
GXXFLAGS = -Wall -O3 -g -std=c++17

# Compiler flags for nvcc
NVCCFLAGS = -O3 -std=c++17 -arch=sm_75

# Target executable
TARGET = main_program

# Source directory
SRCDIR = src

# Source files (located in src/)
SRCS = $(SRCDIR)/MLP_Network.cpp $(SRCDIR)/MLP_Layer.cpp $(SRCDIR)/MNIST.cpp $(SRCDIR)/RBM.cpp main.cpp

# Object files (place object files in the src directory)
OBJS = $(SRCS:.cpp=.o)

# Header files (located in src/)
HDRS = $(SRCDIR)/MLP_Network.h $(SRCDIR)/MLP_Layer.h $(SRCDIR)/MNIST.h $(SRCDIR)/RBM.h 

# The default rule: clean first, then build
all: clean $(TARGET)

# If the compiler is nvcc, use nvcc flags; otherwise, use g++ flags
ifeq ($(COMPILER), nvcc)
    CXX = nvcc
    CXXFLAGS = $(NVCCFLAGS)
else
    CXX = g++
    CXXFLAGS = $(GXXFLAGS)
endif

# Build rule for the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Build rule for object files
$(SRCDIR)/%.o: $(SRCDIR)/%.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)
