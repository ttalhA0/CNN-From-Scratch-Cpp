CXX = g++
CXXFLAGS = -std=c++14 -O3 -Wall -I/usr/include/eigen3 -I./src

COMMON_SRCS = src/ConvLayer.cpp src/MaxPoolLayer.cpp src/SimpleMLP.cpp src/fileReader.cpp
COMMON_OBJS = $(COMMON_SRCS:.cpp=.o)

TARGET_TRAIN = train
TARGET_INFERENCE = inference

all: $(TARGET_TRAIN) $(TARGET_INFERENCE)

$(TARGET_TRAIN): src/train.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET_TRAIN) src/train.cpp $(COMMON_OBJS)

$(TARGET_INFERENCE): src/inference.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET_INFERENCE) src/inference.cpp $(COMMON_OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(TARGET_TRAIN) $(TARGET_INFERENCE)

clean-weights:
	rm -f *.bin