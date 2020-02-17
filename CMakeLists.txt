cmake_minimum_required(VERSION 3.3)
project(Handwritten_Numbers_Detection)
project(Perceptron_Neural_Network_Test)
project(Cost_Function_Supervise_Learning)
project(Pseudoin_Rule_Hebbian_Supervised_Learning)
project(Adaline_LMS_Learning)
project(Backpropagation_Demo)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_C_COMPILER "/usr/bin/gcc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lfann")

set(HANDWRITTEN_NUMBERS_DETECTION_SOURCE_FILES HandwrittenNumbersDetection/cpp/Handwritten_main.cpp AANN/Learn.cpp AANN/Learn.h)
add_executable(Handwritten_Numbers_Detection ${HANDWRITTEN_NUMBERS_DETECTION_SOURCE_FILES})

set(PERCEPTRON_NEURAL_NETWORK_TEST_SOURCE_FILES PerceptronNeuralNetwork/Neurons/PerceptronNeuron.cpp PerceptronNeuralNetwork/Neurons/PerceptronNeuron.h PerceptronNeuralNetwork/Neurons/PerceptronNetwork.cpp PerceptronNeuralNetwork/Neurons/PerceptronNetwork.h PerceptronNeuralNetwork/Neurons/DEFINITIONS.h PerceptronNeuralNetwork/Neurons/Perceptron_main.cpp)
add_executable(Perceptron_Neural_Network_Test ${PERCEPTRON_NEURAL_NETWORK_TEST_SOURCE_FILES})

set(COST_FUNCTION_SUPERVISE_LEARNING_SOURCE_FILES CostFunctoinSupervisedLearning/main.cpp CostFunctoinSupervisedLearning/Learn.cpp CostFunctoinSupervisedLearning/Learn.h CostFunctoinSupervisedLearning/ConjugateGradientLearn.cpp CostFunctoinSupervisedLearning/ConjugateGradientLearn.h)#CostFunctoinSupervisedLearning/OpenCLCostFunctionRouter.cpp CostFunctoinSupervisedLearning/OpenCLCostFunctionRouter.h)
add_executable(Cost_Function_Supervise_Learning ${COST_FUNCTION_SUPERVISE_LEARNING_SOURCE_FILES})

set(PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_SOURCE_FILES PerceptronNeuralNetwork/Supervised/PseudionRule-DigitalNumberDetection/main.cpp PerceptronNeuralNetwork/Supervised/PseudionRule-DigitalNumberDetection/Matrix.h PerceptronNeuralNetwork/Supervised/PseudionRule-DigitalNumberDetection/SupervisedHebbianNetwork.cpp PerceptronNeuralNetwork/Supervised/PseudionRule-DigitalNumberDetection/SupervisedHebbianNetwork.h PerceptronNeuralNetwork/Supervised/PseudionRule-DigitalNumberDetection/SupervisedHebbianNeuron.cpp PerceptronNeuralNetwork/Supervised/PseudionRule-DigitalNumberDetection/SupervisedHebbianNeuron.h PerceptronNeuralNetwork/Supervised/PseudionRule-DigitalNumberDetection/DEFINITIONS.h PerceptronNeuralNetwork/Supervised/PseudionRule-DigitalNumberDetection/Learn.cpp PerceptronNeuralNetwork/Supervised/PseudionRule-DigitalNumberDetection/Learn.h)
add_executable(Pseudoin_Rule_Hebbian_Supervised_Learning ${PSEUDOIN_RULE_HEBBIAN_SUPERVISED_LEARNING_SOURCE_FILES})

set(ADALINE_LMS_LEARNING_SOURCE_FILES Widrow-Hoff-Learning/ADALINE-Network/LMS-Learning-Algorithm/Learn.cpp Widrow-Hoff-Learning/ADALINE-Network/LMS-Learning-Algorithm/Learn.h Widrow-Hoff-Learning/ADALINE-Network/LMS-Learning-Algorithm/main.cpp )
add_executable(Adaline_LMS_Learning ${ADALINE_LMS_LEARNING_SOURCE_FILES})

set(BACKPROPAGATION_DEMO_SOURCE_FILES Backpropagation/Demo/main.cpp AANN/Learn.cpp AANN/Learn.h)
add_executable(Backpropagation_Demo ${BACKPROPAGATION_DEMO_SOURCE_FILES})

file(GLOB DL_HANDWRITTEN_NUMBERS_DETECTION_TRAINING_FILES
#        "HandwrittenNumbersDetection/training/train-images-idx3-ubyte.gz"
#        "HandwrittenNumbersDetection/training/train-labels-idx1-ubyte.gz"
#        "HandwrittenNumbersDetection/training/train-images-idx3-ubyte"
#        "HandwrittenNumbersDetection/training/train-labels-idx1-ubyte"
)

file(GLOB DL_HANDWRITTEN_NUMBERS_DETECTION_TEST_FILES
#        "HandwrittenNumbersDetection/test/t10k-images-idx3-ubyte.gz"
#        "HandwrittenNumbersDetection/test/t10k-labels-idx1-ubyte.gz"
)
file(GLOB DL_HANDWRITTEN_NUMBERS_DETECTION_CL_FILES

)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

file(COPY ${DL_HANDWRITTEN_NUMBERS_DETECTION_TRAINING_FILES} DESTINATION ${CMAKE_BINARY_DIR}/training/)
file(COPY ${DL_HANDWRITTEN_NUMBERS_DETECTION_CL_FILES} DESTINATION ${CMAKE_BINARY_DIR}/opencl/)
file(COPY ${DL_HANDWRITTEN_NUMBERS_DETECTION_TEST_FILES} DESTINATION ${CMAKE_BINARY_DIR}/test/)

file(GLOB DL_COST_FUNCTION_FILES
#        "HandwrittenNumbersDetection/test/t10k-images-idx3-ubyte.gz"
        "CostFunctoinSupervisedLearning/data_test02.txt"
)
file(COPY ${DL_COST_FUNCTION_FILES} DESTINATION ${CMAKE_BINARY_DIR}/test/)

find_library(OpenCL REQUIRED)
find_library(fann2.2.0 REQUIRED)
find_package(Eigen3 REQUIRED)
FIND_PACKAGE( Boost 1.59 COMPONENTS REQUIRED )
FIND_PACKAGE(VTK 7.0 REQUIRED)

include_directories(/usr/local/include/)

include(${VTK_USE_FILE})
target_link_libraries(Handwritten_Numbers_Detection ${OPENCL_LIBRARIES})
target_link_libraries(Handwritten_Numbers_Detection ${VTK_LIBRARIES})
target_link_libraries(Cost_Function_Supervise_Learning ${VTK_LIBRARIES})
target_link_libraries(Adaline_LMS_Learning ${VTK_LIBRARIES})
target_link_libraries(Backpropagation_Demo ${VTK_LIBRARIES})
target_link_libraries(Backpropagation_Demo ${OPENCL_LIBRARIES})
target_link_libraries(Backpropagation_Demo ${FANN_LIBRARIES})
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories( ${OPENCL_INCLUDE_DIR} )
include_directories( ${VTK_INCLUDE_DIR} )