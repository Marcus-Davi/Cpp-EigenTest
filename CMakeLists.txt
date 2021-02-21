cmake_minimum_required(VERSION 3.8)
project(EigenTest)

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTABLE_OUTPUT_PATH "bin")

#set(CMAKE_CXX_FLAGS "-O3 -mfpu=neon-fp-armv8 -mfloat-abi=hard -funsafe-math-optimizations -mcpu=cortex-a53")

find_package(Eigen3)

message(STATUS ${Eigen3_DIR})

include_directories(${EIGEN3_INCLUDE_DIRS})
# include_directories("../")

# Matplotlib bindings (optional)
find_package(Python3 COMPONENTS Development NumPy)


add_executable(main src/main.cpp)
add_executable(lm-test src/lm-test.cpp)
add_executable(numerical_diff src/numerical_diff.cpp)

# for plots
target_include_directories(lm-test PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(lm-test Python3::Python Python3::NumPy)

#target_link_libraries(main Eigen3::Eigen)
# target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIRS})



