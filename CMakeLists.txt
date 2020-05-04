cmake_minimum_required(VERSION 3.8)
project(EigenTest)


set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(CMAKE_CXX_FLAGS "-O3 -mfpu=neon-fp-armv8 -mfloat-abi=hard -funsafe-math-optimizations -mcpu=cortex-a53")

find_package(Eigen3)

message(STATUS ${Eigen3_DIR})


add_executable(main src/main.cpp)
#target_link_libraries(main Eigen3::Eigen)
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIRS})
