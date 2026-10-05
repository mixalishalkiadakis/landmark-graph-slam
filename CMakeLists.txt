cmake_minimum_required(VERSION 3.16)

project(slam_playground)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GTSAM REQUIRED)

add_executable(
    slam_main
    main_keyframes.cpp
    slam_backend.cpp
)

target_link_libraries(
    slam_main
    PRIVATE
    gtsam
)