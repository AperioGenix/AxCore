cmake_minimum_required(VERSION 3.24)

project(AxCore VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(axcore SHARED
    src/axcore.cpp
    src/connectome.cpp
    src/math.cpp
    src/memory.cpp
)

target_compile_features(axcore PRIVATE cxx_std_20)
target_include_directories(axcore
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(axcore PRIVATE AXCORE_BUILD_DLL)

if(MSVC)
    target_compile_options(axcore PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(axcore PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(axcore PROPERTIES
    OUTPUT_NAME "axcore"
)
