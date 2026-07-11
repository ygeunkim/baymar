cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Get version number for DESCRIPTION
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/DESCRIPTION" DESC_CONTENT)
string(REGEX MATCH "Version:[ \t]*([0-9.]+)" _ "${DESC_CONTENT}")
if(CMAKE_MATCH_1)
    set(BAYMAR_VERSION "${CMAKE_MATCH_1}")
    message(STATUS "Detected baymar version from DESCRIPTION: ${BAYMAR_VERSION}")
else()
    message(WARNING "Could not parse version from DESCRIPTION file. Defaulting to 0.0.0.9000")
    set(BAYMAR_VERSION "0.0.0.9000")
endif()

project(baymar VERSION ${BAYMAR_VERSION} LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
    bvhar
    GIT_REPOSITORY https://github.com/ygeunkim/bvhar.git
    GIT_TAG develop
    GIT_SHALLOW TRUE
    OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(bvhar)

# Header-only library
add_library(baymar INTERFACE)
add_library(baymar::baymar ALIAS baymar)

target_compile_features(baymar INTERFACE cxx_std_17)

# Add /bigobj for MSVC to handle large object files
target_compile_options(baymar INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
)

target_compile_definitions(baymar INTERFACE
    BAYMAR_VERSION="${PROJECT_VERSION}"
)

target_include_directories(baymar INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inst/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(baymar INTERFACE
    bvhar
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BAYMAR_BUILD_EXAMPLES "Build example C++ programs" OFF)
option(BAYMAR_BUILD_TESTS "Build test C++ programs" OFF)
option(BAYMAR_ENABLE_COVERAGE "Enable C++ test coverage" OFF)

if(BAYMAR_ENABLE_COVERAGE)
    target_compile_options(baymar INTERFACE --coverage -O0 -g)
    target_link_libraries(baymar INTERFACE --coverage)
endif()

if(BAYMAR_BUILD_EXAMPLES)
    add_subdirectory(inst/examples examples)
endif()

if(BAYMAR_BUILD_TESTS)
    add_subdirectory(inst/tests tests)
endif()

# Install header files
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inst/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

