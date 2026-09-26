get_filename_component(LLAMA_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(llama_kvzip_oracle LANGUAGES C CXX)
    set(AI_LOADER_BUILD_DIR "" CACHE PATH "Existing Ai-Loader build with llama/ggml import libraries")
    if(NOT AI_LOADER_BUILD_DIR)
        message(FATAL_ERROR "Pass -DAI_LOADER_BUILD_DIR=<existing Ai-Loader build>")
    endif()
endif()

set(TARGET llama-kvzip-oracle)
add_executable(${TARGET} kvzip-oracle.cpp "${LLAMA_SOURCE_ROOT}/vendor/hash/sha256/sha256.c")
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE "${LLAMA_SOURCE_ROOT}/vendor" "${LLAMA_SOURCE_ROOT}/vendor/hash")
if(AI_LOADER_BUILD_DIR)
    target_include_directories(${TARGET} PRIVATE
        "${LLAMA_SOURCE_ROOT}/include"
        "${LLAMA_SOURCE_ROOT}/ggml/include")
    target_compile_definitions(${TARGET} PRIVATE LLAMA_SHARED GGML_SHARED)
    target_link_libraries(${TARGET} PRIVATE
        "${AI_LOADER_BUILD_DIR}/llama.cpp/src/Release/llama.lib"
        "${AI_LOADER_BUILD_DIR}/llama.cpp/ggml/src/Release/ggml.lib"
        "${AI_LOADER_BUILD_DIR}/llama.cpp/ggml/src/Release/ggml-base.lib")
else()
    target_link_libraries(${TARGET} PRIVATE llama ggml)
endif()
