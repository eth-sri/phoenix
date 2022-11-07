# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.12)

project(Fhe VERSION 1.0 LANGUAGES CXX)

SET(SEAL_DIR "~/libs/lib/cmake/SEAL-3.6")
find_package(SEAL 3.6.6 REQUIRED)

find_package(hcephes REQUIRED)
# https://github.com/limix/hcephes
# conda install -c conda-forge hcephes

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)


SET(srcs
${CMAKE_CURRENT_LIST_DIR}/smoothing/sgn.cpp
${CMAKE_CURRENT_LIST_DIR}/nn/mlp.cpp
${CMAKE_CURRENT_LIST_DIR}/nn/dense_layer.cpp
${CMAKE_CURRENT_LIST_DIR}/nn/learnable_square.cpp
${CMAKE_CURRENT_LIST_DIR}/nn/dataloader.cpp
${CMAKE_CURRENT_LIST_DIR}/utils/ckks_manager.cpp
${CMAKE_CURRENT_LIST_DIR}/linalg/linalg.cpp
${CMAKE_CURRENT_LIST_DIR}/utils/jsoncpp/jsoncpp.cpp
)


include_directories(${CMAKE_CURRENT_LIST_DIR})

add_executable(phoenix)
target_sources(phoenix
    PRIVATE
    ${srcs}
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)
target_link_libraries(phoenix SEAL::seal HCEPHES::hcephes Threads::Threads)

add_executable(mockphoenix)
target_sources(mockphoenix
    PRIVATE
    ${srcs}
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)
target_compile_definitions(mockphoenix PUBLIC MOCK)
target_link_libraries(mockphoenix PRIVATE HCEPHES::hcephes Threads::Threads)
# do not link with seal