cmake_minimum_required(VERSION 3.4)

if(${CMAKE_VERSION} VERSION_LESS 3.22)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.22)
endif()

if(DEFINED PROJECT_NAME)
    message(FATAL_ERROR "PROJECT_NAME is already defined")
    return()
endif()

set(PROJECT_NAME "GEORD")

project(${PROJECT_NAME} VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_library(${PROJECT_NAME} INTERFACE)

#########################################
#
#  Find MPI
#
#########################################
find_package(MPI 3 REQUIRED)

if(DEFINED GEORD_BUILD_TESTS AND GEORD_BUILD_TESTS)
    find_package(GTest)
    if(GTEST_FOUND)
        add_subdirectory(tests)
    else()
        message(WARNING "GTest not found, tests will not be built")
    endif()
endif()

if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
else()
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if(DEFINED GEORD_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${GEORD_INSTALL_PREFIX} CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
else()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
endif()

add_subdirectory(include)

install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
