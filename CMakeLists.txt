cmake_minimum_required(VERSION 3.12)

#use the below command to sync clocks
#sudo ntpdate time.windows.com

project(FinEngLech
        VERSION 0.1.0
        DESCRIPTION "A C++ library based on Lech's Fin Eng course on YouTube'"
        LANGUAGES CXX
        )

# Require C++11 for Catch2
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 



#set(CMAKE_CXX_CLANG_TIDY clang-tidy; -header-filter=.; -checks=*;)

# Header-only library
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

include(GNUInstallDirs)
file(GLOB_RECURSE SRC_FILES src/Alglib/*.cpp)

target_include_directories(
  ${PROJECT_NAME}
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

#target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

if (MSVC_VERSION GREATER_EQUAL "1900")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("/std:c++latest" _cpp_latest_flag_supported)
   if (_cpp_latest_flag_supported)
        add_compile_options("/std:c++latest")
   endif()
else(MSVC_VERSION)
   target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
endif()


find_package(Threads)
find_package(fmt CONFIG REQUIRED) 
find_package(TBB REQUIRED tbb)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Catch2 CONFIG REQUIRED)
find_package(NLopt CONFIG REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})


include_directories(extern)

target_compile_options(${PROJECT_NAME} INTERFACE 
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall -Wextra -Wpedantic -Werror -fopenmp>
     $<$<CXX_COMPILER_ID:MSVC>:
          /W4 /openmp /PROFILE>)

# Test

#FITPACK only on Linux WSL to match Lech's Scipy interpolation
add_library(FitPackAppLib SHARED IMPORTED)
set_target_properties(FitPackAppLib PROPERTIES IMPORTED_LOCATION /home/sghorp/fitpackpp/build/libfitpackpp.a)
add_library(FitPackLib SHARED IMPORTED)
set_target_properties(FitPackLib PROPERTIES IMPORTED_LOCATION /home/sghorp/fitpackpp/build/libfitpack.a)

#add_executable(BSMonteCarloConvergenceTest tests/catch_config.cpp tests/BSMonteCarloConvergenceTest.cpp)
#target_link_libraries(BSMonteCarloConvergenceTest PRIVATE TBB::tbb Threads::Threads fmt::fmt Catch2::Catch2 Eigen3::Eigen)
#target_include_directories(BSMonteCarloConvergenceTest PRIVATE ${PROJECT_SOURCE_DIR}/include)

#add_executable(HullWhiteZCBTest tests/catch_config.cpp tests/HullWhiteZCBTest.cpp ${SRC_FILES})
#target_link_libraries(HullWhiteZCBTest PRIVATE TBB::tbb Threads::Threads fmt::fmt Catch2::Catch2 Eigen3::Eigen FitPackAppLib FitPackLib)
#target_include_directories(HullWhiteZCBTest PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(YieldCurveTest tests/catch_config.cpp src/IRSwap.cpp src/YieldCurve.cpp tests/YieldCurveTest.cpp )
#target_link_libraries(YieldCurveTest PRIVATE TBB::tbb Threads::Threads fmt::fmt Catch2::Catch2 Eigen3::Eigen FitPackAppLib FitPackLib)
target_link_libraries(YieldCurveTest PRIVATE TBB::tbb Threads::Threads fmt::fmt Catch2::Catch2 Eigen3::Eigen)
target_include_directories(YieldCurveTest PRIVATE ${PROJECT_SOURCE_DIR}/include)

include(CTest)
include(Catch)
#catch_discover_tests(BSMonteCarloConvergenceTest HullWhiteZCBTest YieldCurveTest)
catch_discover_tests(YieldCurveTest)
enable_testing()

if(FINENGLECH_INSTALL_LIBRARY)
  # locations are provided by GNUInstallDirs
  install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

  install(
    EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

  install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/FinEnglech
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
