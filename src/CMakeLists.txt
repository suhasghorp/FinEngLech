# Find all headers in the project include folder

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)

set(SOURCES ${SOURCES})


# Make a static library (tradeoff of memory for better performance)
add_library(FinEngLech STATIC ${SOURCES} "BlackScholesJumps.cpp" "BlackScholesJumps.h" "HullWhiteZCB.h")

# Add an alias so the target can be used while building
add_library(FinEngLech::FinEngLech ALIAS FinEngLech)

# Set the include directory for builds (tests, examples, etc) and for clients
target_include_directories(FinEngLech
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src)

# Associate the include files with the target, so they will be installed
set_target_properties(FinEngLech PROPERTIES PUBLIC_HEADER "${HEADERS}")

find_package(TBB REQUIRED tbb)

find_package(Eigen3 CONFIG REQUIRED)

find_package(NLopt CONFIG REQUIRED)

find_package(Boost COMPONENTS system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(../extern)

target_link_libraries(FinEngLech ${TBB_IMPORTED_TARGETS} ${Boost_LIBRARIES} Eigen3::Eigen NLopt::nlopt)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

# Set install targets and destinations
install(TARGETS FinEngLech
        EXPORT FinEngLechTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include/FinEngLech)

# Export additional install files (for clients using CMake)
install(EXPORT FinEngLechTargets
        NAMESPACE FinEngLech::
        FILE FinEngLechConfig.cmake
        DESTINATION lib/cmake/FinEngLech)