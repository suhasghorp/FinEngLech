include_directories(../extern/catch2)
include_directories(../extern)

#install TBB from https://github.com/wjakob/tbb
#git clone https://github.com/wjakob/tbb.git
#cd tbb/build/
#cmake ..
#make -j
#sudo make install

find_package(TBB CONFIG REQUIRED tbb)
find_package(Threads REQUIRED)
find_package(fmt CONFIG REQUIRED)

add_executable(BSMonteCarloConvergenceTest catch_config.cpp BSMonteCarloConvergenceTest.cpp)
target_link_libraries(BSMonteCarloConvergenceTest FinEngLech TBB::tbb Threads::Threads fmt::fmt)
add_test(NAME BSMonteCarloConvergenceTest COMMAND BSMonteCarloConvergenceTest)

add_executable(HullWhiteZCBTest catch_config.cpp HullWhiteZCBTest.cpp)
target_link_libraries(HullWhiteZCBTest FinEngLech TBB::tbb Threads::Threads fmt::fmt)
add_test(NAME HullWhiteZCBTest COMMAND HullWhiteZCBTest)



