cmake_minimum_required(VERSION 3.13)
project(Curvature)

set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -g -Wextra -pthread")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

FIND_PACKAGE( Boost COMPONENTS graph iostreams)
#set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -pthread -g")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread -g")
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

LINK_DIRECTORIES(/opt/homebrew/Cellar/boost/1.75.0_2/lib)
link_directories(/opt/homebrew/Cellar/tbb/2021.5.0_2/lib)

include_directories(/usr/include/tbb/)
#link_directories(/usr/libx86_64-linux-gnu/)

add_executable(Curvature main.cpp)
#add_executable(flot flot.cpp)
add_executable(correctFile correctFile.cpp test.cpp curvature.cpp)
add_executable(test test.cpp)
add_executable(regular regular.cpp)
add_executable(singleCurvature curvature.cpp)

#TARGET_LINK_LIBRARIES(flot ${Boost_LIBRARIES} tbb)
TARGET_LINK_LIBRARIES(Curvature ${Boost_LIBRARIES} tbb)
TARGET_LINK_LIBRARIES(correctFile ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(regular ${Boost_LIBRARIES} tbb)
TARGET_LINK_LIBRARIES(singleCurvature ${Boost_LIBRARIES} tbb)

#TARGET_LINK_LIBRARIES(test tbb )
