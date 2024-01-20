cmake_minimum_required(VERSION 3.13)
project(Curvature)

set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -g -Wextra -pthread")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

FIND_PACKAGE( Boost COMPONENTS graph iostreams)
#set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -pthread -g")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread -g")
#INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers/ )

#LINK_DIRECTORIES(/opt/homebrew/Cellar/boost/1.81.0_1/lib /opt/homebrew/Cellar/graph-tool/2.53/lib/python3.11/site-packages/graph_tool)

#link_directories(/usr/libx86_64-linux-gnu/)

add_executable(Bgp bgp.cpp curvatureHandler.h)
add_executable(Road road.cpp curvatureHandler.h)
add_executable(Regular regular.cpp curvatureHandler.h)
#add_executable(Python ricci.cpp ricci.h)
#add_executable(test test.cpp)
#add_executable(Prot prot.cpp curvatureHandler.h)

#add_executable(regular regular.cpp curvatureHandler.h GraphSpecial.h)
#add_executable(singleCurvature curvature.cpp curvatureHandler.h GraphSpecial.h)

#TARGET_LINK_LIBRARIES(flot ${Boost_LIBRARIES} tbb)
TARGET_LINK_LIBRARIES(Bgp ${Boost_LIBRARIES} )
#TARGET_LINK_LIBRARIES(correctFile ${Boost_LIBRARIES})
#TARGET_LINK_LIBRARIES(regular ${Boost_LIBRARIES})
#TARGET_LINK_LIBRARIES(singleCurvature ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(Road ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(Regular ${Boost_LIBRARIES} )
#TARGET_LINK_LIBRARIES(Python ${Boost_LIBRARIES} /opt/homebrew/Cellar/graph-tool/2.53/lib/python3.11/site-packages/graph_tool/libgraph_tool_core.so)
#TARGET_LINK_LIBRARIES(test  /Library/Frameworks/Python.framework/Versions/3.9/lib/libpython3.9.dylib)
#TARGET_LINK_LIBRARIES(Prot ${Boost_LIBRARIES} )

