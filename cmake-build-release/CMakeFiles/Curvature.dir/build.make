# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ksalamatian/CLionProjects/Curvature

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ksalamatian/CLionProjects/Curvature/cmake-build-release

# Include any dependencies generated for this target.
include CMakeFiles/Curvature.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/Curvature.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Curvature.dir/flags.make

CMakeFiles/Curvature.dir/main.cpp.o: CMakeFiles/Curvature.dir/flags.make
CMakeFiles/Curvature.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/ksalamatian/CLionProjects/Curvature/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Curvature.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Curvature.dir/main.cpp.o -c /Users/ksalamatian/CLionProjects/Curvature/main.cpp

CMakeFiles/Curvature.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Curvature.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ksalamatian/CLionProjects/Curvature/main.cpp > CMakeFiles/Curvature.dir/main.cpp.i

CMakeFiles/Curvature.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Curvature.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ksalamatian/CLionProjects/Curvature/main.cpp -o CMakeFiles/Curvature.dir/main.cpp.s

# Object files for target Curvature
Curvature_OBJECTS = \
"CMakeFiles/Curvature.dir/main.cpp.o"

# External object files for target Curvature
Curvature_EXTERNAL_OBJECTS =

Curvature: CMakeFiles/Curvature.dir/main.cpp.o
Curvature: CMakeFiles/Curvature.dir/build.make
Curvature: /opt/homebrew/lib/libboost_graph-mt.dylib
Curvature: /opt/homebrew/lib/libboost_iostreams-mt.dylib
Curvature: /opt/homebrew/lib/libboost_regex-mt.dylib
Curvature: CMakeFiles/Curvature.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/ksalamatian/CLionProjects/Curvature/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Curvature"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Curvature.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Curvature.dir/build: Curvature
.PHONY : CMakeFiles/Curvature.dir/build

CMakeFiles/Curvature.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Curvature.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Curvature.dir/clean

CMakeFiles/Curvature.dir/depend:
	cd /Users/ksalamatian/CLionProjects/Curvature/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ksalamatian/CLionProjects/Curvature /Users/ksalamatian/CLionProjects/Curvature /Users/ksalamatian/CLionProjects/Curvature/cmake-build-release /Users/ksalamatian/CLionProjects/Curvature/cmake-build-release /Users/ksalamatian/CLionProjects/Curvature/cmake-build-release/CMakeFiles/Curvature.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Curvature.dir/depend
