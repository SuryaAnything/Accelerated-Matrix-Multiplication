# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/surya/CLionProjects/Matrix-Mult

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/surya/CLionProjects/Matrix-Mult/cmake-build-release

# Include any dependencies generated for this target.
include CMakeFiles/Matrix_Mult.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Matrix_Mult.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Matrix_Mult.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Matrix_Mult.dir/flags.make

CMakeFiles/Matrix_Mult.dir/main.c.o: CMakeFiles/Matrix_Mult.dir/flags.make
CMakeFiles/Matrix_Mult.dir/main.c.o: ../main.c
CMakeFiles/Matrix_Mult.dir/main.c.o: CMakeFiles/Matrix_Mult.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/surya/CLionProjects/Matrix-Mult/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/Matrix_Mult.dir/main.c.o"
	gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/Matrix_Mult.dir/main.c.o -MF CMakeFiles/Matrix_Mult.dir/main.c.o.d -o CMakeFiles/Matrix_Mult.dir/main.c.o -c /mnt/c/Users/surya/CLionProjects/Matrix-Mult/main.c

CMakeFiles/Matrix_Mult.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Matrix_Mult.dir/main.c.i"
	gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/surya/CLionProjects/Matrix-Mult/main.c > CMakeFiles/Matrix_Mult.dir/main.c.i

CMakeFiles/Matrix_Mult.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Matrix_Mult.dir/main.c.s"
	gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/surya/CLionProjects/Matrix-Mult/main.c -o CMakeFiles/Matrix_Mult.dir/main.c.s

CMakeFiles/Matrix_Mult.dir/tensor.c.o: CMakeFiles/Matrix_Mult.dir/flags.make
CMakeFiles/Matrix_Mult.dir/tensor.c.o: ../tensor.c
CMakeFiles/Matrix_Mult.dir/tensor.c.o: CMakeFiles/Matrix_Mult.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/surya/CLionProjects/Matrix-Mult/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/Matrix_Mult.dir/tensor.c.o"
	gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/Matrix_Mult.dir/tensor.c.o -MF CMakeFiles/Matrix_Mult.dir/tensor.c.o.d -o CMakeFiles/Matrix_Mult.dir/tensor.c.o -c /mnt/c/Users/surya/CLionProjects/Matrix-Mult/tensor.c

CMakeFiles/Matrix_Mult.dir/tensor.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Matrix_Mult.dir/tensor.c.i"
	gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/surya/CLionProjects/Matrix-Mult/tensor.c > CMakeFiles/Matrix_Mult.dir/tensor.c.i

CMakeFiles/Matrix_Mult.dir/tensor.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Matrix_Mult.dir/tensor.c.s"
	gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/surya/CLionProjects/Matrix-Mult/tensor.c -o CMakeFiles/Matrix_Mult.dir/tensor.c.s

# Object files for target Matrix_Mult
Matrix_Mult_OBJECTS = \
"CMakeFiles/Matrix_Mult.dir/main.c.o" \
"CMakeFiles/Matrix_Mult.dir/tensor.c.o"

# External object files for target Matrix_Mult
Matrix_Mult_EXTERNAL_OBJECTS =

Matrix_Mult: CMakeFiles/Matrix_Mult.dir/main.c.o
Matrix_Mult: CMakeFiles/Matrix_Mult.dir/tensor.c.o
Matrix_Mult: CMakeFiles/Matrix_Mult.dir/build.make
Matrix_Mult: CMakeFiles/Matrix_Mult.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/surya/CLionProjects/Matrix-Mult/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable Matrix_Mult"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Matrix_Mult.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Matrix_Mult.dir/build: Matrix_Mult
.PHONY : CMakeFiles/Matrix_Mult.dir/build

CMakeFiles/Matrix_Mult.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Matrix_Mult.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Matrix_Mult.dir/clean

CMakeFiles/Matrix_Mult.dir/depend:
	cd /mnt/c/Users/surya/CLionProjects/Matrix-Mult/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/surya/CLionProjects/Matrix-Mult /mnt/c/Users/surya/CLionProjects/Matrix-Mult /mnt/c/Users/surya/CLionProjects/Matrix-Mult/cmake-build-release /mnt/c/Users/surya/CLionProjects/Matrix-Mult/cmake-build-release /mnt/c/Users/surya/CLionProjects/Matrix-Mult/cmake-build-release/CMakeFiles/Matrix_Mult.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Matrix_Mult.dir/depend

