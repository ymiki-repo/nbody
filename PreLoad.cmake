# set build target
find_program (NINJA "ninja")
if (NOT NINJA)
  find_program (NINJA_BUILD "ninja-build")
endif (NOT NINJA)
if (NINJA OR NINJA_BUILD)
  set (CMAKE_GENERATOR "Ninja" CACHE INTERNAL "" FORCE)
endif (NINJA OR NINJA_BUILD)
