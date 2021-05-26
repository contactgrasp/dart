get_filename_component(SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

# include the exported targets
include(${SELF_DIR}/HandFitterTargets.cmake)

get_filename_component(handfitter_INCLUDE_DIRS
        "${SELF_DIR}/../../include/handfitter" ABSOLUTE)
set(handfitter_LIBS dart handfitter)