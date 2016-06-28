#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TH" for configuration "Release"
set_property(TARGET TH APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TH PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "/usr/lib/libblas.so;/usr/lib/liblapack.so;rt;m"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libTH.so"
  IMPORTED_SONAME_RELEASE "libTH.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS TH )
list(APPEND _IMPORT_CHECK_FILES_FOR_TH "${_IMPORT_PREFIX}/lib/libTH.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
