# -*- cmake -*-

MACRO(ADD_TORCH_PACKAGE package src luasrc)
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
  INCLUDE_DIRECTORIES(${Torch_LUA_INCLUDE_DIR})

 ### C/C++ sources
 # As per CMake doc, macro arguments are not variables, so simple test syntax not working
  IF(NOT "${src}" STREQUAL "")

    if ("${src}" MATCHES "cu$" OR "${src}" MATCHES "cu;")
      CUDA_ADD_LIBRARY(${package} MODULE ${src})
      if(BUILD_STATIC)
        CUDA_ADD_LIBRARY(${package}_static STATIC ${src})
      endif()
    else()
      ADD_LIBRARY(${package} MODULE ${src})
      if(BUILD_STATIC)
        ADD_LIBRARY(${package}_static STATIC ${src})
      endif()
    endif()

    ### Torch packages supposes libraries prefix is "lib"
    SET_TARGET_PROPERTIES(${package} PROPERTIES
      PREFIX "lib"
      IMPORT_PREFIX "lib"
      INSTALL_NAME_DIR "@executable_path/${Torch_INSTALL_BIN2CPATH}")

    IF(APPLE)
      SET_TARGET_PROPERTIES(${package} PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup")
    ENDIF()

    if(BUILD_STATIC)
      SET_TARGET_PROPERTIES(${package}_static PROPERTIES
        COMPILE_FLAGS "-fPIC")
      SET_TARGET_PROPERTIES(${package}_static PROPERTIES
        PREFIX "lib" IMPORT_PREFIX "lib" OUTPUT_NAME "${package}")
    endif()

    INSTALL(TARGETS ${package}
      RUNTIME DESTINATION ${Torch_INSTALL_LUA_CPATH_SUBDIR}
      LIBRARY DESTINATION ${Torch_INSTALL_LUA_CPATH_SUBDIR})

  ENDIF(NOT "${src}" STREQUAL "")

  ### lua sources
  IF(NOT "${luasrc}" STREQUAL "")
    INSTALL(FILES ${luasrc}
      DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/${package})
  ENDIF(NOT "${luasrc}" STREQUAL "")

ENDMACRO(ADD_TORCH_PACKAGE)
