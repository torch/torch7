INSTALL(EXPORT torch-exports
  DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}"
  FILE "TorchExports.cmake")

CONFIGURE_FILE("cmake/TorchConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/TorchConfig.cmake" @ONLY)

INSTALL(
  FILES
  "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/TorchConfig.cmake"
  "cmake/TorchPathsInit.cmake"
  "cmake/TorchPackage.cmake"
  "cmake/TorchWrap.cmake"
  DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")
