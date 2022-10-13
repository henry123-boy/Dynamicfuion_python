include(ExternalProject)

set(lib_name Catch2)
set(main_lib_name Catch2Main)

ExternalProject_Add(
    ext_Catch2
    PREFIX Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.1.0.tar.gz
    URL_HASH SHA256=c252b2d9537e18046d8b82535069d2567f77043f8e644acf9a9fffc22ea6e6f7
    DOWNLOAD_DIR "${NNRT_THIRD_PARTY_DOWNLOAD_DIR}/Catch2"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_CXX_STANDARD=14
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${NNRT_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${NNRT_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${main_lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
)


ExternalProject_Get_Property(ext_Catch2 INSTALL_DIR)
set(CATCH2_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(CATCH2_LIB_DIR ${INSTALL_DIR}/${NNRT_INSTALL_LIB_DIR})
set(CATCH2_LIBRARIES ${lib_name} ${main_lib_name})