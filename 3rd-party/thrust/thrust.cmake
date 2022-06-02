include(ExternalProject)

ExternalProject_Add(
    ext_thrust
    PREFIX thrust
    URL https://github.com/NVIDIA/thrust/archive/refs/tags/1.17.0-rc2.tar.gz
    URL_HASH SHA256=678e75f8d892a6ebfc9eb925bbdac3b98a5d8e4c1601c8a29d47f10cda6316d8
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/thrust"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_thrust SOURCE_DIR)
set(THRUST_INCLUDE_DIRS ${SOURCE_DIR})
