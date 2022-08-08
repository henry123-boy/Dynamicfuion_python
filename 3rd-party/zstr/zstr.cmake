include(FetchContent)

FetchContent_Declare(
    ext_zstr_headers
    GIT_REPOSITORY https://github.com/mateidavid/zstr
    GIT_TAG 85a5bd5283e9eb310ceba156271af0ec61d4cb17 # Aug 8, 2022
)

FetchContent_MakeAvailable(ext_zstr_headers)

add_library(ext_zstr ${CMAKE_CURRENT_LIST_DIR}/zstr.cpp)
target_link_libraries(ext_zstr PUBLIC zstr::zstr)