set(ALL_FILES_EXIST TRUE)
foreach(file BYPRODUCTS)
    if(NOT EXISTS ${file})
        set(ALL_FILES_EXIST FALSE)
        break()
    endif()
endforeach()
if(NOT ALL_FILES_EXIST)
    foreach(file BYPRODUCTS)
        file(REMOVE ${file})
    endforeach()
    file(DOWNLOAD ${URL} ${DESTINATION} EXPECTED_HASH SHA256=${SHA256})
    cmake_path(SET download_path ${DESTINATION})
    cmake_path(GET download_path PARENT_PATH DOWNLOAD_DIRECTORY)
    file(ARCHIVE_EXTRACT INPUT ${download_path} DESTINATION ${DOWNLOAD_DIRECTORY})
endif()
