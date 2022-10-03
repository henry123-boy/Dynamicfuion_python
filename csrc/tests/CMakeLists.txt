function(nnrt_add_test)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs DEPENDENCIES SOURCES)
    cmake_parse_arguments(ADD_TEST "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})
    list(LENGTH ARGN ARGC)

    if (${ARGC} EQUAL 0)
        message(FATAL_ERROR "nnrt_add_test needs at least a single positional argument.")
    else ()
        list(GET ARGN 0 name)
    endif ()

    if (ADD_TEST_SOURCES)
        add_executable(${name} ${ADD_TEST_SOURCES})
    else ()
        add_executable(${name} ${name}.cpp)
    endif ()

    target_link_libraries(${name} PUBLIC nnrt_cpp test_utils)
    target_link_libraries(${name} PUBLIC Python3::Python ${CATCH2_TARGET})
    add_test(NAME ${name} COMMAND ${name})
    if (ADD_TEST_DEPENDENCIES)
        add_dependencies(${name} ${ADD_TEST_DEPENDENCIES})
    endif ()
endfunction()

function(nnrt_add_test_data_download_target)
    set(options)
    set(oneValueArgs URL SHA256 DESTINATION)
    set(multiValueArgs EXTRACTED_NAMES)
    cmake_parse_arguments(DOWNLOAD_DATA "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    list(LENGTH ARGN ARGC)
    if (${ARGC} EQUAL 0)
        message(FATAL_ERROR "nnrt_download_test_data needs at least a single positional argument.")
    else ()
        list(GET ARGN 0 target_name)
    endif ()

    if (NOT DOWNLOAD_DATA_URL)
        message(FATAL_ERROR "nnrt_download_test_data needs the URL one-value argument.")
    endif ()
    if (NOT DOWNLOAD_DATA_SHA256)
        message(FATAL_ERROR "nnrt_download_test_data needs the SHA256 one-value argument.")
    endif ()
    if (NOT DOWNLOAD_DATA_DESTINATION)
        message(FATAL_ERROR "nnrt_download_test_data needs the DESTINATION one-value argument.")
    endif ()
        if (NOT DOWNLOAD_DATA_EXTRACTED_NAMES)
        message(FATAL_ERROR "nnrt_download_test_data needs the EXTRACTED_NAMES multi-value argument.")
    endif ()

    cmake_path(SET destination_path ${GENERATED_TEST_DATA_DIRECTORY})
    cmake_path(APPEND destination_path ${DOWNLOAD_DATA_DESTINATION})
    set(byproduct_paths)
    foreach(filename DOWNLOAD_DATA_EXTRACTED_NAMES)
        cmake_path(GET destination_path PARENT_PATH extracted_path)
        cmake_path(APPEND extracted_path ${DOWNLOAD_DATA_EXTRACTED_NAMES})
        list(APPEND byproduct_paths ${extracted_path})
    endforeach()

    list(APPEND byproduct_paths ${destination_path})



    add_custom_target(
        ${target_name}
        BYPRODUCTS ${byproduct_paths}
        COMMAND ${CMAKE_COMMAND}
        -DBYPRODUCTS=${byproduct_paths}
        -DURL=${DOWNLOAD_DATA_URL}
        -DDESTINATION=${destination_path}
        -DSHA256=${DOWNLOAD_DATA_SHA256}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/DownloadAndExtractFile.cmake
    )
endfunction()

if (BUILD_CPP_TESTS)
    ### test utilities library ###

    set(STATIC_TEST_DATA_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_data/)
    set(GENERATED_TEST_DATA_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_data/)
    file(MAKE_DIRECTORY ${GENERATED_TEST_DATA_DIRECTORY})
    file(MAKE_DIRECTORY ${GENERATED_TEST_DATA_DIRECTORY}/meshes)
    configure_file(test_utils/test_data_paths.hpp.in test_utils/test_data_paths.hpp @ONLY)

    add_library(
        test_utils STATIC
        test_utils/test_utils.hpp
        test_utils/test_utils.tpp
        test_utils/test_utils.cpp
        test_utils/compile_time_string_concatenation.hpp
        test_utils/numpy_image.h
        test_utils/image_io/load_png.h
        test_utils/image_io/load_png.cpp
        test_utils/pixel.h
        test_utils/image_view.h
        test_utils/image_io/dng_shared.h
        test_utils/image_io/load_dng.h
        test_utils/image_io/load_dng.cpp
        test_utils/enable_if.h
        test_utils/entropy_range_decoder.h
        test_utils/entropy_range_decoder.cpp
        test_utils/uintn.h
        test_utils/entropy_decoder_model.h
        test_utils/float_details.h
        test_utils/vectorstream.h
        test_utils/algs.h
        test_utils/image_io/save_png.h
        test_utils/image_io/save_png.cpp
        test_utils/array2d.h
        test_utils/image_io/save_dng.h
        test_utils/entropy_encoder_model.h
        test_utils/entropy_range_encoder.h
        test_utils/entropy_range_encoder.cpp
        test_utils/geometry.cpp
        test_utils/geometry.h)

    target_link_libraries(test_utils PUBLIC NNRT::3rdparty_libpng Python3::Module nnrt_cpp NNRT::3rdparty_fmt)
    target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/test_utils)


    ### add tests ###
    nnrt_add_test(test_graph_proc)
    nnrt_add_test(test_index_conversions)
    nnrt_add_test(test_array_comparisons)
    nnrt_add_test(test_heap)
    nnrt_add_test(test_kdtree)
    nnrt_add_test(test_graph_warp_field)
    nnrt_add_test(test_mesh_warping)
    nnrt_add_test(test_downsampling)
    nnrt_add_test(test_non_rigid_surface_voxel_block_grid)
    nnrt_add_test(test_matmul3d)
    nnrt_add_test(test_normals_operations)
    nnrt_add_test(test_tensor_io)
    nnrt_add_test(test_voxel_block_grid_io)
    nnrt_add_test(test_unproject_3d_points)
    nnrt_add_test(test_extract_face_vertices)

    nnrt_add_test_data_download_target(
        test_data_plane_and_cube_rendering
        URL https://algomorph.com/storage/reco/plane_and_cube_rendering_test_data.7z
        SHA256 529944b4e5714f4fdfb5dfd3965ac4ff53d3ea306698fdb2083ce4cbf51a76f9
        DESTINATION mesh_64_bunny_array.7z
        EXTRACTED_NAMES
        arrays/cube_0_pixel_barycentric_coordinates.npy
        arrays/cube_0_pixel_depths.npy
        arrays/cube_0_pixel_face_distances.npy
        arrays/cube_0_pixel_face_indices.npy
        arrays/plane_0_pixel_barycentric_coordinates.npy
        arrays/plane_0_pixel_depths.npy
        arrays/plane_0_pixel_face_distances.npy
        arrays/plane_0_pixel_face_indices.npy
    )

    nnrt_add_test_data_download_target(
        test_data_mesh_bunny_res4
        URL https://algomorph.com/storage/reco/mesh_bunny_res4.7z
        SHA256 8db8ae2812b012271116a64fe873fb0d9073936c1f99c4615693df6fa1b5563e
        DESTINATION mesh_bunny_res4.7z
        EXTRACTED_NAMES
        meshes/mesh_bunny_res4.ply
        arrays/mesh_bunny_res4_pixel_barycentric_coordinates.npy
        arrays/mesh_bunny_res4_pixel_depths.npy
        arrays/mesh_bunny_res4_pixel_face_distances.npy
        arrays/mesh_bunny_res4_pixel_face_indices.npy
        images/mesh_bunny_res4_render_preview.png
    )

    nnrt_add_test_data_download_target(
        test_data_mesh_bunny_res2
        URL https://algomorph.com/storage/reco/mesh_bunny_res2.7z
        SHA256 5086803a46be1db566fcbc784cde962a7282c72e72f09703a2f7b739b1a9241e
        DESTINATION mesh_bunny_res2.7z
        EXTRACTED_NAMES
        meshes/mesh_bunny_res2.ply
        arrays/mesh_bunny_res2_pixel_barycentric_coordinates.npy
        arrays/mesh_bunny_res2_pixel_depths.npy
        arrays/mesh_bunny_res2_pixel_face_distances.npy
        arrays/mesh_bunny_res2_pixel_face_indices.npy
        images/mesh_bunny_res2_render_preview.png
    )

    nnrt_add_test_data_download_target(
        test_data_mesh_64_bunny_array
        URL https://algomorph.com/storage/reco/mesh_64_bunny_array.7z
        SHA256 529944b4e5714f4fdfb5dfd3965ac4ff53d3ea306698fdb2083ce4cbf51a76f9
        DESTINATION mesh_64_bunny_array.7z
        EXTRACTED_NAMES
        meshes/mesh_64_bunny_array.ply
        arrays/mesh_64_bunny_array_pixel_barycentric_coordinates.npy
        arrays/mesh_64_bunny_array_pixel_depths.npy
        arrays/mesh_64_bunny_array_pixel_face_distances.npy
        arrays/mesh_64_bunny_array_pixel_face_indices.npy
        images/mesh_64_bunny_array_render_preview.png
    )


    nnrt_add_test(test_rasterize_mesh DEPENDENCIES test_data_plane_and_cube_rendering test_data_mesh_bunny_res4 test_data_mesh_bunny_res2 test_data_mesh_64_bunny_array)

endif ()
