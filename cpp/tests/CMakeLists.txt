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

    target_link_libraries(${name} PUBLIC nnrt_cpp test_utils Python3::Python)
    target_link_libraries(${name} PRIVATE Catch2::Catch2)
    target_compile_options(${name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")
    target_link_options(${name} PRIVATE -no-pie)


    if (ADD_TEST_DEPENDENCIES)
        add_dependencies(${name} ${ADD_TEST_DEPENDENCIES})
    endif ()

    if (UNIX)
        target_link_libraries(${name} PUBLIC rt)
    endif ()

    add_test(NAME cmake_test_${name} COMMAND ${name})
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
    foreach (filename ${DOWNLOAD_DATA_EXTRACTED_NAMES})
        cmake_path(GET destination_path PARENT_PATH extracted_path)
        cmake_path(APPEND extracted_path ${DOWNLOAD_DATA_EXTRACTED_NAMES})
        list(APPEND byproduct_paths "${extracted_path}")
        list(LENGTH byproduct_paths byproduct_paths_length)
    endforeach ()

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
        test_utils/geometry.h
        test_utils/fitter_testing.cpp
        test_utils/fitter_testing.h)

    target_link_libraries(test_utils PUBLIC NNRT::3rdparty_libpng Python3::Module nnrt_cpp NNRT::3rdparty_fmt)
    target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/test_utils)

    # add downloaded test data
    nnrt_add_test_data_download_target(
        test_data_plane_and_cube_rendering
        URL https://algomorph.com/storage/reco/plane_and_cube_rendering_test_data.7z
        SHA256 0c43e50fc52f5c31b5b24aa826891db251ebee328edc55c279a78af3cce343c9
        DESTINATION plane_and_cube_rendering_test_data.7z
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

    nnrt_add_test_data_download_target(
        suzanne_and_bunny_res_4_2_rasterization_test
        URL https://algomorph.com/storage/reco/suzanne_and_bunny_res_4_2_rasterization_test.7z
        SHA256 9de26bbc3ad92e004671582c661ab854cca4df05192c68b3352bd014f55e248d
        DESTINATION suzanne_and_bunny_res_4_2_rasterization_test.7z
        EXTRACTED_NAMES
        arrays/suzanne_and_mesh_bunny_res4_2_pixel_face_indices.npy
        arrays/suzanne_and_mesh_bunny_res4_2_pixel_face_distances.npy
        arrays/suzanne_and_mesh_bunny_res4_2_pixel_depths.npy
        arrays/suzanne_and_mesh_bunny_res4_2_pixel_barycentric_coordinates.npy
    )

    nnrt_add_test_data_download_target(
        test_data_rendered_normals
        URL https://algomorph.com/storage/reco/interpolate_face_attribute_test_data.7z
        SHA256 4247cef37c9f4fe8bda3ec1d628b169618b891602e648bdc64807c12fe9eddde
        DESTINATION interpolate_face_attribute_test_data.7z
        EXTRACTED_NAMES
        arrays/mesh_64_bunny_array_rendered_normals.npy
        arrays/mesh_bunny_res2_rendered_normals.npy
        arrays/mesh_bunny_res4_rendered_normals.npy
    )

    nnrt_add_test_data_download_target(
        berlin_frames_28_29
        URL https://algomorph.com/storage/reco/berlin_frames_28-29.7z
        SHA256 bc2a96a4790756535b1cbb518b607df231d3f5c4290f41b36c28c7bd7ad96bc3
        DESTINATION berlin_frames_28-29.7z
        EXTRACTED_NAMES
        arrays/berlin_000028_faces.npy
        arrays/berlin_000028_node_coords.npy
        arrays/berlin_000028_node_point_indices.npy
        arrays/berlin_000028_vertices.npy
        images/berlin_000028_color.jpg
        images/berlin_000028_depth.png
        images/berlin_000028_mask.png
        images/berlin_000029_color.jpg
        images/berlin_000029_depth.png
        images/berlin_000029_mask.png
    )

    nnrt_add_test_data_download_target(
        test_data_block_sparse_arrowhead
        URL https://algomorph.com/storage/reco/block_sparse_arrowhead_test_data.7z
        SHA256 d9698c6a09194dae4587fdf3560d5e074e2fe7e68fb20c0247bba6a34b8472e3
        DESTINATION block_sparse_arrowhead_test_data.7z
        EXTRACTED_NAMES
        arrays/breadboard.npy
        arrays/U_diag_upper_left.npy
        arrays/U_upper.npy
        arrays/U_lower_right_dense.npy
        arrays/upper_column_block_lists.npy
        arrays/upper_column_block_counts.npy
        arrays/upper_row_block_lists.npy
        arrays/upper_row_block_counts.npy
    )


    ### add tests ###
    nnrt_add_test(test_graph_proc)
    nnrt_add_test(test_index_conversions)
    nnrt_add_test(test_array_comparisons)
    nnrt_add_test(test_heap)
    nnrt_add_test(test_kdtree)
    nnrt_add_test(test_graph_warp_field)
    nnrt_add_test(test_mesh_warping)
    nnrt_add_test(test_sampling)
    nnrt_add_test(test_non_rigid_surface_voxel_block_grid)
    nnrt_add_test(test_matmul3d)
    nnrt_add_test(test_normals_operations)
    nnrt_add_test(test_tensor_io)
    nnrt_add_test(test_voxel_block_grid_io)
    nnrt_add_test(test_unproject_3d_points)
    nnrt_add_test(test_extract_face_vertices)
    nnrt_add_test(test_deformable_mesh_fitter_basic)
    nnrt_add_test(test_deformable_mesh_fitter_intermediate)
    nnrt_add_test(test_deformable_mesh_fitter_advanced DEPENDENCIES berlin_frames_28_29)
    nnrt_add_test(test_flat_edge_shader)
    nnrt_add_test(test_triangle_fitter)
    nnrt_add_test(test_vertex_color_shader)
    nnrt_add_test(
        test_atomic_counter_array
        SOURCES
        test_atomic_counter_array/test_atomic_counter_array.cpp
        test_atomic_counter_array/test_atomic_counter_array.h
        test_atomic_counter_array/test_atomic_counter_array_impl.h
        test_atomic_counter_array/test_atomic_counter_array_CPU.cpp
        test_atomic_counter_array/test_atomic_counter_array_CUDA.cu
    )
    nnrt_add_test(test_block_cholesky)
    nnrt_add_test(test_linalg_block_routines)
    nnrt_add_test(test_linalg_matmul_block_sparse_row_wise)
    nnrt_add_test(test_rodrigues)
    nnrt_add_test(test_mesh_to_adjacency_array_conversion)
    nnrt_add_test(test_sorting)
    nnrt_add_test(test_rasterize DEPENDENCIES test_data_plane_and_cube_rendering test_data_mesh_bunny_res4 test_data_mesh_bunny_res2 test_data_mesh_64_bunny_array suzanne_and_bunny_res_4_2_rasterization_test)
    nnrt_add_test(test_interpolate_vertex_attributes DEPENDENCIES test_data_rendered_normals test_data_mesh_bunny_res4 test_data_mesh_bunny_res2 test_data_mesh_64_bunny_array)
    nnrt_add_test(test_block_sparse_arrowhead_solver DEPENDENCIES test_data_block_sparse_arrowhead)

endif ()
