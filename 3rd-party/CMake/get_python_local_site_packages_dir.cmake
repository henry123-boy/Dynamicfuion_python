function(get_python_local_site_packages_dir _PYTHON_IN_PATH OUTPUT_VARIABLE)
    execute_process(
        COMMAND ${_PYTHON_IN_PATH} -m site --user-site
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(${OUTPUT_VARIABLE} ${OUTPUT} PARENT_SCOPE)
endfunction()