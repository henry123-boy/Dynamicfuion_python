# We need this file for cross-platform support on Windows
# For Ubuntu/Mac, we can simply do `pip install ${PYTHON_PACKAGE_DST_DIR}/pip_package/*.whl -U`

# Note: Since `make python-package` clears PYTHON_COMPILED_MODULE_DIR every time,
#       it is guaranteed that there is only one wheel in ${PYTHON_PACKAGE_DST_DIR}/pip_package/*.whl
file(GLOB WHEEL_FILE "${PIP_PACKAGE_DST_DIR}/nnrt*.whl")
execute_process(COMMAND pip install ${WHEEL_FILE} -U --user --force-reinstall)
