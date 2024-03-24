cmake_minimum_required(VERSION 3.8)
project(skeleton_extractor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# Build Python virtual environment
add_custom_target(
    poetry_virtualenv ALL
    COMMAND poetry install --no-root --no-ansi --no-interaction
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building ${CMAKE_PROJECT_NAME} virtualenv to ${CMAKE_SOURCE_DIR}... this may take some time"
)

install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}/
)

install(
    DIRECTORY
    skeleton_extractor
    DESTINATION lib/${PROJECT_NAME}
)

file(
    # "file" commands are executed at build time. The install directory won't
    # exist yet during a fresh build, which will cause the next CREATE_LINK to
    # fail.
    MAKE_DIRECTORY
    ${CMAKE_INSTALL_PREFIX}
)
file(
    CREATE_LINK
    ${CMAKE_SOURCE_DIR}/.venv
    ${CMAKE_INSTALL_PREFIX}/.venv
    SYMBOLIC
)

install(
    PROGRAMS
    scripts/python_entrypoint.py
    DESTINATION lib/${PROJECT_NAME}
)

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # comment the line when a copyright and license is added to all source files
#   set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # comment the line when this package is in a git repo and when
#   # a copyright and license is added to all source files
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

ament_package()
