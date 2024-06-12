cmake_minimum_required(VERSION 3.10)
project(CV-SAS VERSION 0.1 LANGUAGES C CXX)

# Установите стандарт C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER clang)
endif()
if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
endif()

# Укажите путь к файлу toolchain vcpkg
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")

# Найдите пакеты
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(SqliteOrm CONFIG REQUIRED)

# Исполняемый файл
add_executable(${PROJECT_NAME} sources/main.cpp sources/User/User.cpp)

# Линковка библиотек
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} dlib::dlib Qt5::Widgets SQLite::SQLite3 sqlite_orm::sqlite_orm)