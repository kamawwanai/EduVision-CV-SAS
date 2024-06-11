# Минимальная версия CMake
cmake_minimum_required(VERSION 3.15)

# Имя проекта
project(CV-SAS VERSION 0.1 LANGUAGES C CXX)

# Включаем генерацию файла компиляции
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER clang)
endif()
if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
endif()

# Стандарт C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Находим OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# dlib
find_package(dlib REQUIRED)
include_directories(${DLIB_INCLUDE_DIRS})

# Путь к SQLite
set(SQLITE3_DIR ${CMAKE_SOURCE_DIR}/external/sqlite3)
include_directories(${SQLITE3_DIR})

# Источники SQLite
set(SQLITE3_SRC ${SQLITE3_DIR}/sqlite3.c)

# Создание статической библиотеки SQLite
add_library(sqlite3 STATIC ${SQLITE3_SRC})
set_target_properties(sqlite3 PROPERTIES LINKER_LANGUAGE C)

# Указание пути к библиотеке SQLite3 для sqlite_orm
set(SQLite3_LIBRARY sqlite3)
set(SQLite3_INCLUDE_DIR ${SQLITE3_DIR})

# Укажите, что SQLite3 будет найден
set(SQLITE3_FOUND TRUE)

# Определите sqlite3 в проекте
add_subdirectory(external/sqlite_orm)

# Исполняемый файл
add_executable(${PROJECT_NAME} sources/main.cpp)

# Линковка библиотек
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} dlib::dlib sqlite3 sqlite_orm::sqlite_orm)

target_include_directories(${PROJECT_NAME} PRIVATE ${SQLITE3_DIR})

# Установки для Clang
if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <OBJECTS>")
    set(CMAKE_C_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()
