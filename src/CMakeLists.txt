# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
  
#Create a variable called linalg_SOURCES containing all .cpp files:

add_subdirectory(main)
add_subdirectory(reader)

set(mklSparse_Src 
    ${main_Src}
    ${reader_Src}
    PARENT_SCOPE
    )

