set(SOURCES_NON_LINEAR 
    main.cpp
)
set_source_files_properties(${SOURCES_NON_LINEAR} PROPERTIES LANGUAGE CUDA)

add_executable(non-linear ${SOURCES_NON_LINEAR} ${HEADERS_NON_LINEAR})

target_link_libraries(non-linear PRIVATE amrex treesap)