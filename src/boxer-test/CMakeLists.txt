set(SOURCES_BOXER_TEST 
    main.cpp
    AMRContainer.cpp
)
set_source_files_properties(${SOURCES_BOXER_TEST} PROPERTIES LANGUAGE CUDA)

add_executable(boxer-test ${SOURCES_BOXER_TEST} ${HEADERS_BOXERT_TEST})

set_target_properties(boxer-test PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

target_link_libraries(boxer-test PRIVATE amrex)