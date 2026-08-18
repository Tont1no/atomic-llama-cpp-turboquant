execute_process(
    COMMAND "${TEST_EXE}" --test-rs-seq 2
    RESULT_VARIABLE test_result
    OUTPUT_VARIABLE test_stdout
    ERROR_VARIABLE test_stderr
)

if (test_result EQUAL 0)
    message(FATAL_ERROR "--test-rs-seq 2 unexpectedly succeeded")
endif()

string(CONCAT test_output "${test_stdout}" "${test_stderr}")
if (NOT test_output MATCHES "error: --test-rs-seq must be an integer")
    message(FATAL_ERROR "unexpected validation output: ${test_output}")
endif()
