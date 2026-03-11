#pragma once

#if defined(_WIN32)
#    if defined(AXCORE_BUILD_DLL)
#        define AXCORE_API __declspec(dllexport)
#    else
#        define AXCORE_API __declspec(dllimport)
#    endif
#    define AXCORE_CALL __cdecl
#else
#    define AXCORE_API __attribute__((visibility("default")))
#    define AXCORE_CALL
#endif

#if defined(__cplusplus)
#    define AXCORE_EXTERN_C extern "C"
#else
#    define AXCORE_EXTERN_C
#endif
