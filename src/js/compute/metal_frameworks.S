#if defined(__APPLE__)
.linker_option "-framework", "Foundation"
#ifdef __APPLE__
.linker_option "-framework", "Metal"
.linker_option "-framework", "MetalPerformanceShaders"
#endif
.linker_option "-framework", "Accelerate"
#endif
