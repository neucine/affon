// Profiling-only interposer. Reads public MTLCommandBuffer GPU timestamps after
// completion; never touches tensor data or changes command encoding/submission.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static void (*originalCommit)(id, SEL);
static FILE *profileFile;
static uint64_t clockNs(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
static void profileCommit(id<MTLCommandBuffer> command, SEL selector) {
    const uint64_t commitNs = clockNs();
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        double start = completed.GPUStartTime, end = completed.GPUEndTime;
        if (end > start && completed.status == MTLCommandBufferStatusCompleted) {
            flockfile(profileFile);
            fprintf(profileFile, "{\"commit_ns\":%llu,\"gpu_ns\":%.0f}\n", (unsigned long long)commitNs, (end-start)*1e9);
            fflush(profileFile);
            funlockfile(profileFile);
        }
    }];
    originalCommit(command, selector);
}
__attribute__((constructor)) static void installTiming(void) {
    const char *path = getenv("AFFON_GPU_PROFILE_PATH");
    if (!path) return;
    profileFile = fopen(path, "w");
    if (!profileFile) abort();
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        Class cls = object_getClass(command);
        Method method = class_getInstanceMethod(cls, @selector(commit));
        if (!method) abort();
        originalCommit = (void (*)(id,SEL))method_getImplementation(method);
        if (!class_addMethod(cls, @selector(commit), (IMP)profileCommit, method_getTypeEncoding(method)))
            method_setImplementation(method, (IMP)profileCommit);
    }
}
