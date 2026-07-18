#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#import <objc/runtime.h>
#include "affon_metal_metallib.h"

extern size_t affon_config_metal_threadgroup_size(void);

@interface AffonMetalContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLComputePipelineState> addF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> subF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> mulF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> divF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> eqF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> ltF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> gtF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> addI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> subI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> mulI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> divI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> eqI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> ltI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> gtI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> muladdInplaceF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> axpyInplaceF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> subInplaceF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> subInplaceI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> scaleInplaceF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> fillF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> dotF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> dotI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> adamStepF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> contiguousF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> contiguousI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> contiguousSignedF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> contiguousSignedI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> addBroadcastF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> subBroadcastF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> mulBroadcastF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> divBroadcastF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> eqBroadcastF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> ltBroadcastF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> gtBroadcastF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> expF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> logF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> sqrtF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> absF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> absI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> signF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> signI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> geluF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> geluGradF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> negF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> negI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reluF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reluI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> sigmoidF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> siluF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> tanhF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> clampF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> clampI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> clampGradF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> softmaxF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> softmaxNdF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> logSoftmaxNllNdF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> crossEntropyIndexedF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> crossEntropyIndexedTransposedF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> crossEntropyIndexedBackwardF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> layerNormNdF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> rmsNormNdF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> addLayerNormNdF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> whereF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> whereI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> whereF32I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> whereI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> maskedFillI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> maskedFillI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> whereBroadcastF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> whereBroadcastI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> maskedFillBroadcastI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> maskedFillBroadcastI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> scatterAddF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> scatterAddI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceSumF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceSumI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceMeanF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceMinF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceMinI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceMaxF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceMaxI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceArgminF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceArgmaxF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceArgminI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceArgmaxI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceArgminI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceArgmaxI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceVarianceF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceStdF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisSumF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisSumI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisMeanF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisMinF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisMinI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisMaxF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisMaxI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisVarianceF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisStdF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisArgminF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisArgmaxF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisArgminI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisArgmaxI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisArgminI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisArgmaxI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisNdF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisNdI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisNdArgI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisNdArgI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matmulF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matmulI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matmulStridedF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matmulStridedI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matmulAddF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matmulAddI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> matmulAddGeluF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionScoresF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> embeddingI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> embeddingI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> unaryChainF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> unaryChainI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> binaryThenUnaryChainF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> binaryThenUnaryChainI64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> castF32F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> castI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> castF32I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> castI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> validateCastIndexF32I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> gatherF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> gatherI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> gatherI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexSelectAxis0F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexSelectAxis0I64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexSelectAxis0I64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexSelectI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexSelectI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> indexSelectAxis0ScatterAddF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> topkF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> topkI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> topkDirI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> topkNdI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> topkDirI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> topkNdI64I64Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> oneHotI64F32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceSumParallelF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceMeanParallelF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisSumParallelF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> reduceAxisMeanParallelF32Pipeline;
@end

@implementation AffonMetalContext
@end

@interface AffonMetalBuffer : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@end

@implementation AffonMetalBuffer
@end

static const uint32_t affon_max_broadcast_dims = 8;
static char affon_metal_last_error_message[256] = "uninitialized";

// Opaque Metal buffer handles follow one ownership rule:
// - affon_metal_buffer_create returns a +1 retained wrapper handle
// - affon_metal_buffer_destroy consumes that +1 exactly once
// - all other bridge entrypoints only borrow the wrapper for the duration of the call
static inline void *affon_metal_retain_buffer_wrapper(AffonMetalBuffer *wrapper) {
    return (__bridge_retained void *)wrapper;
}

static inline AffonMetalBuffer *affon_metal_transfer_buffer_wrapper(void *handle) {
    return (__bridge_transfer AffonMetalBuffer *)handle;
}

static inline AffonMetalBuffer *affon_metal_borrow_buffer_wrapper(void *handle) {
    return (__bridge AffonMetalBuffer *)handle;
}

static inline id<MTLBuffer> affon_metal_borrow_buffer(void *handle) {
    AffonMetalBuffer *wrapper = affon_metal_borrow_buffer_wrapper(handle);
    return wrapper ? wrapper.buffer : nil;
}

static void affon_set_last_error(NSString *message) {
    if (message == nil) {
        snprintf(affon_metal_last_error_message, sizeof(affon_metal_last_error_message), "%s", "unknown");
        return;
    }
    const char *utf8 = message.UTF8String;
    if (utf8 == NULL) {
        snprintf(affon_metal_last_error_message, sizeof(affon_metal_last_error_message), "%s", "non-utf8 error");
        return;
    }
    snprintf(affon_metal_last_error_message, sizeof(affon_metal_last_error_message), "%s", utf8);
}

static id<MTLBuffer> affon_metal_create_status_buffer(AffonMetalContext *ctx) {
    if (ctx == nil || ctx.device == nil) return nil;
    id<MTLBuffer> buffer = [ctx.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        affon_set_last_error(@"status buffer allocation failed");
        return nil;
    }
    *((uint32_t *)buffer.contents) = 0;
    return buffer;
}

static bool affon_metal_status_buffer_failed(id<MTLBuffer> statusBuffer) {
    if (statusBuffer == nil) return true;
    return *((uint32_t *)statusBuffer.contents) != 0;
}

static inline int affon_metal_finish_command_buffer(id<MTLCommandBuffer> commandBuffer, int command_error_code) {
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return commandBuffer.error ? command_error_code : 0;
}

static inline int affon_metal_finish_command_buffer_with_status(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLBuffer> statusBuffer,
    int command_error_code,
    int status_error_code
) {
    const int rc = affon_metal_finish_command_buffer(commandBuffer, command_error_code);
    if (rc != 0) return rc;
    return affon_metal_status_buffer_failed(statusBuffer) ? status_error_code : 0;
}

int affon_metal_matmul_offset_f32(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    size_t a_offset_bytes,
    size_t b_offset_bytes,
    size_t out_offset_bytes,
    size_t m,
    size_t n,
    size_t k
);
int affon_metal_matmul_offset_many_f32(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    const size_t *a_offsets_bytes,
    const size_t *b_offsets_bytes,
    const size_t *out_offsets_bytes,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k
);
int affon_metal_matmul_offset_i64(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    size_t a_offset_bytes,
    size_t b_offset_bytes,
    size_t out_offset_bytes,
    size_t m,
    size_t n,
    size_t k
);
int affon_metal_matmul_offset_many_i64(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    const size_t *a_offsets_bytes,
    const size_t *b_offsets_bytes,
    const size_t *out_offsets_bytes,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k
);

static AffonMetalContext *affon_get_context(void) {
    static AffonMetalContext *ctx = nil;
    if (ctx != nil) return ctx;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        affon_set_last_error(@"MTLCreateSystemDefaultDevice returned nil");
        return nil;
    }

    NSError *error = nil;
    dispatch_data_t data = dispatch_data_create(
        affon_metal_metallib_data,
        affon_metal_metallib_data_len,
        dispatch_get_main_queue(),
        ^{}
    );
    if (!data) {
        affon_set_last_error(@"dispatch_data_create failed");
        return nil;
    }

    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    if (!library || error) {
        affon_set_last_error(error.localizedDescription ?: @"newLibraryWithData failed");
        return nil;
    }
    id<MTLFunction> addFn = [library newFunctionWithName:@"affon_add_f32"];
    if (!addFn) {
        affon_set_last_error(@"missing function affon_add_f32");
        return nil;
    }
    id<MTLComputePipelineState> addPipeline = [device newComputePipelineStateWithFunction:addFn error:&error];
    if (!addPipeline || error) {
        affon_set_last_error(error.localizedDescription ?: @"add pipeline creation failed");
        return nil;
    }
    id<MTLFunction> subFn = [library newFunctionWithName:@"affon_sub_f32"];
    if (!subFn) {
        affon_set_last_error(@"missing function affon_sub_f32");
        return nil;
    }
    id<MTLComputePipelineState> subPipeline = [device newComputePipelineStateWithFunction:subFn error:&error];
    if (!subPipeline || error) {
        affon_set_last_error(error.localizedDescription ?: @"sub pipeline creation failed");
        return nil;
    }
    id<MTLFunction> mulFn = [library newFunctionWithName:@"affon_mul_f32"];
    if (!mulFn) {
        affon_set_last_error(@"missing function affon_mul_f32");
        return nil;
    }
    id<MTLComputePipelineState> mulPipeline = [device newComputePipelineStateWithFunction:mulFn error:&error];
    if (!mulPipeline || error) {
        affon_set_last_error(error.localizedDescription ?: @"mul pipeline creation failed");
        return nil;
    }
    id<MTLFunction> divFn = [library newFunctionWithName:@"affon_div_f32"];
    if (!divFn) {
        affon_set_last_error(@"missing function affon_div_f32");
        return nil;
    }
    id<MTLComputePipelineState> divPipeline = [device newComputePipelineStateWithFunction:divFn error:&error];
    if (!divPipeline || error) {
        affon_set_last_error(error.localizedDescription ?: @"div pipeline creation failed");
        return nil;
    }
    id<MTLFunction> eqFn = [library newFunctionWithName:@"affon_eq_f32"];
    if (!eqFn) return nil;
    id<MTLComputePipelineState> eqPipeline = [device newComputePipelineStateWithFunction:eqFn error:&error];
    if (!eqPipeline || error) return nil;
    id<MTLFunction> ltFn = [library newFunctionWithName:@"affon_lt_f32"];
    if (!ltFn) return nil;
    id<MTLComputePipelineState> ltPipeline = [device newComputePipelineStateWithFunction:ltFn error:&error];
    if (!ltPipeline || error) return nil;
    id<MTLFunction> gtFn = [library newFunctionWithName:@"affon_gt_f32"];
    if (!gtFn) return nil;
    id<MTLComputePipelineState> gtPipeline = [device newComputePipelineStateWithFunction:gtFn error:&error];
    if (!gtPipeline || error) return nil;
    id<MTLFunction> addI64Fn = [library newFunctionWithName:@"affon_add_i64"];
    if (!addI64Fn) return nil;
    id<MTLComputePipelineState> addI64Pipeline = [device newComputePipelineStateWithFunction:addI64Fn error:&error];
    if (!addI64Pipeline || error) return nil;
    id<MTLFunction> subI64Fn = [library newFunctionWithName:@"affon_sub_i64"];
    if (!subI64Fn) return nil;
    id<MTLComputePipelineState> subI64Pipeline = [device newComputePipelineStateWithFunction:subI64Fn error:&error];
    if (!subI64Pipeline || error) return nil;
    id<MTLFunction> mulI64Fn = [library newFunctionWithName:@"affon_mul_i64"];
    if (!mulI64Fn) return nil;
    id<MTLComputePipelineState> mulI64Pipeline = [device newComputePipelineStateWithFunction:mulI64Fn error:&error];
    if (!mulI64Pipeline || error) return nil;
    id<MTLFunction> divI64Fn = [library newFunctionWithName:@"affon_div_i64"];
    if (!divI64Fn) return nil;
    id<MTLComputePipelineState> divI64Pipeline = [device newComputePipelineStateWithFunction:divI64Fn error:&error];
    if (!divI64Pipeline || error) return nil;
    id<MTLFunction> eqI64Fn = [library newFunctionWithName:@"affon_eq_i64"];
    if (!eqI64Fn) return nil;
    id<MTLComputePipelineState> eqI64Pipeline = [device newComputePipelineStateWithFunction:eqI64Fn error:&error];
    if (!eqI64Pipeline || error) return nil;
    id<MTLFunction> ltI64Fn = [library newFunctionWithName:@"affon_lt_i64"];
    if (!ltI64Fn) return nil;
    id<MTLComputePipelineState> ltI64Pipeline = [device newComputePipelineStateWithFunction:ltI64Fn error:&error];
    if (!ltI64Pipeline || error) return nil;
    id<MTLFunction> gtI64Fn = [library newFunctionWithName:@"affon_gt_i64"];
    if (!gtI64Fn) return nil;
    id<MTLComputePipelineState> gtI64Pipeline = [device newComputePipelineStateWithFunction:gtI64Fn error:&error];
    if (!gtI64Pipeline || error) return nil;
    id<MTLFunction> muladdInplaceFn = [library newFunctionWithName:@"affon_muladd_inplace_f32"];
    if (!muladdInplaceFn) return nil;
    id<MTLComputePipelineState> muladdInplacePipeline = [device newComputePipelineStateWithFunction:muladdInplaceFn error:&error];
    if (!muladdInplacePipeline || error) return nil;
    id<MTLFunction> axpyInplaceFn = [library newFunctionWithName:@"affon_axpy_inplace_f32"];
    if (!axpyInplaceFn) return nil;
    id<MTLComputePipelineState> axpyInplacePipeline = [device newComputePipelineStateWithFunction:axpyInplaceFn error:&error];
    if (!axpyInplacePipeline || error) return nil;
    id<MTLFunction> subInplaceFn = [library newFunctionWithName:@"affon_sub_inplace_f32"];
    if (!subInplaceFn) return nil;
    id<MTLComputePipelineState> subInplacePipeline = [device newComputePipelineStateWithFunction:subInplaceFn error:&error];
    if (!subInplacePipeline || error) return nil;
    id<MTLFunction> subInplaceI64Fn = [library newFunctionWithName:@"affon_sub_inplace_i64"];
    if (!subInplaceI64Fn) return nil;
    id<MTLComputePipelineState> subInplaceI64Pipeline = [device newComputePipelineStateWithFunction:subInplaceI64Fn error:&error];
    if (!subInplaceI64Pipeline || error) return nil;
    id<MTLFunction> scaleInplaceFn = [library newFunctionWithName:@"affon_scale_inplace_f32"];
    if (!scaleInplaceFn) return nil;
    id<MTLComputePipelineState> scaleInplacePipeline = [device newComputePipelineStateWithFunction:scaleInplaceFn error:&error];
    if (!scaleInplacePipeline || error) return nil;
    id<MTLFunction> fillFn = [library newFunctionWithName:@"affon_fill_f32"];
    if (!fillFn) {
        affon_set_last_error(@"missing function affon_fill_f32");
        return nil;
    }
    id<MTLComputePipelineState> fillPipeline = [device newComputePipelineStateWithFunction:fillFn error:&error];
    if (!fillPipeline || error) {
        affon_set_last_error(error.localizedDescription ?: @"fill pipeline creation failed");
        return nil;
    }
    id<MTLFunction> dotFn = [library newFunctionWithName:@"affon_dot_f32"];
    if (!dotFn) {
        affon_set_last_error(@"missing function affon_dot_f32");
        return nil;
    }
    id<MTLComputePipelineState> dotPipeline = [device newComputePipelineStateWithFunction:dotFn error:&error];
    if (!dotPipeline || error) {
        affon_set_last_error(error.localizedDescription ?: @"dot pipeline creation failed");
        return nil;
    }
    id<MTLFunction> dotI64Fn = [library newFunctionWithName:@"affon_dot_i64"];
    if (!dotI64Fn) {
        affon_set_last_error(@"missing function affon_dot_i64");
        return nil;
    }
    id<MTLComputePipelineState> dotI64Pipeline = [device newComputePipelineStateWithFunction:dotI64Fn error:&error];
    if (!dotI64Pipeline || error) {
        affon_set_last_error(error.localizedDescription ?: @"dot i64 pipeline creation failed");
        return nil;
    }
    id<MTLFunction> adamStepFn = [library newFunctionWithName:@"affon_adam_step_f32"];
    if (!adamStepFn) return nil;
    id<MTLComputePipelineState> adamStepPipeline = [device newComputePipelineStateWithFunction:adamStepFn error:&error];
    if (!adamStepPipeline || error) return nil;
    id<MTLFunction> contiguousFn = [library newFunctionWithName:@"affon_contiguous_f32"];
    if (!contiguousFn) return nil;
    id<MTLComputePipelineState> contiguousPipeline = [device newComputePipelineStateWithFunction:contiguousFn error:&error];
    if (!contiguousPipeline || error) return nil;
    id<MTLFunction> contiguousI64Fn = [library newFunctionWithName:@"affon_contiguous_i64"];
    if (!contiguousI64Fn) return nil;
    id<MTLComputePipelineState> contiguousI64Pipeline = [device newComputePipelineStateWithFunction:contiguousI64Fn error:&error];
    if (!contiguousI64Pipeline || error) return nil;
    id<MTLFunction> contiguousSignedFn = [library newFunctionWithName:@"affon_contiguous_signed_f32"];
    if (!contiguousSignedFn) return nil;
    id<MTLComputePipelineState> contiguousSignedPipeline = [device newComputePipelineStateWithFunction:contiguousSignedFn error:&error];
    if (!contiguousSignedPipeline || error) return nil;
    id<MTLFunction> contiguousSignedI64Fn = [library newFunctionWithName:@"affon_contiguous_signed_i64"];
    if (!contiguousSignedI64Fn) return nil;
    id<MTLComputePipelineState> contiguousSignedI64Pipeline = [device newComputePipelineStateWithFunction:contiguousSignedI64Fn error:&error];
    if (!contiguousSignedI64Pipeline || error) return nil;
    id<MTLFunction> addBroadcastFn = [library newFunctionWithName:@"affon_add_broadcast_f32"];
    if (!addBroadcastFn) return nil;
    id<MTLComputePipelineState> addBroadcastPipeline = [device newComputePipelineStateWithFunction:addBroadcastFn error:&error];
    if (!addBroadcastPipeline || error) return nil;
    id<MTLFunction> subBroadcastFn = [library newFunctionWithName:@"affon_sub_broadcast_f32"];
    if (!subBroadcastFn) return nil;
    id<MTLComputePipelineState> subBroadcastPipeline = [device newComputePipelineStateWithFunction:subBroadcastFn error:&error];
    if (!subBroadcastPipeline || error) return nil;
    id<MTLFunction> mulBroadcastFn = [library newFunctionWithName:@"affon_mul_broadcast_f32"];
    if (!mulBroadcastFn) return nil;
    id<MTLComputePipelineState> mulBroadcastPipeline = [device newComputePipelineStateWithFunction:mulBroadcastFn error:&error];
    if (!mulBroadcastPipeline || error) return nil;
    id<MTLFunction> divBroadcastFn = [library newFunctionWithName:@"affon_div_broadcast_f32"];
    if (!divBroadcastFn) return nil;
    id<MTLComputePipelineState> divBroadcastPipeline = [device newComputePipelineStateWithFunction:divBroadcastFn error:&error];
    if (!divBroadcastPipeline || error) return nil;
    id<MTLFunction> eqBroadcastFn = [library newFunctionWithName:@"affon_eq_broadcast_f32"];
    if (!eqBroadcastFn) return nil;
    id<MTLComputePipelineState> eqBroadcastPipeline = [device newComputePipelineStateWithFunction:eqBroadcastFn error:&error];
    if (!eqBroadcastPipeline || error) return nil;
    id<MTLFunction> ltBroadcastFn = [library newFunctionWithName:@"affon_lt_broadcast_f32"];
    if (!ltBroadcastFn) return nil;
    id<MTLComputePipelineState> ltBroadcastPipeline = [device newComputePipelineStateWithFunction:ltBroadcastFn error:&error];
    if (!ltBroadcastPipeline || error) return nil;
    id<MTLFunction> gtBroadcastFn = [library newFunctionWithName:@"affon_gt_broadcast_f32"];
    if (!gtBroadcastFn) return nil;
    id<MTLComputePipelineState> gtBroadcastPipeline = [device newComputePipelineStateWithFunction:gtBroadcastFn error:&error];
    if (!gtBroadcastPipeline || error) return nil;
    id<MTLFunction> expFn = [library newFunctionWithName:@"affon_exp_f32"];
    if (!expFn) return nil;
    id<MTLComputePipelineState> expPipeline = [device newComputePipelineStateWithFunction:expFn error:&error];
    if (!expPipeline || error) return nil;
    id<MTLFunction> logFn = [library newFunctionWithName:@"affon_log_f32"];
    if (!logFn) return nil;
    id<MTLComputePipelineState> logPipeline = [device newComputePipelineStateWithFunction:logFn error:&error];
    if (!logPipeline || error) return nil;
    id<MTLFunction> sqrtFn = [library newFunctionWithName:@"affon_sqrt_f32"];
    if (!sqrtFn) return nil;
    id<MTLComputePipelineState> sqrtPipeline = [device newComputePipelineStateWithFunction:sqrtFn error:&error];
    if (!sqrtPipeline || error) return nil;
    id<MTLFunction> absFn = [library newFunctionWithName:@"affon_abs_f32"];
    if (!absFn) return nil;
    id<MTLComputePipelineState> absPipeline = [device newComputePipelineStateWithFunction:absFn error:&error];
    if (!absPipeline || error) return nil;
    id<MTLFunction> absI64Fn = [library newFunctionWithName:@"affon_abs_i64"];
    if (!absI64Fn) return nil;
    id<MTLComputePipelineState> absI64Pipeline = [device newComputePipelineStateWithFunction:absI64Fn error:&error];
    if (!absI64Pipeline || error) return nil;
    id<MTLFunction> signFn = [library newFunctionWithName:@"affon_sign_f32"];
    if (!signFn) return nil;
    id<MTLComputePipelineState> signPipeline = [device newComputePipelineStateWithFunction:signFn error:&error];
    if (!signPipeline || error) return nil;
    id<MTLFunction> signI64Fn = [library newFunctionWithName:@"affon_sign_i64"];
    if (!signI64Fn) return nil;
    id<MTLComputePipelineState> signI64Pipeline = [device newComputePipelineStateWithFunction:signI64Fn error:&error];
    if (!signI64Pipeline || error) return nil;
    id<MTLFunction> geluFn = [library newFunctionWithName:@"affon_gelu_f32"];
    if (!geluFn) return nil;
    id<MTLComputePipelineState> geluPipeline = [device newComputePipelineStateWithFunction:geluFn error:&error];
    if (!geluPipeline || error) return nil;
    id<MTLFunction> geluGradFn = [library newFunctionWithName:@"affon_gelu_grad_f32"];
    if (!geluGradFn) return nil;
    id<MTLComputePipelineState> geluGradPipeline = [device newComputePipelineStateWithFunction:geluGradFn error:&error];
    if (!geluGradPipeline || error) return nil;
    id<MTLFunction> negFn = [library newFunctionWithName:@"affon_neg_f32"];
    if (!negFn) return nil;
    id<MTLComputePipelineState> negPipeline = [device newComputePipelineStateWithFunction:negFn error:&error];
    if (!negPipeline || error) return nil;
    id<MTLFunction> negI64Fn = [library newFunctionWithName:@"affon_neg_i64"];
    if (!negI64Fn) return nil;
    id<MTLComputePipelineState> negI64Pipeline = [device newComputePipelineStateWithFunction:negI64Fn error:&error];
    if (!negI64Pipeline || error) return nil;
    id<MTLFunction> reluFn = [library newFunctionWithName:@"affon_relu_f32"];
    if (!reluFn) return nil;
    id<MTLComputePipelineState> reluPipeline = [device newComputePipelineStateWithFunction:reluFn error:&error];
    if (!reluPipeline || error) return nil;
    id<MTLFunction> reluI64Fn = [library newFunctionWithName:@"affon_relu_i64"];
    if (!reluI64Fn) return nil;
    id<MTLComputePipelineState> reluI64Pipeline = [device newComputePipelineStateWithFunction:reluI64Fn error:&error];
    if (!reluI64Pipeline || error) return nil;
    id<MTLFunction> sigmoidFn = [library newFunctionWithName:@"affon_sigmoid_f32"];
    if (!sigmoidFn) return nil;
    id<MTLComputePipelineState> sigmoidPipeline = [device newComputePipelineStateWithFunction:sigmoidFn error:&error];
    if (!sigmoidPipeline || error) return nil;
    id<MTLFunction> siluFn = [library newFunctionWithName:@"affon_silu_f32"];
    if (!siluFn) return nil;
    id<MTLComputePipelineState> siluPipeline = [device newComputePipelineStateWithFunction:siluFn error:&error];
    if (!siluPipeline || error) return nil;
    id<MTLFunction> tanhFn = [library newFunctionWithName:@"affon_tanh_f32"];
    if (!tanhFn) return nil;
    id<MTLComputePipelineState> tanhPipeline = [device newComputePipelineStateWithFunction:tanhFn error:&error];
    if (!tanhPipeline || error) return nil;
    id<MTLFunction> clampFn = [library newFunctionWithName:@"affon_clamp_f32"];
    if (!clampFn) return nil;
    id<MTLComputePipelineState> clampPipeline = [device newComputePipelineStateWithFunction:clampFn error:&error];
    if (!clampPipeline || error) return nil;
    id<MTLFunction> clampI64Fn = [library newFunctionWithName:@"affon_clamp_i64"];
    if (!clampI64Fn) return nil;
    id<MTLComputePipelineState> clampI64Pipeline = [device newComputePipelineStateWithFunction:clampI64Fn error:&error];
    if (!clampI64Pipeline || error) return nil;
    id<MTLFunction> clampGradFn = [library newFunctionWithName:@"affon_clamp_grad_f32"];
    if (!clampGradFn) return nil;
    id<MTLComputePipelineState> clampGradPipeline = [device newComputePipelineStateWithFunction:clampGradFn error:&error];
    if (!clampGradPipeline || error) return nil;
    id<MTLFunction> softmaxFn = [library newFunctionWithName:@"affon_softmax_f32"];
    if (!softmaxFn) return nil;
    id<MTLComputePipelineState> softmaxPipeline = [device newComputePipelineStateWithFunction:softmaxFn error:&error];
    if (!softmaxPipeline || error) return nil;
    id<MTLFunction> softmaxNdFn = [library newFunctionWithName:@"affon_softmax_nd_f32"];
    if (!softmaxNdFn) return nil;
    id<MTLComputePipelineState> softmaxNdPipeline = [device newComputePipelineStateWithFunction:softmaxNdFn error:&error];
    if (!softmaxNdPipeline || error) return nil;
    id<MTLFunction> logSoftmaxNllNdFn = [library newFunctionWithName:@"affon_log_softmax_nll_nd_f32"];
    if (!logSoftmaxNllNdFn) return nil;
    id<MTLComputePipelineState> logSoftmaxNllNdPipeline = [device newComputePipelineStateWithFunction:logSoftmaxNllNdFn error:&error];
    if (!logSoftmaxNllNdPipeline || error) return nil;
    id<MTLFunction> crossEntropyIndexedFn = [library newFunctionWithName:@"affon_cross_entropy_indexed_f32"];
    if (!crossEntropyIndexedFn) return nil;
    id<MTLComputePipelineState> crossEntropyIndexedPipeline = [device newComputePipelineStateWithFunction:crossEntropyIndexedFn error:&error];
    if (!crossEntropyIndexedPipeline || error) return nil;
    id<MTLFunction> crossEntropyIndexedTransposedFn = [library newFunctionWithName:@"affon_cross_entropy_indexed_transposed_f32"];
    if (!crossEntropyIndexedTransposedFn) return nil;
    id<MTLComputePipelineState> crossEntropyIndexedTransposedPipeline = [device newComputePipelineStateWithFunction:crossEntropyIndexedTransposedFn error:&error];
    if (!crossEntropyIndexedTransposedPipeline || error) return nil;
    id<MTLFunction> crossEntropyIndexedBackwardFn = [library newFunctionWithName:@"affon_cross_entropy_indexed_backward_f32"];
    if (!crossEntropyIndexedBackwardFn) return nil;
    id<MTLComputePipelineState> crossEntropyIndexedBackwardPipeline = [device newComputePipelineStateWithFunction:crossEntropyIndexedBackwardFn error:&error];
    if (!crossEntropyIndexedBackwardPipeline || error) return nil;
    id<MTLFunction> layerNormNdFn = [library newFunctionWithName:@"affon_layer_norm_nd_f32"];
    if (!layerNormNdFn) return nil;
    id<MTLComputePipelineState> layerNormNdPipeline = [device newComputePipelineStateWithFunction:layerNormNdFn error:&error];
    if (!layerNormNdPipeline || error) return nil;
    id<MTLFunction> rmsNormNdFn = [library newFunctionWithName:@"affon_rms_norm_nd_f32"];
    if (!rmsNormNdFn) return nil;
    id<MTLComputePipelineState> rmsNormNdPipeline = [device newComputePipelineStateWithFunction:rmsNormNdFn error:&error];
    if (!rmsNormNdPipeline || error) return nil;
    id<MTLFunction> addLayerNormNdFn = [library newFunctionWithName:@"affon_add_layer_norm_nd_f32"];
    if (!addLayerNormNdFn) return nil;
    id<MTLComputePipelineState> addLayerNormNdPipeline = [device newComputePipelineStateWithFunction:addLayerNormNdFn error:&error];
    if (!addLayerNormNdPipeline || error) return nil;
    id<MTLFunction> whereFn = [library newFunctionWithName:@"affon_where_f32"];
    if (!whereFn) return nil;
    id<MTLComputePipelineState> wherePipeline = [device newComputePipelineStateWithFunction:whereFn error:&error];
    if (!wherePipeline || error) return nil;
    id<MTLFunction> whereI64Fn = [library newFunctionWithName:@"affon_where_i64_f32"];
    if (!whereI64Fn) return nil;
    id<MTLComputePipelineState> whereI64Pipeline = [device newComputePipelineStateWithFunction:whereI64Fn error:&error];
    if (!whereI64Pipeline || error) return nil;
    id<MTLFunction> whereF32I64Fn = [library newFunctionWithName:@"affon_where_f32_i64"];
    if (!whereF32I64Fn) return nil;
    id<MTLComputePipelineState> whereF32I64Pipeline = [device newComputePipelineStateWithFunction:whereF32I64Fn error:&error];
    if (!whereF32I64Pipeline || error) return nil;
    id<MTLFunction> whereI64I64Fn = [library newFunctionWithName:@"affon_where_i64_i64"];
    if (!whereI64I64Fn) return nil;
    id<MTLComputePipelineState> whereI64I64Pipeline = [device newComputePipelineStateWithFunction:whereI64I64Fn error:&error];
    if (!whereI64I64Pipeline || error) return nil;
    id<MTLFunction> maskedFillFn = [library newFunctionWithName:@"affon_masked_fill_i64_f32"];
    if (!maskedFillFn) return nil;
    id<MTLComputePipelineState> maskedFillPipeline = [device newComputePipelineStateWithFunction:maskedFillFn error:&error];
    if (!maskedFillPipeline || error) return nil;
    id<MTLFunction> maskedFillI64I64Fn = [library newFunctionWithName:@"affon_masked_fill_i64_i64"];
    if (!maskedFillI64I64Fn) return nil;
    id<MTLComputePipelineState> maskedFillI64I64Pipeline = [device newComputePipelineStateWithFunction:maskedFillI64I64Fn error:&error];
    if (!maskedFillI64I64Pipeline || error) return nil;
    id<MTLFunction> whereBroadcastFn = [library newFunctionWithName:@"affon_where_broadcast_f32"];
    if (!whereBroadcastFn) return nil;
    id<MTLComputePipelineState> whereBroadcastPipeline = [device newComputePipelineStateWithFunction:whereBroadcastFn error:&error];
    if (!whereBroadcastPipeline || error) return nil;
    id<MTLFunction> whereBroadcastI64F32Fn = [library newFunctionWithName:@"affon_where_broadcast_i64_f32"];
    if (!whereBroadcastI64F32Fn) return nil;
    id<MTLComputePipelineState> whereBroadcastI64F32Pipeline = [device newComputePipelineStateWithFunction:whereBroadcastI64F32Fn error:&error];
    if (!whereBroadcastI64F32Pipeline || error) return nil;
    id<MTLFunction> maskedFillBroadcastF32Fn = [library newFunctionWithName:@"affon_masked_fill_broadcast_i64_f32"];
    if (!maskedFillBroadcastF32Fn) return nil;
    id<MTLComputePipelineState> maskedFillBroadcastF32Pipeline = [device newComputePipelineStateWithFunction:maskedFillBroadcastF32Fn error:&error];
    if (!maskedFillBroadcastF32Pipeline || error) return nil;
    id<MTLFunction> maskedFillBroadcastI64Fn = [library newFunctionWithName:@"affon_masked_fill_broadcast_i64_i64"];
    if (!maskedFillBroadcastI64Fn) return nil;
    id<MTLComputePipelineState> maskedFillBroadcastI64Pipeline = [device newComputePipelineStateWithFunction:maskedFillBroadcastI64Fn error:&error];
    if (!maskedFillBroadcastI64Pipeline || error) return nil;
    id<MTLFunction> scatterAddFn = [library newFunctionWithName:@"affon_scatter_add_f32"];
    if (!scatterAddFn) return nil;
    id<MTLComputePipelineState> scatterAddPipeline = [device newComputePipelineStateWithFunction:scatterAddFn error:&error];
    if (!scatterAddPipeline || error) return nil;
    id<MTLFunction> scatterAddI64Fn = [library newFunctionWithName:@"affon_scatter_add_i64_f32"];
    if (!scatterAddI64Fn) return nil;
    id<MTLComputePipelineState> scatterAddI64Pipeline = [device newComputePipelineStateWithFunction:scatterAddI64Fn error:&error];
    if (!scatterAddI64Pipeline || error) return nil;
    id<MTLFunction> reduceSumFn = [library newFunctionWithName:@"affon_reduce_sum_f32"];
    if (!reduceSumFn) return nil;
    id<MTLComputePipelineState> reduceSumPipeline = [device newComputePipelineStateWithFunction:reduceSumFn error:&error];
    if (!reduceSumPipeline || error) return nil;
    id<MTLFunction> reduceSumI64Fn = [library newFunctionWithName:@"affon_reduce_sum_i64"];
    if (!reduceSumI64Fn) return nil;
    id<MTLComputePipelineState> reduceSumI64Pipeline = [device newComputePipelineStateWithFunction:reduceSumI64Fn error:&error];
    if (!reduceSumI64Pipeline || error) return nil;
    id<MTLFunction> reduceMeanFn = [library newFunctionWithName:@"affon_reduce_mean_f32"];
    if (!reduceMeanFn) return nil;
    id<MTLComputePipelineState> reduceMeanPipeline = [device newComputePipelineStateWithFunction:reduceMeanFn error:&error];
    if (!reduceMeanPipeline || error) return nil;
    id<MTLFunction> reduceMinFn = [library newFunctionWithName:@"affon_reduce_min_f32"];
    if (!reduceMinFn) return nil;
    id<MTLComputePipelineState> reduceMinPipeline = [device newComputePipelineStateWithFunction:reduceMinFn error:&error];
    if (!reduceMinPipeline || error) return nil;
    id<MTLFunction> reduceMinI64Fn = [library newFunctionWithName:@"affon_reduce_min_i64"];
    if (!reduceMinI64Fn) return nil;
    id<MTLComputePipelineState> reduceMinI64Pipeline = [device newComputePipelineStateWithFunction:reduceMinI64Fn error:&error];
    if (!reduceMinI64Pipeline || error) return nil;
    id<MTLFunction> reduceMaxFn = [library newFunctionWithName:@"affon_reduce_max_f32"];
    if (!reduceMaxFn) return nil;
    id<MTLComputePipelineState> reduceMaxPipeline = [device newComputePipelineStateWithFunction:reduceMaxFn error:&error];
    if (!reduceMaxPipeline || error) return nil;
    id<MTLFunction> reduceMaxI64Fn = [library newFunctionWithName:@"affon_reduce_max_i64"];
    if (!reduceMaxI64Fn) return nil;
    id<MTLComputePipelineState> reduceMaxI64Pipeline = [device newComputePipelineStateWithFunction:reduceMaxI64Fn error:&error];
    if (!reduceMaxI64Pipeline || error) return nil;
    id<MTLFunction> reduceArgminFn = [library newFunctionWithName:@"affon_reduce_argmin_f32"];
    if (!reduceArgminFn) return nil;
    id<MTLComputePipelineState> reduceArgminPipeline = [device newComputePipelineStateWithFunction:reduceArgminFn error:&error];
    if (!reduceArgminPipeline || error) return nil;
    id<MTLFunction> reduceArgmaxFn = [library newFunctionWithName:@"affon_reduce_argmax_f32"];
    if (!reduceArgmaxFn) return nil;
    id<MTLComputePipelineState> reduceArgmaxPipeline = [device newComputePipelineStateWithFunction:reduceArgmaxFn error:&error];
    if (!reduceArgmaxPipeline || error) return nil;
    id<MTLFunction> reduceArgminI64Fn = [library newFunctionWithName:@"affon_reduce_argmin_i64_f32"];
    if (!reduceArgminI64Fn) return nil;
    id<MTLComputePipelineState> reduceArgminI64Pipeline = [device newComputePipelineStateWithFunction:reduceArgminI64Fn error:&error];
    if (!reduceArgminI64Pipeline || error) return nil;
    id<MTLFunction> reduceArgmaxI64Fn = [library newFunctionWithName:@"affon_reduce_argmax_i64_f32"];
    if (!reduceArgmaxI64Fn) return nil;
    id<MTLComputePipelineState> reduceArgmaxI64Pipeline = [device newComputePipelineStateWithFunction:reduceArgmaxI64Fn error:&error];
    if (!reduceArgmaxI64Pipeline || error) return nil;
    id<MTLFunction> reduceArgminI64I64Fn = [library newFunctionWithName:@"affon_reduce_argmin_i64_i64"];
    if (!reduceArgminI64I64Fn) return nil;
    id<MTLComputePipelineState> reduceArgminI64I64Pipeline = [device newComputePipelineStateWithFunction:reduceArgminI64I64Fn error:&error];
    if (!reduceArgminI64I64Pipeline || error) return nil;
    id<MTLFunction> reduceArgmaxI64I64Fn = [library newFunctionWithName:@"affon_reduce_argmax_i64_i64"];
    if (!reduceArgmaxI64I64Fn) return nil;
    id<MTLComputePipelineState> reduceArgmaxI64I64Pipeline = [device newComputePipelineStateWithFunction:reduceArgmaxI64I64Fn error:&error];
    if (!reduceArgmaxI64I64Pipeline || error) return nil;
    id<MTLFunction> reduceVarianceFn = [library newFunctionWithName:@"affon_reduce_variance_f32"];
    if (!reduceVarianceFn) return nil;
    id<MTLComputePipelineState> reduceVariancePipeline = [device newComputePipelineStateWithFunction:reduceVarianceFn error:&error];
    if (!reduceVariancePipeline || error) return nil;
    id<MTLFunction> reduceStdFn = [library newFunctionWithName:@"affon_reduce_std_f32"];
    if (!reduceStdFn) return nil;
    id<MTLComputePipelineState> reduceStdPipeline = [device newComputePipelineStateWithFunction:reduceStdFn error:&error];
    if (!reduceStdPipeline || error) return nil;
    id<MTLFunction> reduceAxisSumFn = [library newFunctionWithName:@"affon_reduce_axis_sum_f32"];
    if (!reduceAxisSumFn) return nil;
    id<MTLComputePipelineState> reduceAxisSumPipeline = [device newComputePipelineStateWithFunction:reduceAxisSumFn error:&error];
    if (!reduceAxisSumPipeline || error) return nil;
    id<MTLFunction> reduceAxisSumI64Fn = [library newFunctionWithName:@"affon_reduce_axis_sum_i64"];
    if (!reduceAxisSumI64Fn) return nil;
    id<MTLComputePipelineState> reduceAxisSumI64Pipeline = [device newComputePipelineStateWithFunction:reduceAxisSumI64Fn error:&error];
    if (!reduceAxisSumI64Pipeline || error) return nil;
    id<MTLFunction> reduceAxisMeanFn = [library newFunctionWithName:@"affon_reduce_axis_mean_f32"];
    if (!reduceAxisMeanFn) return nil;
    id<MTLComputePipelineState> reduceAxisMeanPipeline = [device newComputePipelineStateWithFunction:reduceAxisMeanFn error:&error];
    if (!reduceAxisMeanPipeline || error) return nil;
    id<MTLFunction> reduceAxisMinFn = [library newFunctionWithName:@"affon_reduce_axis_min_f32"];
    if (!reduceAxisMinFn) return nil;
    id<MTLComputePipelineState> reduceAxisMinPipeline = [device newComputePipelineStateWithFunction:reduceAxisMinFn error:&error];
    if (!reduceAxisMinPipeline || error) return nil;
    id<MTLFunction> reduceAxisMinI64Fn = [library newFunctionWithName:@"affon_reduce_axis_min_i64"];
    if (!reduceAxisMinI64Fn) return nil;
    id<MTLComputePipelineState> reduceAxisMinI64Pipeline = [device newComputePipelineStateWithFunction:reduceAxisMinI64Fn error:&error];
    if (!reduceAxisMinI64Pipeline || error) return nil;
    id<MTLFunction> reduceAxisMaxFn = [library newFunctionWithName:@"affon_reduce_axis_max_f32"];
    if (!reduceAxisMaxFn) return nil;
    id<MTLComputePipelineState> reduceAxisMaxPipeline = [device newComputePipelineStateWithFunction:reduceAxisMaxFn error:&error];
    if (!reduceAxisMaxPipeline || error) return nil;
    id<MTLFunction> reduceAxisMaxI64Fn = [library newFunctionWithName:@"affon_reduce_axis_max_i64"];
    if (!reduceAxisMaxI64Fn) return nil;
    id<MTLComputePipelineState> reduceAxisMaxI64Pipeline = [device newComputePipelineStateWithFunction:reduceAxisMaxI64Fn error:&error];
    if (!reduceAxisMaxI64Pipeline || error) return nil;
    id<MTLFunction> reduceAxisVarianceFn = [library newFunctionWithName:@"affon_reduce_axis_variance_f32"];
    if (!reduceAxisVarianceFn) return nil;
    id<MTLComputePipelineState> reduceAxisVariancePipeline = [device newComputePipelineStateWithFunction:reduceAxisVarianceFn error:&error];
    if (!reduceAxisVariancePipeline || error) return nil;
    id<MTLFunction> reduceAxisStdFn = [library newFunctionWithName:@"affon_reduce_axis_std_f32"];
    if (!reduceAxisStdFn) return nil;
    id<MTLComputePipelineState> reduceAxisStdPipeline = [device newComputePipelineStateWithFunction:reduceAxisStdFn error:&error];
    if (!reduceAxisStdPipeline || error) return nil;
    id<MTLFunction> reduceAxisArgminFn = [library newFunctionWithName:@"affon_reduce_axis_argmin_f32"];
    if (!reduceAxisArgminFn) return nil;
    id<MTLComputePipelineState> reduceAxisArgminPipeline = [device newComputePipelineStateWithFunction:reduceAxisArgminFn error:&error];
    if (!reduceAxisArgminPipeline || error) return nil;
    id<MTLFunction> reduceAxisArgmaxFn = [library newFunctionWithName:@"affon_reduce_axis_argmax_f32"];
    if (!reduceAxisArgmaxFn) return nil;
    id<MTLComputePipelineState> reduceAxisArgmaxPipeline = [device newComputePipelineStateWithFunction:reduceAxisArgmaxFn error:&error];
    if (!reduceAxisArgmaxPipeline || error) return nil;
    id<MTLFunction> reduceAxisArgminI64Fn = [library newFunctionWithName:@"affon_reduce_axis_argmin_i64_f32"];
    if (!reduceAxisArgminI64Fn) return nil;
    id<MTLComputePipelineState> reduceAxisArgminI64Pipeline = [device newComputePipelineStateWithFunction:reduceAxisArgminI64Fn error:&error];
    if (!reduceAxisArgminI64Pipeline || error) return nil;
    id<MTLFunction> reduceAxisArgmaxI64Fn = [library newFunctionWithName:@"affon_reduce_axis_argmax_i64_f32"];
    if (!reduceAxisArgmaxI64Fn) return nil;
    id<MTLComputePipelineState> reduceAxisArgmaxI64Pipeline = [device newComputePipelineStateWithFunction:reduceAxisArgmaxI64Fn error:&error];
    if (!reduceAxisArgmaxI64Pipeline || error) return nil;
    id<MTLFunction> reduceAxisArgminI64I64Fn = [library newFunctionWithName:@"affon_reduce_axis_argmin_i64_i64"];
    if (!reduceAxisArgminI64I64Fn) return nil;
    id<MTLComputePipelineState> reduceAxisArgminI64I64Pipeline = [device newComputePipelineStateWithFunction:reduceAxisArgminI64I64Fn error:&error];
    if (!reduceAxisArgminI64I64Pipeline || error) return nil;
    id<MTLFunction> reduceAxisArgmaxI64I64Fn = [library newFunctionWithName:@"affon_reduce_axis_argmax_i64_i64"];
    if (!reduceAxisArgmaxI64I64Fn) return nil;
    id<MTLComputePipelineState> reduceAxisArgmaxI64I64Pipeline = [device newComputePipelineStateWithFunction:reduceAxisArgmaxI64I64Fn error:&error];
    if (!reduceAxisArgmaxI64I64Pipeline || error) return nil;
    id<MTLFunction> reduceAxisNdFn = [library newFunctionWithName:@"affon_reduce_axis_nd_f32"];
    if (!reduceAxisNdFn) return nil;
    id<MTLComputePipelineState> reduceAxisNdPipeline = [device newComputePipelineStateWithFunction:reduceAxisNdFn error:&error];
    if (!reduceAxisNdPipeline || error) return nil;
    id<MTLFunction> reduceAxisNdI64Fn = [library newFunctionWithName:@"affon_reduce_axis_nd_i64"];
    if (!reduceAxisNdI64Fn) return nil;
    id<MTLComputePipelineState> reduceAxisNdI64Pipeline = [device newComputePipelineStateWithFunction:reduceAxisNdI64Fn error:&error];
    if (!reduceAxisNdI64Pipeline || error) return nil;
    id<MTLFunction> reduceAxisNdArgFn = [library newFunctionWithName:@"affon_reduce_axis_nd_arg_i64_f32"];
    if (!reduceAxisNdArgFn) return nil;
    id<MTLComputePipelineState> reduceAxisNdArgPipeline = [device newComputePipelineStateWithFunction:reduceAxisNdArgFn error:&error];
    if (!reduceAxisNdArgPipeline || error) return nil;
    id<MTLFunction> reduceAxisNdArgI64I64Fn = [library newFunctionWithName:@"affon_reduce_axis_nd_arg_i64_i64"];
    if (!reduceAxisNdArgI64I64Fn) return nil;
    id<MTLComputePipelineState> reduceAxisNdArgI64I64Pipeline = [device newComputePipelineStateWithFunction:reduceAxisNdArgI64I64Fn error:&error];
    if (!reduceAxisNdArgI64I64Pipeline || error) return nil;
    id<MTLFunction> matmulFn = [library newFunctionWithName:@"affon_matmul_f32"];
    if (!matmulFn) return nil;
    id<MTLComputePipelineState> matmulPipeline = [device newComputePipelineStateWithFunction:matmulFn error:&error];
    if (!matmulPipeline || error) return nil;
    id<MTLFunction> matmulI64Fn = [library newFunctionWithName:@"affon_matmul_i64"];
    if (!matmulI64Fn) return nil;
    id<MTLComputePipelineState> matmulI64Pipeline = [device newComputePipelineStateWithFunction:matmulI64Fn error:&error];
    if (!matmulI64Pipeline || error) return nil;
    id<MTLFunction> matmulStridedFn = [library newFunctionWithName:@"affon_matmul_strided_f32"];
    if (!matmulStridedFn) return nil;
    id<MTLComputePipelineState> matmulStridedPipeline = [device newComputePipelineStateWithFunction:matmulStridedFn error:&error];
    if (!matmulStridedPipeline || error) return nil;
    id<MTLFunction> matmulStridedI64Fn = [library newFunctionWithName:@"affon_matmul_strided_i64"];
    if (!matmulStridedI64Fn) return nil;
    id<MTLComputePipelineState> matmulStridedI64Pipeline = [device newComputePipelineStateWithFunction:matmulStridedI64Fn error:&error];
    if (!matmulStridedI64Pipeline || error) return nil;
    id<MTLFunction> matmulAddFn = [library newFunctionWithName:@"affon_matmul_add_f32"];
    if (!matmulAddFn) return nil;
    id<MTLComputePipelineState> matmulAddPipeline = [device newComputePipelineStateWithFunction:matmulAddFn error:&error];
    if (!matmulAddPipeline || error) return nil;
    id<MTLFunction> matmulAddI64Fn = [library newFunctionWithName:@"affon_matmul_add_i64"];
    if (!matmulAddI64Fn) return nil;
    id<MTLComputePipelineState> matmulAddI64Pipeline = [device newComputePipelineStateWithFunction:matmulAddI64Fn error:&error];
    if (!matmulAddI64Pipeline || error) return nil;
    id<MTLFunction> matmulAddGeluFn = [library newFunctionWithName:@"affon_matmul_add_gelu_f32"];
    if (!matmulAddGeluFn) return nil;
    id<MTLComputePipelineState> matmulAddGeluPipeline = [device newComputePipelineStateWithFunction:matmulAddGeluFn error:&error];
    if (!matmulAddGeluPipeline || error) return nil;
    id<MTLFunction> attentionScoresFn = [library newFunctionWithName:@"affon_attention_scores_f32"];
    if (!attentionScoresFn) return nil;
    id<MTLComputePipelineState> attentionScoresPipeline = [device newComputePipelineStateWithFunction:attentionScoresFn error:&error];
    if (!attentionScoresPipeline || error) return nil;
    id<MTLFunction> embeddingI64F32Fn = [library newFunctionWithName:@"affon_embedding_i64_f32"];
    if (!embeddingI64F32Fn) return nil;
    id<MTLComputePipelineState> embeddingI64F32Pipeline = [device newComputePipelineStateWithFunction:embeddingI64F32Fn error:&error];
    if (!embeddingI64F32Pipeline || error) return nil;
    id<MTLFunction> embeddingI64I64Fn = [library newFunctionWithName:@"affon_embedding_i64_i64"];
    if (!embeddingI64I64Fn) return nil;
    id<MTLComputePipelineState> embeddingI64I64Pipeline = [device newComputePipelineStateWithFunction:embeddingI64I64Fn error:&error];
    if (!embeddingI64I64Pipeline || error) return nil;
    id<MTLFunction> unaryChainFn = [library newFunctionWithName:@"affon_unary_chain_f32"];
    if (!unaryChainFn) return nil;
    id<MTLComputePipelineState> unaryChainPipeline = [device newComputePipelineStateWithFunction:unaryChainFn error:&error];
    if (!unaryChainPipeline || error) return nil;
    id<MTLFunction> unaryChainI64Fn = [library newFunctionWithName:@"affon_unary_chain_i64"];
    if (!unaryChainI64Fn) return nil;
    id<MTLComputePipelineState> unaryChainI64Pipeline = [device newComputePipelineStateWithFunction:unaryChainI64Fn error:&error];
    if (!unaryChainI64Pipeline || error) return nil;
    id<MTLFunction> binaryThenUnaryChainFn = [library newFunctionWithName:@"affon_binary_then_unary_chain_f32"];
    if (!binaryThenUnaryChainFn) return nil;
    id<MTLComputePipelineState> binaryThenUnaryChainPipeline = [device newComputePipelineStateWithFunction:binaryThenUnaryChainFn error:&error];
    if (!binaryThenUnaryChainPipeline || error) return nil;
    id<MTLFunction> binaryThenUnaryChainI64Fn = [library newFunctionWithName:@"affon_binary_then_unary_chain_i64"];
    if (!binaryThenUnaryChainI64Fn) return nil;
    id<MTLComputePipelineState> binaryThenUnaryChainI64Pipeline = [device newComputePipelineStateWithFunction:binaryThenUnaryChainI64Fn error:&error];
    if (!binaryThenUnaryChainI64Pipeline || error) return nil;
    id<MTLFunction> castF32F32Fn = [library newFunctionWithName:@"affon_cast_f32_f32"];
    if (!castF32F32Fn) return nil;
    id<MTLComputePipelineState> castF32F32Pipeline = [device newComputePipelineStateWithFunction:castF32F32Fn error:&error];
    if (!castF32F32Pipeline || error) return nil;
    id<MTLFunction> castI64I64Fn = [library newFunctionWithName:@"affon_cast_i64_i64"];
    if (!castI64I64Fn) return nil;
    id<MTLComputePipelineState> castI64I64Pipeline = [device newComputePipelineStateWithFunction:castI64I64Fn error:&error];
    if (!castI64I64Pipeline || error) return nil;
    id<MTLFunction> castF32I64Fn = [library newFunctionWithName:@"affon_cast_f32_i64"];
    if (!castF32I64Fn) return nil;
    id<MTLComputePipelineState> castF32I64Pipeline = [device newComputePipelineStateWithFunction:castF32I64Fn error:&error];
    if (!castF32I64Pipeline || error) return nil;
    id<MTLFunction> castI64F32Fn = [library newFunctionWithName:@"affon_cast_i64_f32"];
    if (!castI64F32Fn) return nil;
    id<MTLComputePipelineState> castI64F32Pipeline = [device newComputePipelineStateWithFunction:castI64F32Fn error:&error];
    if (!castI64F32Pipeline || error) return nil;
    id<MTLFunction> validateCastIndexF32I64Fn = [library newFunctionWithName:@"affon_validate_cast_index_f32_i64"];
    if (!validateCastIndexF32I64Fn) return nil;
    id<MTLComputePipelineState> validateCastIndexF32I64Pipeline = [device newComputePipelineStateWithFunction:validateCastIndexF32I64Fn error:&error];
    if (!validateCastIndexF32I64Pipeline || error) return nil;
    id<MTLFunction> gatherFn = [library newFunctionWithName:@"affon_gather_f32"];
    if (!gatherFn) return nil;
    id<MTLComputePipelineState> gatherPipeline = [device newComputePipelineStateWithFunction:gatherFn error:&error];
    if (!gatherPipeline || error) return nil;
    id<MTLFunction> gatherI64Fn = [library newFunctionWithName:@"affon_gather_i64_f32"];
    if (!gatherI64Fn) return nil;
    id<MTLComputePipelineState> gatherI64Pipeline = [device newComputePipelineStateWithFunction:gatherI64Fn error:&error];
    if (!gatherI64Pipeline || error) return nil;
    id<MTLFunction> gatherI64I64Fn = [library newFunctionWithName:@"affon_gather_i64_i64"];
    if (!gatherI64I64Fn) return nil;
    id<MTLComputePipelineState> gatherI64I64Pipeline = [device newComputePipelineStateWithFunction:gatherI64I64Fn error:&error];
    if (!gatherI64I64Pipeline || error) return nil;
    id<MTLFunction> indexSelectAxis0Fn = [library newFunctionWithName:@"affon_index_select_axis0_f32"];
    if (!indexSelectAxis0Fn) return nil;
    id<MTLComputePipelineState> indexSelectAxis0Pipeline = [device newComputePipelineStateWithFunction:indexSelectAxis0Fn error:&error];
    if (!indexSelectAxis0Pipeline || error) return nil;
    id<MTLFunction> indexSelectAxis0I64Fn = [library newFunctionWithName:@"affon_index_select_axis0_i64_f32"];
    if (!indexSelectAxis0I64Fn) return nil;
    id<MTLComputePipelineState> indexSelectAxis0I64Pipeline = [device newComputePipelineStateWithFunction:indexSelectAxis0I64Fn error:&error];
    if (!indexSelectAxis0I64Pipeline || error) return nil;
    id<MTLFunction> indexSelectAxis0I64I64Fn = [library newFunctionWithName:@"affon_index_select_axis0_i64_i64"];
    if (!indexSelectAxis0I64I64Fn) return nil;
    id<MTLComputePipelineState> indexSelectAxis0I64I64Pipeline = [device newComputePipelineStateWithFunction:indexSelectAxis0I64I64Fn error:&error];
    if (!indexSelectAxis0I64I64Pipeline || error) return nil;
    id<MTLFunction> indexSelectI64Fn = [library newFunctionWithName:@"affon_index_select_i64_f32"];
    if (!indexSelectI64Fn) return nil;
    id<MTLComputePipelineState> indexSelectI64Pipeline = [device newComputePipelineStateWithFunction:indexSelectI64Fn error:&error];
    if (!indexSelectI64Pipeline || error) return nil;
    id<MTLFunction> indexSelectI64I64Fn = [library newFunctionWithName:@"affon_index_select_i64_i64"];
    if (!indexSelectI64I64Fn) return nil;
    id<MTLComputePipelineState> indexSelectI64I64Pipeline = [device newComputePipelineStateWithFunction:indexSelectI64I64Fn error:&error];
    if (!indexSelectI64I64Pipeline || error) return nil;
    id<MTLFunction> indexSelectAxis0ScatterAddFn = [library newFunctionWithName:@"affon_index_select_axis0_scatter_add_f32"];
    if (!indexSelectAxis0ScatterAddFn) return nil;
    id<MTLComputePipelineState> indexSelectAxis0ScatterAddPipeline = [device newComputePipelineStateWithFunction:indexSelectAxis0ScatterAddFn error:&error];
    if (!indexSelectAxis0ScatterAddPipeline || error) return nil;
    id<MTLFunction> topkFn = [library newFunctionWithName:@"affon_topk_f32"];
    if (!topkFn) return nil;
    id<MTLComputePipelineState> topkPipeline = [device newComputePipelineStateWithFunction:topkFn error:&error];
    if (!topkPipeline || error) return nil;
    id<MTLFunction> topkI64Fn = [library newFunctionWithName:@"affon_topk_i64_f32"];
    if (!topkI64Fn) return nil;
    id<MTLComputePipelineState> topkI64Pipeline = [device newComputePipelineStateWithFunction:topkI64Fn error:&error];
    if (!topkI64Pipeline || error) return nil;
    id<MTLFunction> topkDirI64Fn = [library newFunctionWithName:@"affon_topk_dir_i64_f32"];
    if (!topkDirI64Fn) return nil;
    id<MTLComputePipelineState> topkDirI64Pipeline = [device newComputePipelineStateWithFunction:topkDirI64Fn error:&error];
    if (!topkDirI64Pipeline || error) return nil;
    id<MTLFunction> topkNdI64Fn = [library newFunctionWithName:@"affon_topk_nd_i64_f32"];
    if (!topkNdI64Fn) return nil;
    id<MTLComputePipelineState> topkNdI64Pipeline = [device newComputePipelineStateWithFunction:topkNdI64Fn error:&error];
    if (!topkNdI64Pipeline || error) return nil;
    id<MTLFunction> topkDirI64I64Fn = [library newFunctionWithName:@"affon_topk_dir_i64_i64"];
    if (!topkDirI64I64Fn) return nil;
    id<MTLComputePipelineState> topkDirI64I64Pipeline = [device newComputePipelineStateWithFunction:topkDirI64I64Fn error:&error];
    if (!topkDirI64I64Pipeline || error) return nil;
    id<MTLFunction> topkNdI64I64Fn = [library newFunctionWithName:@"affon_topk_nd_i64_i64"];
    if (!topkNdI64I64Fn) return nil;
    id<MTLComputePipelineState> topkNdI64I64Pipeline = [device newComputePipelineStateWithFunction:topkNdI64I64Fn error:&error];
    if (!topkNdI64I64Pipeline || error) return nil;
    id<MTLFunction> oneHotI64Fn = [library newFunctionWithName:@"affon_one_hot_i64_f32"];
    if (!oneHotI64Fn) return nil;
    id<MTLComputePipelineState> oneHotI64Pipeline = [device newComputePipelineStateWithFunction:oneHotI64Fn error:&error];
    if (!oneHotI64Pipeline || error) return nil;
    id<MTLFunction> reduceSumParallelFn = [library newFunctionWithName:@"affon_reduce_sum_parallel_f32"];
    if (!reduceSumParallelFn) return nil;
    id<MTLComputePipelineState> reduceSumParallelPipeline = [device newComputePipelineStateWithFunction:reduceSumParallelFn error:&error];
    if (!reduceSumParallelPipeline || error) return nil;
    id<MTLFunction> reduceMeanParallelFn = [library newFunctionWithName:@"affon_reduce_mean_parallel_f32"];
    if (!reduceMeanParallelFn) return nil;
    id<MTLComputePipelineState> reduceMeanParallelPipeline = [device newComputePipelineStateWithFunction:reduceMeanParallelFn error:&error];
    if (!reduceMeanParallelPipeline || error) return nil;
    id<MTLFunction> reduceAxisSumParallelFn = [library newFunctionWithName:@"affon_reduce_axis_sum_parallel_f32"];
    if (!reduceAxisSumParallelFn) return nil;
    id<MTLComputePipelineState> reduceAxisSumParallelPipeline = [device newComputePipelineStateWithFunction:reduceAxisSumParallelFn error:&error];
    if (!reduceAxisSumParallelPipeline || error) return nil;
    id<MTLFunction> reduceAxisMeanParallelFn = [library newFunctionWithName:@"affon_reduce_axis_mean_parallel_f32"];
    if (!reduceAxisMeanParallelFn) return nil;
    id<MTLComputePipelineState> reduceAxisMeanParallelPipeline = [device newComputePipelineStateWithFunction:reduceAxisMeanParallelFn error:&error];
    if (!reduceAxisMeanParallelPipeline || error) return nil;

    ctx = [AffonMetalContext new];
    ctx.device = device;
    ctx.queue = [device newCommandQueue];
    ctx.library = library;
    ctx.addF32Pipeline = addPipeline;
    ctx.subF32Pipeline = subPipeline;
    ctx.mulF32Pipeline = mulPipeline;
    ctx.divF32Pipeline = divPipeline;
    ctx.eqF32Pipeline = eqPipeline;
    ctx.ltF32Pipeline = ltPipeline;
    ctx.gtF32Pipeline = gtPipeline;
    ctx.addI64Pipeline = addI64Pipeline;
    ctx.subI64Pipeline = subI64Pipeline;
    ctx.mulI64Pipeline = mulI64Pipeline;
    ctx.divI64Pipeline = divI64Pipeline;
    ctx.eqI64Pipeline = eqI64Pipeline;
    ctx.ltI64Pipeline = ltI64Pipeline;
    ctx.gtI64Pipeline = gtI64Pipeline;
    ctx.muladdInplaceF32Pipeline = muladdInplacePipeline;
    ctx.axpyInplaceF32Pipeline = axpyInplacePipeline;
    ctx.subInplaceF32Pipeline = subInplacePipeline;
    ctx.subInplaceI64Pipeline = subInplaceI64Pipeline;
    ctx.scaleInplaceF32Pipeline = scaleInplacePipeline;
    ctx.fillF32Pipeline = fillPipeline;
    ctx.dotF32Pipeline = dotPipeline;
    ctx.dotI64Pipeline = dotI64Pipeline;
    ctx.adamStepF32Pipeline = adamStepPipeline;
    ctx.contiguousF32Pipeline = contiguousPipeline;
    ctx.contiguousI64Pipeline = contiguousI64Pipeline;
    ctx.contiguousSignedF32Pipeline = contiguousSignedPipeline;
    ctx.contiguousSignedI64Pipeline = contiguousSignedI64Pipeline;
    ctx.addBroadcastF32Pipeline = addBroadcastPipeline;
    ctx.subBroadcastF32Pipeline = subBroadcastPipeline;
    ctx.mulBroadcastF32Pipeline = mulBroadcastPipeline;
    ctx.divBroadcastF32Pipeline = divBroadcastPipeline;
    ctx.eqBroadcastF32Pipeline = eqBroadcastPipeline;
    ctx.ltBroadcastF32Pipeline = ltBroadcastPipeline;
    ctx.gtBroadcastF32Pipeline = gtBroadcastPipeline;
    ctx.expF32Pipeline = expPipeline;
    ctx.logF32Pipeline = logPipeline;
    ctx.sqrtF32Pipeline = sqrtPipeline;
    ctx.absF32Pipeline = absPipeline;
    ctx.absI64Pipeline = absI64Pipeline;
    ctx.signF32Pipeline = signPipeline;
    ctx.signI64Pipeline = signI64Pipeline;
    ctx.geluF32Pipeline = geluPipeline;
    ctx.geluGradF32Pipeline = geluGradPipeline;
    ctx.negF32Pipeline = negPipeline;
    ctx.negI64Pipeline = negI64Pipeline;
    ctx.reluF32Pipeline = reluPipeline;
    ctx.reluI64Pipeline = reluI64Pipeline;
    ctx.sigmoidF32Pipeline = sigmoidPipeline;
    ctx.siluF32Pipeline = siluPipeline;
    ctx.tanhF32Pipeline = tanhPipeline;
    ctx.clampF32Pipeline = clampPipeline;
    ctx.clampI64Pipeline = clampI64Pipeline;
    ctx.clampGradF32Pipeline = clampGradPipeline;
    ctx.softmaxF32Pipeline = softmaxPipeline;
    ctx.softmaxNdF32Pipeline = softmaxNdPipeline;
    ctx.logSoftmaxNllNdF32Pipeline = logSoftmaxNllNdPipeline;
    ctx.crossEntropyIndexedF32Pipeline = crossEntropyIndexedPipeline;
    ctx.crossEntropyIndexedTransposedF32Pipeline = crossEntropyIndexedTransposedPipeline;
    ctx.crossEntropyIndexedBackwardF32Pipeline = crossEntropyIndexedBackwardPipeline;
    ctx.layerNormNdF32Pipeline = layerNormNdPipeline;
    ctx.rmsNormNdF32Pipeline = rmsNormNdPipeline;
    ctx.addLayerNormNdF32Pipeline = addLayerNormNdPipeline;
    ctx.whereF32Pipeline = wherePipeline;
    ctx.whereI64F32Pipeline = whereI64Pipeline;
    ctx.whereF32I64Pipeline = whereF32I64Pipeline;
    ctx.whereI64I64Pipeline = whereI64I64Pipeline;
    ctx.maskedFillI64F32Pipeline = maskedFillPipeline;
    ctx.maskedFillI64I64Pipeline = maskedFillI64I64Pipeline;
    ctx.whereBroadcastF32Pipeline = whereBroadcastPipeline;
    ctx.whereBroadcastI64F32Pipeline = whereBroadcastI64F32Pipeline;
    ctx.maskedFillBroadcastI64F32Pipeline = maskedFillBroadcastF32Pipeline;
    ctx.maskedFillBroadcastI64I64Pipeline = maskedFillBroadcastI64Pipeline;
    ctx.scatterAddF32Pipeline = scatterAddPipeline;
    ctx.scatterAddI64F32Pipeline = scatterAddI64Pipeline;
    ctx.reduceSumF32Pipeline = reduceSumPipeline;
    ctx.reduceSumI64Pipeline = reduceSumI64Pipeline;
    ctx.reduceMeanF32Pipeline = reduceMeanPipeline;
    ctx.reduceMinF32Pipeline = reduceMinPipeline;
    ctx.reduceMinI64Pipeline = reduceMinI64Pipeline;
    ctx.reduceMaxF32Pipeline = reduceMaxPipeline;
    ctx.reduceMaxI64Pipeline = reduceMaxI64Pipeline;
    ctx.reduceArgminF32Pipeline = reduceArgminPipeline;
    ctx.reduceArgmaxF32Pipeline = reduceArgmaxPipeline;
    ctx.reduceArgminI64F32Pipeline = reduceArgminI64Pipeline;
    ctx.reduceArgmaxI64F32Pipeline = reduceArgmaxI64Pipeline;
    ctx.reduceArgminI64I64Pipeline = reduceArgminI64I64Pipeline;
    ctx.reduceArgmaxI64I64Pipeline = reduceArgmaxI64I64Pipeline;
    ctx.reduceVarianceF32Pipeline = reduceVariancePipeline;
    ctx.reduceStdF32Pipeline = reduceStdPipeline;
    ctx.reduceAxisSumF32Pipeline = reduceAxisSumPipeline;
    ctx.reduceAxisSumI64Pipeline = reduceAxisSumI64Pipeline;
    ctx.reduceAxisMeanF32Pipeline = reduceAxisMeanPipeline;
    ctx.reduceAxisMinF32Pipeline = reduceAxisMinPipeline;
    ctx.reduceAxisMinI64Pipeline = reduceAxisMinI64Pipeline;
    ctx.reduceAxisMaxF32Pipeline = reduceAxisMaxPipeline;
    ctx.reduceAxisMaxI64Pipeline = reduceAxisMaxI64Pipeline;
    ctx.reduceAxisVarianceF32Pipeline = reduceAxisVariancePipeline;
    ctx.reduceAxisStdF32Pipeline = reduceAxisStdPipeline;
    ctx.reduceAxisArgminF32Pipeline = reduceAxisArgminPipeline;
    ctx.reduceAxisArgmaxF32Pipeline = reduceAxisArgmaxPipeline;
    ctx.reduceAxisArgminI64F32Pipeline = reduceAxisArgminI64Pipeline;
    ctx.reduceAxisArgmaxI64F32Pipeline = reduceAxisArgmaxI64Pipeline;
    ctx.reduceAxisArgminI64I64Pipeline = reduceAxisArgminI64I64Pipeline;
    ctx.reduceAxisArgmaxI64I64Pipeline = reduceAxisArgmaxI64I64Pipeline;
    ctx.reduceAxisNdF32Pipeline = reduceAxisNdPipeline;
    ctx.reduceAxisNdI64Pipeline = reduceAxisNdI64Pipeline;
    ctx.reduceAxisNdArgI64F32Pipeline = reduceAxisNdArgPipeline;
    ctx.reduceAxisNdArgI64I64Pipeline = reduceAxisNdArgI64I64Pipeline;
    ctx.matmulF32Pipeline = matmulPipeline;
    ctx.matmulI64Pipeline = matmulI64Pipeline;
    ctx.matmulStridedF32Pipeline = matmulStridedPipeline;
    ctx.matmulStridedI64Pipeline = matmulStridedI64Pipeline;
    ctx.matmulAddF32Pipeline = matmulAddPipeline;
    ctx.matmulAddI64Pipeline = matmulAddI64Pipeline;
    ctx.matmulAddGeluF32Pipeline = matmulAddGeluPipeline;
    ctx.attentionScoresF32Pipeline = attentionScoresPipeline;
    ctx.embeddingI64F32Pipeline = embeddingI64F32Pipeline;
    ctx.embeddingI64I64Pipeline = embeddingI64I64Pipeline;
    ctx.unaryChainF32Pipeline = unaryChainPipeline;
    ctx.unaryChainI64Pipeline = unaryChainI64Pipeline;
    ctx.binaryThenUnaryChainF32Pipeline = binaryThenUnaryChainPipeline;
    ctx.binaryThenUnaryChainI64Pipeline = binaryThenUnaryChainI64Pipeline;
    ctx.castF32F32Pipeline = castF32F32Pipeline;
    ctx.castI64I64Pipeline = castI64I64Pipeline;
    ctx.castF32I64Pipeline = castF32I64Pipeline;
    ctx.castI64F32Pipeline = castI64F32Pipeline;
    ctx.validateCastIndexF32I64Pipeline = validateCastIndexF32I64Pipeline;
    ctx.gatherF32Pipeline = gatherPipeline;
    ctx.gatherI64F32Pipeline = gatherI64Pipeline;
    ctx.gatherI64I64Pipeline = gatherI64I64Pipeline;
    ctx.indexSelectAxis0F32Pipeline = indexSelectAxis0Pipeline;
    ctx.indexSelectAxis0I64F32Pipeline = indexSelectAxis0I64Pipeline;
    ctx.indexSelectAxis0I64I64Pipeline = indexSelectAxis0I64I64Pipeline;
    ctx.indexSelectI64F32Pipeline = indexSelectI64Pipeline;
    ctx.indexSelectI64I64Pipeline = indexSelectI64I64Pipeline;
    ctx.indexSelectAxis0ScatterAddF32Pipeline = indexSelectAxis0ScatterAddPipeline;
    ctx.topkF32Pipeline = topkPipeline;
    ctx.topkI64F32Pipeline = topkI64Pipeline;
    ctx.topkDirI64F32Pipeline = topkDirI64Pipeline;
    ctx.topkNdI64F32Pipeline = topkNdI64Pipeline;
    ctx.topkDirI64I64Pipeline = topkDirI64I64Pipeline;
    ctx.topkNdI64I64Pipeline = topkNdI64I64Pipeline;
    ctx.oneHotI64F32Pipeline = oneHotI64Pipeline;
    ctx.reduceSumParallelF32Pipeline = reduceSumParallelPipeline;
    ctx.reduceMeanParallelF32Pipeline = reduceMeanParallelPipeline;
    ctx.reduceAxisSumParallelF32Pipeline = reduceAxisSumParallelPipeline;
    ctx.reduceAxisMeanParallelF32Pipeline = reduceAxisMeanParallelPipeline;
    return ctx;
}

int affon_metal_matmul_strided_f32(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    size_t a_offset_bytes,
    size_t b_offset_bytes,
    size_t out_offset_bytes,
    ptrdiff_t a_row_stride,
    ptrdiff_t a_col_stride,
    ptrdiff_t b_row_stride,
    ptrdiff_t b_col_stride,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.matmulStridedF32Pipeline || !a_handle || !b_handle || !out_handle) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *b = (__bridge AffonMetalBuffer *)b_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    int32_t aRow32 = (int32_t)a_row_stride;
    int32_t aCol32 = (int32_t)a_col_stride;
    int32_t bRow32 = (int32_t)b_row_stride;
    int32_t bCol32 = (int32_t)b_col_stride;
    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.matmulStridedF32Pipeline];
    [encoder setBuffer:a.buffer offset:a_offset_bytes atIndex:0];
    [encoder setBuffer:b.buffer offset:b_offset_bytes atIndex:1];
    [encoder setBuffer:out.buffer offset:out_offset_bytes atIndex:2];
    [encoder setBytes:&aRow32 length:sizeof(aRow32) atIndex:3];
    [encoder setBytes:&aCol32 length:sizeof(aCol32) atIndex:4];
    [encoder setBytes:&bRow32 length:sizeof(bRow32) atIndex:5];
    [encoder setBytes:&bCol32 length:sizeof(bCol32) atIndex:6];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:7];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:8];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:9];

    NSUInteger w = ctx.matmulStridedF32Pipeline.threadExecutionWidth;
    if (w == 0) w = 8;
    NSUInteger h = ctx.matmulStridedF32Pipeline.maxTotalThreadsPerThreadgroup / w;
    if (h == 0) h = 1;
    if (h > 8) h = 8;

    MTLSize gridSize = MTLSizeMake(n, m, 1);
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_matmul_strided_many_f32(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    const size_t *a_offsets_bytes,
    const size_t *b_offsets_bytes,
    const size_t *out_offsets_bytes,
    size_t batch_count,
    ptrdiff_t a_row_stride,
    ptrdiff_t a_col_stride,
    ptrdiff_t b_row_stride,
    ptrdiff_t b_col_stride,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.matmulStridedF32Pipeline || !a_handle || !b_handle || !out_handle) return -1;
    if ((batch_count > 0) && (!a_offsets_bytes || !b_offsets_bytes || !out_offsets_bytes)) return -1;
    if (batch_count == 0) return 0;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *b = (__bridge AffonMetalBuffer *)b_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    int32_t aRow32 = (int32_t)a_row_stride;
    int32_t aCol32 = (int32_t)a_col_stride;
    int32_t bRow32 = (int32_t)b_row_stride;
    int32_t bCol32 = (int32_t)b_col_stride;
    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.matmulStridedF32Pipeline];
    [encoder setBytes:&aRow32 length:sizeof(aRow32) atIndex:3];
    [encoder setBytes:&aCol32 length:sizeof(aCol32) atIndex:4];
    [encoder setBytes:&bRow32 length:sizeof(bRow32) atIndex:5];
    [encoder setBytes:&bCol32 length:sizeof(bCol32) atIndex:6];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:7];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:8];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:9];

    NSUInteger w = ctx.matmulStridedF32Pipeline.threadExecutionWidth;
    if (w == 0) w = 8;
    NSUInteger h = ctx.matmulStridedF32Pipeline.maxTotalThreadsPerThreadgroup / w;
    if (h == 0) h = 1;
    if (h > 8) h = 8;

    MTLSize gridSize = MTLSizeMake(n, m, 1);
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    for (size_t i = 0; i < batch_count; i++) {
        [encoder setBuffer:a.buffer offset:a_offsets_bytes[i] atIndex:0];
        [encoder setBuffer:b.buffer offset:b_offsets_bytes[i] atIndex:1];
        [encoder setBuffer:out.buffer offset:out_offsets_bytes[i] atIndex:2];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_matmul_offset_many_f32(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    const size_t *a_offsets_bytes,
    const size_t *b_offsets_bytes,
    const size_t *out_offsets_bytes,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.matmulF32Pipeline || !a_handle || !b_handle || !out_handle) return -1;
    if ((batch_count > 0) && (!a_offsets_bytes || !b_offsets_bytes || !out_offsets_bytes)) return -1;
    if (batch_count == 0) return 0;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *b = (__bridge AffonMetalBuffer *)b_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.matmulF32Pipeline];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:3];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:4];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:5];

    NSUInteger w = ctx.matmulF32Pipeline.threadExecutionWidth;
    if (w == 0) w = 8;
    NSUInteger h = ctx.matmulF32Pipeline.maxTotalThreadsPerThreadgroup / w;
    if (h == 0) h = 1;
    if (h > 8) h = 8;

    MTLSize gridSize = MTLSizeMake(n, m, 1);
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    for (size_t i = 0; i < batch_count; i++) {
        [encoder setBuffer:a.buffer offset:a_offsets_bytes[i] atIndex:0];
        [encoder setBuffer:b.buffer offset:b_offsets_bytes[i] atIndex:1];
        [encoder setBuffer:out.buffer offset:out_offsets_bytes[i] atIndex:2];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_matmul_strided_i64(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    size_t a_offset_bytes,
    size_t b_offset_bytes,
    size_t out_offset_bytes,
    ptrdiff_t a_row_stride,
    ptrdiff_t a_col_stride,
    ptrdiff_t b_row_stride,
    ptrdiff_t b_col_stride,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.matmulStridedI64Pipeline || !a_handle || !b_handle || !out_handle) return -1;

    id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
    id<MTLBuffer> b = affon_metal_borrow_buffer(b_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!a || !b || !out) return -1;

    int32_t aRow32 = (int32_t)a_row_stride;
    int32_t aCol32 = (int32_t)a_col_stride;
    int32_t bRow32 = (int32_t)b_row_stride;
    int32_t bCol32 = (int32_t)b_col_stride;
    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.matmulStridedI64Pipeline];
    [encoder setBuffer:a offset:a_offset_bytes atIndex:0];
    [encoder setBuffer:b offset:b_offset_bytes atIndex:1];
    [encoder setBuffer:out offset:out_offset_bytes atIndex:2];
    [encoder setBytes:&aRow32 length:sizeof(aRow32) atIndex:3];
    [encoder setBytes:&aCol32 length:sizeof(aCol32) atIndex:4];
    [encoder setBytes:&bRow32 length:sizeof(bRow32) atIndex:5];
    [encoder setBytes:&bCol32 length:sizeof(bCol32) atIndex:6];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:7];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:8];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:9];

    NSUInteger w = ctx.matmulStridedI64Pipeline.threadExecutionWidth;
    if (w == 0) w = 8;
    NSUInteger h = ctx.matmulStridedI64Pipeline.maxTotalThreadsPerThreadgroup / w;
    if (h == 0) h = 1;
    if (h > 8) h = 8;

    MTLSize gridSize = MTLSizeMake(n, m, 1);
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

int affon_metal_matmul_strided_many_i64(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    const size_t *a_offsets_bytes,
    const size_t *b_offsets_bytes,
    const size_t *out_offsets_bytes,
    size_t batch_count,
    ptrdiff_t a_row_stride,
    ptrdiff_t a_col_stride,
    ptrdiff_t b_row_stride,
    ptrdiff_t b_col_stride,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.matmulStridedI64Pipeline || !a_handle || !b_handle || !out_handle) return -1;
    if ((batch_count > 0) && (!a_offsets_bytes || !b_offsets_bytes || !out_offsets_bytes)) return -1;
    if (batch_count == 0) return 0;

    id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
    id<MTLBuffer> b = affon_metal_borrow_buffer(b_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!a || !b || !out) return -1;

    int32_t aRow32 = (int32_t)a_row_stride;
    int32_t aCol32 = (int32_t)a_col_stride;
    int32_t bRow32 = (int32_t)b_row_stride;
    int32_t bCol32 = (int32_t)b_col_stride;
    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.matmulStridedI64Pipeline];
    [encoder setBytes:&aRow32 length:sizeof(aRow32) atIndex:3];
    [encoder setBytes:&aCol32 length:sizeof(aCol32) atIndex:4];
    [encoder setBytes:&bRow32 length:sizeof(bRow32) atIndex:5];
    [encoder setBytes:&bCol32 length:sizeof(bCol32) atIndex:6];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:7];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:8];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:9];

    NSUInteger w = ctx.matmulStridedI64Pipeline.threadExecutionWidth;
    if (w == 0) w = 8;
    NSUInteger h = ctx.matmulStridedI64Pipeline.maxTotalThreadsPerThreadgroup / w;
    if (h == 0) h = 1;
    if (h > 8) h = 8;

    MTLSize gridSize = MTLSizeMake(n, m, 1);
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    for (size_t i = 0; i < batch_count; i++) {
        [encoder setBuffer:a offset:a_offsets_bytes[i] atIndex:0];
        [encoder setBuffer:b offset:b_offsets_bytes[i] atIndex:1];
        [encoder setBuffer:out offset:out_offsets_bytes[i] atIndex:2];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

static int affon_run_binary_f32_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *a_handle,
    void *b_handle,
    void *out_handle,
    size_t len
) {
    @autoreleasepool {
        AffonMetalContext *ctx = affon_get_context();
        if (!ctx || !pipeline || !a_handle || !b_handle || !out_handle) return -1;

        id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
        id<MTLBuffer> b = affon_metal_borrow_buffer(b_handle);
        id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
        if (!a || !b || !out) return -1;

        id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
        if (!commandBuffer) return -2;
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) return -3;

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a offset:0 atIndex:0];
        [encoder setBuffer:b offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];

        MTLSize gridSize = MTLSizeMake(len, 1, 1);
        NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadWidth > len && len > 0) threadWidth = len;
        if (threadWidth == 0) threadWidth = 1;
        MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.error) return -4;
        return 0;
    }
}

static int affon_run_fill_f32_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *out_handle,
    size_t len,
    float value
) {
    @autoreleasepool {
        AffonMetalContext *ctx = affon_get_context();
        if (!ctx || !pipeline || !out_handle) return -1;

        id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
        if (!out) return -1;

        id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
        if (!commandBuffer) return -1;
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) return -1;

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:out offset:0 atIndex:0];
        [encoder setBytes:&value length:sizeof(float) atIndex:1];

        NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadWidth == 0) threadWidth = 1;
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, 1, 1);
        MTLSize threadsPerGrid = MTLSizeMake(len, 1, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        return commandBuffer.error ? -1 : 0;
    }
}

static int affon_run_adam_step_f32_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *param_handle,
    void *m_handle,
    void *v_handle,
    void *grad_handle,
    float beta1,
    float beta2,
    float bias_correction1,
    float bias_correction2,
    float eps,
    float lr,
    float weight_decay,
    size_t len
) {
    @autoreleasepool {
        AffonMetalContext *ctx = affon_get_context();
        if (!ctx || !pipeline || !param_handle || !m_handle || !v_handle || !grad_handle) return -1;

        id<MTLBuffer> param = affon_metal_borrow_buffer(param_handle);
        id<MTLBuffer> m = affon_metal_borrow_buffer(m_handle);
        id<MTLBuffer> v = affon_metal_borrow_buffer(v_handle);
        id<MTLBuffer> grad = affon_metal_borrow_buffer(grad_handle);
        if (!param || !m || !v || !grad) return -1;
        uint32_t len32 = (uint32_t)len;

        id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
        if (!commandBuffer) return -2;
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) return -3;

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:param offset:0 atIndex:0];
        [encoder setBuffer:m offset:0 atIndex:1];
        [encoder setBuffer:v offset:0 atIndex:2];
        [encoder setBuffer:grad offset:0 atIndex:3];
        [encoder setBytes:&beta1 length:sizeof(beta1) atIndex:4];
        [encoder setBytes:&beta2 length:sizeof(beta2) atIndex:5];
        [encoder setBytes:&bias_correction1 length:sizeof(bias_correction1) atIndex:6];
        [encoder setBytes:&bias_correction2 length:sizeof(bias_correction2) atIndex:7];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:8];
        [encoder setBytes:&lr length:sizeof(lr) atIndex:9];
        [encoder setBytes:&weight_decay length:sizeof(weight_decay) atIndex:10];
        [encoder setBytes:&len32 length:sizeof(len32) atIndex:11];

        MTLSize gridSize = MTLSizeMake(len, 1, 1);
        NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadWidth > len && len > 0) threadWidth = len;
        if (threadWidth == 0) threadWidth = 1;
        MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

static int affon_run_contiguous_f32_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *input_handle,
    void *out_handle,
    size_t ndim,
    const uint32_t *shape,
    const uint32_t *strides,
    size_t offset,
    size_t len
) {
    @autoreleasepool {
        AffonMetalContext *ctx = affon_get_context();
        if (!ctx || !pipeline || !input_handle || !out_handle || !shape || !strides) return -1;
        if (ndim == 0 || ndim > affon_max_broadcast_dims) return -1;

        id<MTLBuffer> input = affon_metal_borrow_buffer(input_handle);
        id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
        if (!input || !out) return -1;
        uint32_t ndim32 = (uint32_t)ndim;
        uint32_t offset32 = (uint32_t)offset;
        uint32_t len32 = (uint32_t)len;

        id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
        if (!commandBuffer) return -2;
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) return -3;

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:2];
        [encoder setBytes:strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:4];
        [encoder setBytes:&offset32 length:sizeof(offset32) atIndex:5];
        [encoder setBytes:&len32 length:sizeof(len32) atIndex:6];

        MTLSize gridSize = MTLSizeMake(len, 1, 1);
        NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadWidth > len && len > 0) threadWidth = len;
        if (threadWidth == 0) threadWidth = 1;
        MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

static int affon_run_contiguous_signed_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *input_handle,
    void *out_handle,
    size_t ndim,
    const uint32_t *shape,
    const int32_t *strides,
    size_t offset,
    size_t len
) {
    @autoreleasepool {
        AffonMetalContext *ctx = affon_get_context();
        if (!ctx || !pipeline || !input_handle || !out_handle || !shape || !strides) return -1;
        if (ndim == 0 || ndim > affon_max_broadcast_dims) return -1;

        id<MTLBuffer> input = affon_metal_borrow_buffer(input_handle);
        id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
        if (!input || !out) return -1;
        uint32_t ndim32 = (uint32_t)ndim;
        int32_t offset32 = (int32_t)offset;
        uint32_t len32 = (uint32_t)len;

        id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
        if (!commandBuffer) return -2;
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) return -3;

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:2];
        [encoder setBytes:strides length:affon_max_broadcast_dims * sizeof(int32_t) atIndex:3];
        [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:4];
        [encoder setBytes:&offset32 length:sizeof(offset32) atIndex:5];
        [encoder setBytes:&len32 length:sizeof(len32) atIndex:6];

        MTLSize gridSize = MTLSizeMake(len, 1, 1);
        NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadWidth > len && len > 0) threadWidth = len;
        if (threadWidth == 0) threadWidth = 1;
        MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

static int affon_run_binary_broadcast_f32_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *a_handle,
    void *b_handle,
    void *out_handle,
    size_t ndim,
    const uint32_t *shape,
    const uint32_t *a_strides,
    const uint32_t *b_strides,
    size_t a_offset,
    size_t b_offset,
    size_t len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !a_handle || !b_handle || !out_handle || !shape || !a_strides || !b_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims) return -1;

    id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
    id<MTLBuffer> b = affon_metal_borrow_buffer(b_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!a || !b || !out) return -1;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t a_offset32 = (uint32_t)a_offset;
    uint32_t b_offset32 = (uint32_t)b_offset;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a offset:0 atIndex:0];
    [encoder setBuffer:b offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2];
    [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:a_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:b_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:6];
    [encoder setBytes:&a_offset32 length:sizeof(a_offset32) atIndex:7];
    [encoder setBytes:&b_offset32 length:sizeof(b_offset32) atIndex:8];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:9];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

static int affon_run_unary_f32_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *a_handle,
    void *out_handle,
    size_t len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !a_handle || !out_handle) return -1;

    id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!a || !out) return -1;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

static int affon_run_binary_inplace_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *target_handle,
    void *other_handle,
    size_t len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !target_handle || !other_handle) return -1;

    id<MTLBuffer> target = affon_metal_borrow_buffer(target_handle);
    id<MTLBuffer> other = affon_metal_borrow_buffer(other_handle);
    if (!target || !other) return -1;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:target offset:0 atIndex:0];
    [encoder setBuffer:other offset:0 atIndex:1];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

static int affon_run_binary_scalar_inplace_f32_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *target_handle,
    void *other_handle,
    float scale,
    size_t len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !target_handle || !other_handle) return -1;

    id<MTLBuffer> target = affon_metal_borrow_buffer(target_handle);
    id<MTLBuffer> other = affon_metal_borrow_buffer(other_handle);
    if (!target || !other) return -1;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:target offset:0 atIndex:0];
    [encoder setBuffer:other offset:0 atIndex:1];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:2];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

static int affon_run_unary_scalar_inplace_f32_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *target_handle,
    float scale,
    size_t len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !target_handle) return -1;

    id<MTLBuffer> target = affon_metal_borrow_buffer(target_handle);
    if (!target) return -1;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:target offset:0 atIndex:0];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:1];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

static int affon_run_where_broadcast_f32_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *cond_handle,
    void *a_handle,
    void *b_handle,
    void *out_handle,
    size_t ndim,
    const uint32_t *shape,
    const uint32_t *cond_strides,
    const uint32_t *a_strides,
    const uint32_t *b_strides,
    size_t cond_offset,
    size_t a_offset,
    size_t b_offset,
    size_t len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !cond_handle || !a_handle || !b_handle || !out_handle || !shape || !cond_strides || !a_strides || !b_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims) return -1;

    id<MTLBuffer> cond = affon_metal_borrow_buffer(cond_handle);
    id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
    id<MTLBuffer> b = affon_metal_borrow_buffer(b_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!cond || !a || !b || !out) return -1;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t cond_offset32 = (uint32_t)cond_offset;
    uint32_t a_offset32 = (uint32_t)a_offset;
    uint32_t b_offset32 = (uint32_t)b_offset;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:cond offset:0 atIndex:0];
    [encoder setBuffer:a offset:0 atIndex:1];
    [encoder setBuffer:b offset:0 atIndex:2];
    [encoder setBuffer:out offset:0 atIndex:3];
    [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:cond_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:5];
    [encoder setBytes:a_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:6];
    [encoder setBytes:b_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:8];
    [encoder setBytes:&cond_offset32 length:sizeof(cond_offset32) atIndex:9];
    [encoder setBytes:&a_offset32 length:sizeof(a_offset32) atIndex:10];
    [encoder setBytes:&b_offset32 length:sizeof(b_offset32) atIndex:11];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:12];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

static int affon_run_masked_fill_broadcast_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *input_handle,
    void *mask_handle,
    void *out_handle,
    size_t ndim,
    const uint32_t *shape,
    const uint32_t *input_strides,
    const uint32_t *mask_strides,
    size_t input_offset,
    size_t mask_offset,
    const void *fill_value,
    size_t fill_value_size,
    size_t len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !input_handle || !mask_handle || !out_handle || !shape || !input_strides || !mask_strides || !fill_value) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims) return -1;

    id<MTLBuffer> input = affon_metal_borrow_buffer(input_handle);
    id<MTLBuffer> mask = affon_metal_borrow_buffer(mask_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!input || !mask || !out) return -1;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t input_offset32 = (uint32_t)input_offset;
    uint32_t mask_offset32 = (uint32_t)mask_offset;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:mask offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2];
    [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:mask_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:6];
    [encoder setBytes:&input_offset32 length:sizeof(input_offset32) atIndex:7];
    [encoder setBytes:&mask_offset32 length:sizeof(mask_offset32) atIndex:8];
    [encoder setBytes:fill_value length:fill_value_size atIndex:9];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:10];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

bool affon_metal_is_available(void) {
    return affon_get_context() != nil;
}

const char *affon_metal_last_error(void) {
    return affon_metal_last_error_message;
}

void *affon_metal_buffer_create(size_t byte_len) {
    @autoreleasepool {
        AffonMetalContext *ctx = affon_get_context();
        if (!ctx) return NULL;
        id<MTLBuffer> buffer = [ctx.device newBufferWithLength:byte_len options:MTLResourceStorageModeShared];
        if (!buffer) {
            affon_set_last_error([NSString stringWithFormat:@"newBufferWithLength failed for %zu bytes", byte_len]);
            return NULL;
        }

        AffonMetalBuffer *wrapper = [AffonMetalBuffer new];
        wrapper.buffer = buffer;
        return affon_metal_retain_buffer_wrapper(wrapper);
    }
}

void affon_metal_buffer_destroy(void *handle) {
    if (!handle) return;
    @autoreleasepool {
        AffonMetalBuffer *wrapper = affon_metal_transfer_buffer_wrapper(handle);
        if (wrapper.buffer) {
            [wrapper.buffer setPurgeableState:MTLPurgeableStateEmpty];
        }
    }
}

int affon_metal_buffer_write(void *handle, const unsigned char *src, size_t byte_len) {
    if (!handle || !src) return -1;
    id<MTLBuffer> buffer = affon_metal_borrow_buffer(handle);
    if (!buffer || buffer.length < byte_len) return -2;
    memcpy(buffer.contents, src, byte_len);
    return 0;
}

int affon_metal_buffer_read(void *handle, unsigned char *dst, size_t byte_len) {
    if (!handle || !dst) return -1;
    id<MTLBuffer> buffer = affon_metal_borrow_buffer(handle);
    if (!buffer || buffer.length < byte_len) return -2;
    memcpy(dst, buffer.contents, byte_len);
    return 0;
}

const unsigned char *affon_metal_buffer_contents(void *handle) {
    id<MTLBuffer> buffer = affon_metal_borrow_buffer(handle);
    return buffer ? buffer.contents : NULL;
}

unsigned char *affon_metal_buffer_mutable_contents(void *handle) {
    id<MTLBuffer> buffer = affon_metal_borrow_buffer(handle);
    return buffer ? buffer.contents : NULL;
}

int affon_metal_buffer_copy(void *dst_handle, size_t dst_offset, void *src_handle, size_t src_offset, size_t byte_len) {
    @autoreleasepool {
        AffonMetalContext *ctx = affon_get_context();
        if (!ctx || !dst_handle || !src_handle) return -1;

        AffonMetalBuffer *dst = affon_metal_borrow_buffer_wrapper(dst_handle);
        AffonMetalBuffer *src = affon_metal_borrow_buffer_wrapper(src_handle);
        if (dst.buffer.length < dst_offset + byte_len) return -2;
        if (src.buffer.length < src_offset + byte_len) return -3;

        id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
        if (!commandBuffer) return -4;
        id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
        if (!encoder) return -5;

        [encoder copyFromBuffer:src.buffer sourceOffset:src_offset toBuffer:dst.buffer destinationOffset:dst_offset size:byte_len];
        [encoder endEncoding];
        return affon_metal_finish_command_buffer(commandBuffer, -6);
    }
}

int affon_metal_add_f32(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.addF32Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_sub_f32(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.subF32Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_mul_f32(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.mulF32Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_div_f32(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.divF32Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_eq_f32(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.eqF32Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_lt_f32(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.ltF32Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_gt_f32(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.gtF32Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_add_i64(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.addI64Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_sub_i64(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.subI64Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_mul_i64(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.mulI64Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_div_i64(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.divI64Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_eq_i64(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.eqI64Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_lt_i64(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.ltI64Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_gt_i64(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_f32_pipeline(ctx ? ctx.gtI64Pipeline : nil, a_handle, b_handle, out_handle, len);
}

int affon_metal_muladd_inplace_f32(void *target_handle, void *addend_handle, float scale, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_scalar_inplace_f32_pipeline(ctx ? ctx.muladdInplaceF32Pipeline : nil, target_handle, addend_handle, scale, len);
}

int affon_metal_axpy_inplace_f32(void *target_handle, void *addend_handle, float scale, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_scalar_inplace_f32_pipeline(ctx ? ctx.axpyInplaceF32Pipeline : nil, target_handle, addend_handle, scale, len);
}

int affon_metal_sub_inplace_f32(void *target_handle, void *delta_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_inplace_pipeline(ctx ? ctx.subInplaceF32Pipeline : nil, target_handle, delta_handle, len);
}

int affon_metal_sub_inplace_i64(void *target_handle, void *delta_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_inplace_pipeline(ctx ? ctx.subInplaceI64Pipeline : nil, target_handle, delta_handle, len);
}

int affon_metal_scale_inplace_f32(void *target_handle, float scale, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_scalar_inplace_f32_pipeline(ctx ? ctx.scaleInplaceF32Pipeline : nil, target_handle, scale, len);
}

int affon_metal_fill_f32(void *out_handle, size_t len, float value) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_fill_f32_pipeline(ctx ? ctx.fillF32Pipeline : nil, out_handle, len, value);
}

int affon_metal_adam_step_f32(void *param_handle, void *m_handle, void *v_handle, void *grad_handle, float beta1, float beta2, float bias_correction1, float bias_correction2, float eps, float lr, float weight_decay, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_adam_step_f32_pipeline(ctx ? ctx.adamStepF32Pipeline : nil, param_handle, m_handle, v_handle, grad_handle, beta1, beta2, bias_correction1, bias_correction2, eps, lr, weight_decay, len);
}

int affon_metal_adam_step_many_f32(
    void *const *param_handles,
    void *const *m_handles,
    void *const *v_handles,
    void *const *grad_handles,
    const size_t *lens,
    size_t count,
    float beta1,
    float beta2,
    float bias_correction1,
    float bias_correction2,
    float eps,
    float lr,
    float weight_decay
) {
    @autoreleasepool {
        AffonMetalContext *ctx = affon_get_context();
        id<MTLComputePipelineState> pipeline = ctx ? ctx.adamStepF32Pipeline : nil;
        if (!ctx || !pipeline || !param_handles || !m_handles || !v_handles || !grad_handles || !lens) return -1;
        if (count == 0) return 0;

        id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
        if (!commandBuffer) return -2;
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) return -3;

        [encoder setComputePipelineState:pipeline];
        for (size_t i = 0; i < count; ++i) {
            id<MTLBuffer> param = affon_metal_borrow_buffer(param_handles[i]);
            id<MTLBuffer> m = affon_metal_borrow_buffer(m_handles[i]);
            id<MTLBuffer> v = affon_metal_borrow_buffer(v_handles[i]);
            id<MTLBuffer> grad = affon_metal_borrow_buffer(grad_handles[i]);
            if (!param || !m || !v || !grad) {
                [encoder endEncoding];
                return -1;
            }
            uint32_t len32 = (uint32_t)lens[i];

            [encoder setBuffer:param offset:0 atIndex:0];
            [encoder setBuffer:m offset:0 atIndex:1];
            [encoder setBuffer:v offset:0 atIndex:2];
            [encoder setBuffer:grad offset:0 atIndex:3];
            [encoder setBytes:&beta1 length:sizeof(beta1) atIndex:4];
            [encoder setBytes:&beta2 length:sizeof(beta2) atIndex:5];
            [encoder setBytes:&bias_correction1 length:sizeof(bias_correction1) atIndex:6];
            [encoder setBytes:&bias_correction2 length:sizeof(bias_correction2) atIndex:7];
            [encoder setBytes:&eps length:sizeof(eps) atIndex:8];
            [encoder setBytes:&lr length:sizeof(lr) atIndex:9];
            [encoder setBytes:&weight_decay length:sizeof(weight_decay) atIndex:10];
            [encoder setBytes:&len32 length:sizeof(len32) atIndex:11];

            MTLSize gridSize = MTLSizeMake(lens[i], 1, 1);
            NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
            if (threadWidth > lens[i] && lens[i] > 0) threadWidth = lens[i];
            if (threadWidth == 0) threadWidth = 1;
            MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        }
        [encoder endEncoding];

        return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

int affon_metal_contiguous_f32(void *input_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *strides, size_t offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_contiguous_f32_pipeline(ctx ? ctx.contiguousF32Pipeline : nil, input_handle, out_handle, ndim, shape, strides, offset, len);
}

int affon_metal_contiguous_i64(void *input_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *strides, size_t offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_contiguous_f32_pipeline(ctx ? ctx.contiguousI64Pipeline : nil, input_handle, out_handle, ndim, shape, strides, offset, len);
}

int affon_metal_contiguous_signed_f32(void *input_handle, void *out_handle, size_t ndim, const uint32_t *shape, const int32_t *strides, size_t offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_contiguous_signed_pipeline(ctx ? ctx.contiguousSignedF32Pipeline : nil, input_handle, out_handle, ndim, shape, strides, offset, len);
}

int affon_metal_contiguous_signed_i64(void *input_handle, void *out_handle, size_t ndim, const uint32_t *shape, const int32_t *strides, size_t offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_contiguous_signed_pipeline(ctx ? ctx.contiguousSignedI64Pipeline : nil, input_handle, out_handle, ndim, shape, strides, offset, len);
}

int affon_metal_add_broadcast_f32(void *a_handle, void *b_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *a_strides, const uint32_t *b_strides, size_t a_offset, size_t b_offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_broadcast_f32_pipeline(ctx ? ctx.addBroadcastF32Pipeline : nil, a_handle, b_handle, out_handle, ndim, shape, a_strides, b_strides, a_offset, b_offset, len);
}

int affon_metal_sub_broadcast_f32(void *a_handle, void *b_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *a_strides, const uint32_t *b_strides, size_t a_offset, size_t b_offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_broadcast_f32_pipeline(ctx ? ctx.subBroadcastF32Pipeline : nil, a_handle, b_handle, out_handle, ndim, shape, a_strides, b_strides, a_offset, b_offset, len);
}

int affon_metal_mul_broadcast_f32(void *a_handle, void *b_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *a_strides, const uint32_t *b_strides, size_t a_offset, size_t b_offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_broadcast_f32_pipeline(ctx ? ctx.mulBroadcastF32Pipeline : nil, a_handle, b_handle, out_handle, ndim, shape, a_strides, b_strides, a_offset, b_offset, len);
}

int affon_metal_div_broadcast_f32(void *a_handle, void *b_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *a_strides, const uint32_t *b_strides, size_t a_offset, size_t b_offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_broadcast_f32_pipeline(ctx ? ctx.divBroadcastF32Pipeline : nil, a_handle, b_handle, out_handle, ndim, shape, a_strides, b_strides, a_offset, b_offset, len);
}

int affon_metal_eq_broadcast_f32(void *a_handle, void *b_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *a_strides, const uint32_t *b_strides, size_t a_offset, size_t b_offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_broadcast_f32_pipeline(ctx ? ctx.eqBroadcastF32Pipeline : nil, a_handle, b_handle, out_handle, ndim, shape, a_strides, b_strides, a_offset, b_offset, len);
}

int affon_metal_lt_broadcast_f32(void *a_handle, void *b_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *a_strides, const uint32_t *b_strides, size_t a_offset, size_t b_offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_broadcast_f32_pipeline(ctx ? ctx.ltBroadcastF32Pipeline : nil, a_handle, b_handle, out_handle, ndim, shape, a_strides, b_strides, a_offset, b_offset, len);
}

int affon_metal_gt_broadcast_f32(void *a_handle, void *b_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *a_strides, const uint32_t *b_strides, size_t a_offset, size_t b_offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_binary_broadcast_f32_pipeline(ctx ? ctx.gtBroadcastF32Pipeline : nil, a_handle, b_handle, out_handle, ndim, shape, a_strides, b_strides, a_offset, b_offset, len);
}

int affon_metal_exp_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.expF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_log_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.logF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_sqrt_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.sqrtF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_abs_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.absF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_abs_i64(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.absI64Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_sign_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.signF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_sign_i64(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.signI64Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_gelu_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.geluF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_gelu_grad_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.geluGradF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_neg_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.negF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_neg_i64(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.negI64Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_relu_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.reluF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_relu_i64(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.reluI64Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_sigmoid_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.sigmoidF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_silu_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.siluF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_tanh_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_unary_f32_pipeline(ctx ? ctx.tanhF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_clamp_f32(void *a_handle, void *out_handle, size_t len, float min_val, float max_val) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.clampF32Pipeline || !a_handle || !out_handle) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.clampF32Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:&min_val length:sizeof(min_val) atIndex:2];
    [encoder setBytes:&max_val length:sizeof(max_val) atIndex:3];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.clampF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_unary_chain_f32(void *input_handle, void *out_handle, const uint32_t *op_codes, const float *clamp_mins, const float *clamp_maxs, size_t stage_count, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.unaryChainF32Pipeline || !input_handle || !out_handle || !op_codes || !clamp_mins || !clamp_maxs) return -1;
    uint32_t stage32 = (uint32_t)stage_count;
    uint32_t len32 = (uint32_t)len;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    if (!cb) return -2;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return -3;
    [enc setComputePipelineState:ctx.unaryChainF32Pipeline];
    [enc setBuffer:affon_metal_borrow_buffer(input_handle) offset:0 atIndex:0];
    [enc setBuffer:affon_metal_borrow_buffer(out_handle) offset:0 atIndex:1];
    [enc setBytes:op_codes length:stage_count * sizeof(uint32_t) atIndex:2];
    [enc setBytes:clamp_mins length:stage_count * sizeof(float) atIndex:3];
    [enc setBytes:clamp_maxs length:stage_count * sizeof(float) atIndex:4];
    [enc setBytes:&stage32 length:sizeof(stage32) atIndex:5];
    [enc setBytes:&len32 length:sizeof(len32) atIndex:6];
    NSUInteger w = ctx.unaryChainF32Pipeline.threadExecutionWidth;
    if (w == 0) w = 64;
    [enc dispatchThreads:MTLSizeMake(len, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    [enc endEncoding];
    return affon_metal_finish_command_buffer(cb, -4);
    }
}

int affon_metal_unary_chain_i64(void *input_handle, void *out_handle, const uint32_t *op_codes, const long *clamp_mins, const long *clamp_maxs, size_t stage_count, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.unaryChainI64Pipeline || !input_handle || !out_handle || !op_codes || !clamp_mins || !clamp_maxs) return -1;
    uint32_t stage32 = (uint32_t)stage_count;
    uint32_t len32 = (uint32_t)len;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    if (!cb) return -2;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return -3;
    [enc setComputePipelineState:ctx.unaryChainI64Pipeline];
    [enc setBuffer:affon_metal_borrow_buffer(input_handle) offset:0 atIndex:0];
    [enc setBuffer:affon_metal_borrow_buffer(out_handle) offset:0 atIndex:1];
    [enc setBytes:op_codes length:stage_count * sizeof(uint32_t) atIndex:2];
    [enc setBytes:clamp_mins length:stage_count * sizeof(long) atIndex:3];
    [enc setBytes:clamp_maxs length:stage_count * sizeof(long) atIndex:4];
    [enc setBytes:&stage32 length:sizeof(stage32) atIndex:5];
    [enc setBytes:&len32 length:sizeof(len32) atIndex:6];
    NSUInteger w = ctx.unaryChainI64Pipeline.threadExecutionWidth;
    if (w == 0) w = 64;
    [enc dispatchThreads:MTLSizeMake(len, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    [enc endEncoding];
    return affon_metal_finish_command_buffer(cb, -4);
    }
}

int affon_metal_binary_then_unary_chain_f32(void *lhs_handle, void *rhs_handle, void *out_handle, uint32_t binary_code, const uint32_t *op_codes, const float *clamp_mins, const float *clamp_maxs, size_t stage_count, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.binaryThenUnaryChainF32Pipeline || !lhs_handle || !rhs_handle || !out_handle || !op_codes || !clamp_mins || !clamp_maxs) return -1;
    uint32_t stage32 = (uint32_t)stage_count;
    uint32_t len32 = (uint32_t)len;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    if (!cb) return -2;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return -3;
    [enc setComputePipelineState:ctx.binaryThenUnaryChainF32Pipeline];
    [enc setBuffer:affon_metal_borrow_buffer(lhs_handle) offset:0 atIndex:0];
    [enc setBuffer:affon_metal_borrow_buffer(rhs_handle) offset:0 atIndex:1];
    [enc setBuffer:affon_metal_borrow_buffer(out_handle) offset:0 atIndex:2];
    [enc setBytes:&binary_code length:sizeof(binary_code) atIndex:3];
    [enc setBytes:op_codes length:stage_count * sizeof(uint32_t) atIndex:4];
    [enc setBytes:clamp_mins length:stage_count * sizeof(float) atIndex:5];
    [enc setBytes:clamp_maxs length:stage_count * sizeof(float) atIndex:6];
    [enc setBytes:&stage32 length:sizeof(stage32) atIndex:7];
    [enc setBytes:&len32 length:sizeof(len32) atIndex:8];
    NSUInteger w = ctx.binaryThenUnaryChainF32Pipeline.threadExecutionWidth;
    if (w == 0) w = 64;
    [enc dispatchThreads:MTLSizeMake(len, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    [enc endEncoding];
    return affon_metal_finish_command_buffer(cb, -4);
    }
}

int affon_metal_binary_then_unary_chain_i64(void *lhs_handle, void *rhs_handle, void *out_handle, uint32_t binary_code, const uint32_t *op_codes, const long *clamp_mins, const long *clamp_maxs, size_t stage_count, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.binaryThenUnaryChainI64Pipeline || !lhs_handle || !rhs_handle || !out_handle || !op_codes || !clamp_mins || !clamp_maxs) return -1;
    uint32_t stage32 = (uint32_t)stage_count;
    uint32_t len32 = (uint32_t)len;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    if (!cb) return -2;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return -3;
    [enc setComputePipelineState:ctx.binaryThenUnaryChainI64Pipeline];
    [enc setBuffer:affon_metal_borrow_buffer(lhs_handle) offset:0 atIndex:0];
    [enc setBuffer:affon_metal_borrow_buffer(rhs_handle) offset:0 atIndex:1];
    [enc setBuffer:affon_metal_borrow_buffer(out_handle) offset:0 atIndex:2];
    [enc setBytes:&binary_code length:sizeof(binary_code) atIndex:3];
    [enc setBytes:op_codes length:stage_count * sizeof(uint32_t) atIndex:4];
    [enc setBytes:clamp_mins length:stage_count * sizeof(long) atIndex:5];
    [enc setBytes:clamp_maxs length:stage_count * sizeof(long) atIndex:6];
    [enc setBytes:&stage32 length:sizeof(stage32) atIndex:7];
    [enc setBytes:&len32 length:sizeof(len32) atIndex:8];
    NSUInteger w = ctx.binaryThenUnaryChainI64Pipeline.threadExecutionWidth;
    if (w == 0) w = 64;
    [enc dispatchThreads:MTLSizeMake(len, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    [enc endEncoding];
    return affon_metal_finish_command_buffer(cb, -4);
    }
}

static int affon_metal_run_cast_pipeline(id<MTLComputePipelineState> pipeline, void *input_handle, void *out_handle, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !input_handle || !out_handle) return -1;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    if (!cb) return -2;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return -3;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:affon_metal_borrow_buffer(input_handle) offset:0 atIndex:0];
    [enc setBuffer:affon_metal_borrow_buffer(out_handle) offset:0 atIndex:1];
    NSUInteger w = pipeline.threadExecutionWidth;
    if (w == 0) w = 64;
    [enc dispatchThreads:MTLSizeMake(len, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    [enc endEncoding];
    return affon_metal_finish_command_buffer(cb, -4);
    }
}

int affon_metal_cast_f32_f32(void *input_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_metal_run_cast_pipeline(ctx ? ctx.castF32F32Pipeline : nil, input_handle, out_handle, len);
}

int affon_metal_cast_i64_i64(void *input_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_metal_run_cast_pipeline(ctx ? ctx.castI64I64Pipeline : nil, input_handle, out_handle, len);
}

int affon_metal_cast_f32_i64(void *input_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_metal_run_cast_pipeline(ctx ? ctx.castF32I64Pipeline : nil, input_handle, out_handle, len);
}

int affon_metal_cast_i64_f32(void *input_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_metal_run_cast_pipeline(ctx ? ctx.castI64F32Pipeline : nil, input_handle, out_handle, len);
}

int affon_metal_validate_cast_index_f32_i64(void *input_handle, void *out_handle, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.validateCastIndexF32I64Pipeline || !input_handle || !out_handle) return -1;
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    if (!cb) return -3;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return -4;
    [enc setComputePipelineState:ctx.validateCastIndexF32I64Pipeline];
    [enc setBuffer:affon_metal_borrow_buffer(input_handle) offset:0 atIndex:0];
    [enc setBuffer:affon_metal_borrow_buffer(out_handle) offset:0 atIndex:1];
    uint32_t len32 = (uint32_t)len;
    [enc setBytes:&len32 length:sizeof(len32) atIndex:2];
    [enc setBuffer:statusBuffer offset:0 atIndex:3];
    NSUInteger w = ctx.validateCastIndexF32I64Pipeline.threadExecutionWidth;
    if (w == 0) w = 64;
    [enc dispatchThreads:MTLSizeMake(len, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    [enc endEncoding];
    return affon_metal_finish_command_buffer_with_status(cb, statusBuffer, -5, 1);
    }
}

int affon_metal_clamp_i64(void *a_handle, void *out_handle, size_t len, int64_t min_val, int64_t max_val) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.clampI64Pipeline || !a_handle || !out_handle) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.clampI64Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:&min_val length:sizeof(min_val) atIndex:2];
    [encoder setBytes:&max_val length:sizeof(max_val) atIndex:3];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.clampI64Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_clamp_grad_f32(void *a_handle, void *out_handle, size_t len, float min_val, float max_val) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.clampGradF32Pipeline || !a_handle || !out_handle) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.clampGradF32Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:&min_val length:sizeof(min_val) atIndex:2];
    [encoder setBytes:&max_val length:sizeof(max_val) atIndex:3];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.clampGradF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_scatter_add_f32(void *base_handle, void *index_handle, void *src_handle, void *out_handle, size_t ndim, const uint32_t *dst_shape, const uint32_t *src_shape, size_t axis, size_t src_len, size_t dst_len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.scatterAddF32Pipeline || !base_handle || !index_handle || !src_handle || !out_handle) return -1;

    AffonMetalBuffer *base = (__bridge AffonMetalBuffer *)base_handle;
    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *src = (__bridge AffonMetalBuffer *)src_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!blit) return -3;
    [blit copyFromBuffer:base.buffer sourceOffset:0 toBuffer:out.buffer destinationOffset:0 size:dst_len * sizeof(float)];
    [blit endEncoding];

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    uint32_t ndim_u32 = (uint32_t)ndim;
    uint32_t axis_u32 = (uint32_t)axis;
    uint32_t src_len_u32 = (uint32_t)src_len;
    uint32_t dst_len_u32 = (uint32_t)dst_len;

    [encoder setComputePipelineState:ctx.scatterAddF32Pipeline];
    [encoder setBuffer:index.buffer offset:0 atIndex:0];
    [encoder setBuffer:src.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:dst_shape length:sizeof(uint32_t) * ndim_u32 atIndex:3];
    [encoder setBytes:src_shape length:sizeof(uint32_t) * ndim_u32 atIndex:4];
    [encoder setBytes:&ndim_u32 length:sizeof(ndim_u32) atIndex:5];
    [encoder setBytes:&axis_u32 length:sizeof(axis_u32) atIndex:6];
    [encoder setBytes:&src_len_u32 length:sizeof(src_len_u32) atIndex:7];
    [encoder setBytes:&dst_len_u32 length:sizeof(dst_len_u32) atIndex:8];

    MTLSize gridSize = MTLSizeMake(dst_len, 1, 1);
    NSUInteger threadWidth = ctx.scatterAddF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > dst_len && dst_len > 0) threadWidth = dst_len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_scatter_add_i64_f32(void *base_handle, void *index_handle, void *src_handle, void *out_handle, size_t ndim, const uint32_t *dst_shape, const uint32_t *src_shape, size_t axis, size_t src_len, size_t dst_len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.scatterAddI64F32Pipeline || !base_handle || !index_handle || !src_handle || !out_handle) return -1;

    AffonMetalBuffer *base = (__bridge AffonMetalBuffer *)base_handle;
    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *src = (__bridge AffonMetalBuffer *)src_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!blit) return -3;
    [blit copyFromBuffer:base.buffer sourceOffset:0 toBuffer:out.buffer destinationOffset:0 size:dst_len * sizeof(float)];
    [blit endEncoding];

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    uint32_t ndim_u32 = (uint32_t)ndim;
    uint32_t axis_u32 = (uint32_t)axis;
    uint32_t src_len_u32 = (uint32_t)src_len;
    uint32_t dst_len_u32 = (uint32_t)dst_len;

    [encoder setComputePipelineState:ctx.scatterAddI64F32Pipeline];
    [encoder setBuffer:index.buffer offset:0 atIndex:0];
    [encoder setBuffer:src.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:dst_shape length:sizeof(uint32_t) * ndim_u32 atIndex:3];
    [encoder setBytes:src_shape length:sizeof(uint32_t) * ndim_u32 atIndex:4];
    [encoder setBytes:&ndim_u32 length:sizeof(ndim_u32) atIndex:5];
    [encoder setBytes:&axis_u32 length:sizeof(axis_u32) atIndex:6];
    [encoder setBytes:&src_len_u32 length:sizeof(src_len_u32) atIndex:7];
    [encoder setBytes:&dst_len_u32 length:sizeof(dst_len_u32) atIndex:8];

    MTLSize gridSize = MTLSizeMake(dst_len, 1, 1);
    NSUInteger threadWidth = ctx.scatterAddI64F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > dst_len && dst_len > 0) threadWidth = dst_len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_softmax_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.softmaxF32Pipeline || !a_handle || !out_handle || rows == 0 || cols == 0 || axis > 1) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t cols32 = (uint32_t)cols;
    uint32_t axis32 = (uint32_t)axis;
    size_t work_items = axis == 0 ? cols : rows;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.softmaxF32Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:2];
    [encoder setBytes:&cols32 length:sizeof(cols32) atIndex:3];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:4];

    NSUInteger threadWidth = ctx.softmaxF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > work_items && work_items > 0) threadWidth = work_items;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize gridSize = MTLSizeMake(work_items, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_softmax_nd_f32(void *a_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *input_strides, size_t axis, size_t axis_size, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.softmaxNdF32Pipeline || !a_handle || !out_handle || !shape || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim || axis_size == 0) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.softmaxNdF32Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:2];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:4];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:5];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:6];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:7];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.softmaxNdF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_log_softmax_nll_nd_f32(void *logits_handle, void *targets_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *input_strides, size_t axis, size_t axis_size, size_t groups) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.logSoftmaxNllNdF32Pipeline || !logits_handle || !targets_handle || !out_handle || !shape || !input_strides) return -1;
    if (ndim < 2 || ndim > affon_max_broadcast_dims || axis >= ndim || axis_size == 0 || groups == 0) return -1;

    AffonMetalBuffer *logits = (__bridge AffonMetalBuffer *)logits_handle;
    AffonMetalBuffer *targets = (__bridge AffonMetalBuffer *)targets_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t groups32 = (uint32_t)groups;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.logSoftmaxNllNdF32Pipeline];
    [encoder setBuffer:logits.buffer offset:0 atIndex:0];
    [encoder setBuffer:targets.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:5];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:6];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:7];
    [encoder setBytes:&groups32 length:sizeof(groups32) atIndex:8];

    const NSUInteger one = 1;
    MTLSize gridSize = MTLSizeMake(one, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(one, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_cross_entropy_indexed_f32(void *logits_handle, void *targets_handle, void *out_handle, size_t rows, size_t classes) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    NSUInteger tgsize = (NSUInteger)affon_config_metal_threadgroup_size();
    if (!ctx || !ctx.crossEntropyIndexedF32Pipeline || !ctx.reduceMeanParallelF32Pipeline || !logits_handle || !targets_handle || !out_handle) return -1;
    if (rows == 0 || classes == 0) return -1;

    AffonMetalBuffer *logits = (__bridge AffonMetalBuffer *)logits_handle;
    AffonMetalBuffer *targets = (__bridge AffonMetalBuffer *)targets_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t classes32 = (uint32_t)classes;
    id<MTLBuffer> rowLossesBuffer = [ctx.device newBufferWithLength:rows * sizeof(float) options:MTLResourceStorageModeShared];
    if (!rowLossesBuffer) return -2;
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.crossEntropyIndexedF32Pipeline];
    [encoder setBuffer:logits.buffer offset:0 atIndex:0];
    [encoder setBuffer:targets.buffer offset:0 atIndex:1];
    [encoder setBuffer:rowLossesBuffer offset:0 atIndex:2];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:3];
    [encoder setBytes:&classes32 length:sizeof(classes32) atIndex:4];
    [encoder setBuffer:statusBuffer offset:0 atIndex:5];
    [encoder setThreadgroupMemoryLength:tgsize * sizeof(float) atIndex:0];

    MTLSize gridSize = MTLSizeMake(rows, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(tgsize, 1, 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    id<MTLComputeCommandEncoder> reduceEncoder = [commandBuffer computeCommandEncoder];
    if (!reduceEncoder) return -4;
    [reduceEncoder setComputePipelineState:ctx.reduceMeanParallelF32Pipeline];
    [reduceEncoder setBuffer:rowLossesBuffer offset:0 atIndex:0];
    [reduceEncoder setBuffer:out.buffer offset:0 atIndex:1];
    [reduceEncoder setBytes:&rows32 length:sizeof(rows32) atIndex:2];
    [reduceEncoder setThreadgroupMemoryLength:tgsize * sizeof(float) atIndex:0];
    [reduceEncoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:threadgroupSize];
    [reduceEncoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_cross_entropy_indexed_transposed_f32(void *logits_handle, void *targets_handle, void *out_handle, size_t rows, size_t classes) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    NSUInteger tgsize = (NSUInteger)affon_config_metal_threadgroup_size();
    if (!ctx || !ctx.crossEntropyIndexedTransposedF32Pipeline || !ctx.reduceMeanParallelF32Pipeline || !logits_handle || !targets_handle || !out_handle) return -1;
    if (rows == 0 || classes == 0) return -1;

    AffonMetalBuffer *logits = (__bridge AffonMetalBuffer *)logits_handle;
    AffonMetalBuffer *targets = (__bridge AffonMetalBuffer *)targets_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t classes32 = (uint32_t)classes;
    id<MTLBuffer> rowLossesBuffer = [ctx.device newBufferWithLength:rows * sizeof(float) options:MTLResourceStorageModeShared];
    if (!rowLossesBuffer) return -2;
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.crossEntropyIndexedTransposedF32Pipeline];
    [encoder setBuffer:logits.buffer offset:0 atIndex:0];
    [encoder setBuffer:targets.buffer offset:0 atIndex:1];
    [encoder setBuffer:rowLossesBuffer offset:0 atIndex:2];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:3];
    [encoder setBytes:&classes32 length:sizeof(classes32) atIndex:4];
    [encoder setBuffer:statusBuffer offset:0 atIndex:5];
    [encoder setThreadgroupMemoryLength:tgsize * sizeof(float) atIndex:0];

    MTLSize gridSize = MTLSizeMake(rows, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(tgsize, 1, 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    id<MTLComputeCommandEncoder> reduceEncoder = [commandBuffer computeCommandEncoder];
    if (!reduceEncoder) return -4;
    [reduceEncoder setComputePipelineState:ctx.reduceMeanParallelF32Pipeline];
    [reduceEncoder setBuffer:rowLossesBuffer offset:0 atIndex:0];
    [reduceEncoder setBuffer:out.buffer offset:0 atIndex:1];
    [reduceEncoder setBytes:&rows32 length:sizeof(rows32) atIndex:2];
    [reduceEncoder setThreadgroupMemoryLength:tgsize * sizeof(float) atIndex:0];
    [reduceEncoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:threadgroupSize];
    [reduceEncoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_cross_entropy_indexed_backward_f32(void *logits_handle, void *targets_handle, void *grad_out_handle, void *out_handle, size_t rows, size_t classes) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    NSUInteger tgsize = (NSUInteger)affon_config_metal_threadgroup_size();
    if (!ctx || !ctx.crossEntropyIndexedBackwardF32Pipeline || !logits_handle || !targets_handle || !grad_out_handle || !out_handle) return -1;
    if (rows == 0 || classes == 0) return -1;

    AffonMetalBuffer *logits = (__bridge AffonMetalBuffer *)logits_handle;
    AffonMetalBuffer *targets = (__bridge AffonMetalBuffer *)targets_handle;
    AffonMetalBuffer *grad_out = (__bridge AffonMetalBuffer *)grad_out_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t classes32 = (uint32_t)classes;
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.crossEntropyIndexedBackwardF32Pipeline];
    [encoder setBuffer:logits.buffer offset:0 atIndex:0];
    [encoder setBuffer:targets.buffer offset:0 atIndex:1];
    [encoder setBuffer:grad_out.buffer offset:0 atIndex:2];
    [encoder setBuffer:out.buffer offset:0 atIndex:3];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:4];
    [encoder setBytes:&classes32 length:sizeof(classes32) atIndex:5];
    [encoder setBuffer:statusBuffer offset:0 atIndex:6];
    [encoder setThreadgroupMemoryLength:tgsize * sizeof(float) atIndex:0];

    MTLSize gridSize = MTLSizeMake(rows, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(tgsize, 1, 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_dot_f32(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.dotF32Pipeline || !a_handle || !b_handle || !out_handle || len == 0) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *b = (__bridge AffonMetalBuffer *)b_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.dotF32Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:b.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:3];

    MTLSize gridSize = MTLSizeMake(1, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_dot_i64(void *a_handle, void *b_handle, void *out_handle, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.dotI64Pipeline || !a_handle || !b_handle || !out_handle || len == 0) return -1;

    id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
    id<MTLBuffer> b = affon_metal_borrow_buffer(b_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!a || !b || !out) return -1;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.dotI64Pipeline];
    [encoder setBuffer:a offset:0 atIndex:0];
    [encoder setBuffer:b offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:3];

    MTLSize gridSize = MTLSizeMake(1, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}


int affon_metal_where_f32(void *cond_handle, void *a_handle, void *b_handle, void *out_handle, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.whereF32Pipeline || !cond_handle || !a_handle || !b_handle || !out_handle) return -1;

    AffonMetalBuffer *cond = (__bridge AffonMetalBuffer *)cond_handle;
    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *b = (__bridge AffonMetalBuffer *)b_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.whereF32Pipeline];
    [encoder setBuffer:cond.buffer offset:0 atIndex:0];
    [encoder setBuffer:a.buffer offset:0 atIndex:1];
    [encoder setBuffer:b.buffer offset:0 atIndex:2];
    [encoder setBuffer:out.buffer offset:0 atIndex:3];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.whereF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_where_i64_f32(void *cond_handle, void *a_handle, void *b_handle, void *out_handle, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.whereI64F32Pipeline || !cond_handle || !a_handle || !b_handle || !out_handle) return -1;

    AffonMetalBuffer *cond = (__bridge AffonMetalBuffer *)cond_handle;
    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *b = (__bridge AffonMetalBuffer *)b_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.whereI64F32Pipeline];
    [encoder setBuffer:cond.buffer offset:0 atIndex:0];
    [encoder setBuffer:a.buffer offset:0 atIndex:1];
    [encoder setBuffer:b.buffer offset:0 atIndex:2];
    [encoder setBuffer:out.buffer offset:0 atIndex:3];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.whereI64F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_where_f32_i64(void *cond_handle, void *a_handle, void *b_handle, void *out_handle, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.whereF32I64Pipeline || !cond_handle || !a_handle || !b_handle || !out_handle) return -1;

    AffonMetalBuffer *cond = (__bridge AffonMetalBuffer *)cond_handle;
    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *b = (__bridge AffonMetalBuffer *)b_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.whereF32I64Pipeline];
    [encoder setBuffer:cond.buffer offset:0 atIndex:0];
    [encoder setBuffer:a.buffer offset:0 atIndex:1];
    [encoder setBuffer:b.buffer offset:0 atIndex:2];
    [encoder setBuffer:out.buffer offset:0 atIndex:3];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.whereF32I64Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_where_i64_i64(void *cond_handle, void *a_handle, void *b_handle, void *out_handle, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.whereI64I64Pipeline || !cond_handle || !a_handle || !b_handle || !out_handle) return -1;

    AffonMetalBuffer *cond = (__bridge AffonMetalBuffer *)cond_handle;
    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *b = (__bridge AffonMetalBuffer *)b_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.whereI64I64Pipeline];
    [encoder setBuffer:cond.buffer offset:0 atIndex:0];
    [encoder setBuffer:a.buffer offset:0 atIndex:1];
    [encoder setBuffer:b.buffer offset:0 atIndex:2];
    [encoder setBuffer:out.buffer offset:0 atIndex:3];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.whereI64I64Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_masked_fill_i64_f32(void *input_handle, void *mask_handle, void *out_handle, float fill_value, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.maskedFillI64F32Pipeline || !input_handle || !mask_handle || !out_handle) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *mask = (__bridge AffonMetalBuffer *)mask_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.maskedFillI64F32Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:mask.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:&fill_value length:sizeof(fill_value) atIndex:3];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:4];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.maskedFillI64F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_masked_fill_i64_i64(void *input_handle, void *mask_handle, void *out_handle, int64_t fill_value, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.maskedFillI64I64Pipeline || !input_handle || !mask_handle || !out_handle) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *mask = (__bridge AffonMetalBuffer *)mask_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.maskedFillI64I64Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:mask.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:&fill_value length:sizeof(fill_value) atIndex:3];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:4];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.maskedFillI64I64Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_where_broadcast_f32(void *cond_handle, void *a_handle, void *b_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *cond_strides, const uint32_t *a_strides, const uint32_t *b_strides, size_t cond_offset, size_t a_offset, size_t b_offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_where_broadcast_f32_pipeline(ctx ? ctx.whereBroadcastF32Pipeline : nil, cond_handle, a_handle, b_handle, out_handle, ndim, shape, cond_strides, a_strides, b_strides, cond_offset, a_offset, b_offset, len);
}

int affon_metal_where_broadcast_i64_f32(void *cond_handle, void *a_handle, void *b_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *cond_strides, const uint32_t *a_strides, const uint32_t *b_strides, size_t cond_offset, size_t a_offset, size_t b_offset, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_where_broadcast_f32_pipeline(ctx ? ctx.whereBroadcastI64F32Pipeline : nil, cond_handle, a_handle, b_handle, out_handle, ndim, shape, cond_strides, a_strides, b_strides, cond_offset, a_offset, b_offset, len);
}

int affon_metal_masked_fill_broadcast_i64_f32(void *input_handle, void *mask_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *input_strides, const uint32_t *mask_strides, size_t input_offset, size_t mask_offset, float fill_value, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_masked_fill_broadcast_pipeline(ctx ? ctx.maskedFillBroadcastI64F32Pipeline : nil, input_handle, mask_handle, out_handle, ndim, shape, input_strides, mask_strides, input_offset, mask_offset, &fill_value, sizeof(fill_value), len);
}

int affon_metal_masked_fill_broadcast_i64_i64(void *input_handle, void *mask_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *input_strides, const uint32_t *mask_strides, size_t input_offset, size_t mask_offset, int64_t fill_value, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_masked_fill_broadcast_pipeline(ctx ? ctx.maskedFillBroadcastI64I64Pipeline : nil, input_handle, mask_handle, out_handle, ndim, shape, input_strides, mask_strides, input_offset, mask_offset, &fill_value, sizeof(fill_value), len);
}

static int affon_run_reduce_f32_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *a_handle,
    void *out_handle,
    size_t len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !a_handle || !out_handle || len == 0) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:2];

    MTLSize gridSize = MTLSizeMake(1, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_reduce_sum_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceSumF32Pipeline : nil, a_handle, out_handle, len);
}
int affon_metal_reduce_sum_i64(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceSumI64Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_reduce_mean_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceMeanF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_reduce_min_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceMinF32Pipeline : nil, a_handle, out_handle, len);
}
int affon_metal_reduce_min_i64(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceMinI64Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_reduce_max_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceMaxF32Pipeline : nil, a_handle, out_handle, len);
}
int affon_metal_reduce_max_i64(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceMaxI64Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_reduce_argmin_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceArgminF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_reduce_argmax_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceArgmaxF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_reduce_argmin_i64_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceArgminI64F32Pipeline : nil, a_handle, out_handle, len);
}
int affon_metal_reduce_argmin_i64_i64(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceArgminI64I64Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_reduce_argmax_i64_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceArgmaxI64F32Pipeline : nil, a_handle, out_handle, len);
}
int affon_metal_reduce_argmax_i64_i64(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceArgmaxI64I64Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_reduce_variance_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceVarianceF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_reduce_std_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_f32_pipeline(ctx ? ctx.reduceStdF32Pipeline : nil, a_handle, out_handle, len);
}

static int affon_run_reduce_axis_f32_pipeline(
    id<MTLComputePipelineState> pipeline,
    void *a_handle,
    void *out_handle,
    size_t rows,
    size_t cols,
    size_t axis
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !a_handle || !out_handle || rows == 0 || cols == 0 || axis > 1) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t cols32 = (uint32_t)cols;
    uint32_t axis32 = (uint32_t)axis;
    size_t out_len = axis == 0 ? cols : rows;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:2];
    [encoder setBytes:&cols32 length:sizeof(cols32) atIndex:3];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:4];

    NSUInteger threadWidth = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > out_len && out_len > 0) threadWidth = out_len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize gridSize = MTLSizeMake(out_len, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_reduce_axis_sum_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisSumF32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}
int affon_metal_reduce_axis_sum_i64(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisSumI64Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}

int affon_metal_reduce_axis_mean_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisMeanF32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}

int affon_metal_reduce_axis_min_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisMinF32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}
int affon_metal_reduce_axis_min_i64(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisMinI64Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}

int affon_metal_reduce_axis_max_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisMaxF32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}
int affon_metal_reduce_axis_max_i64(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisMaxI64Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}

int affon_metal_reduce_axis_variance_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisVarianceF32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}

int affon_metal_reduce_axis_std_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisStdF32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}

int affon_metal_reduce_axis_argmin_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisArgminF32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}

int affon_metal_reduce_axis_argmax_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisArgmaxF32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}

int affon_metal_reduce_axis_argmin_i64_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisArgminI64F32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}
int affon_metal_reduce_axis_argmin_i64_i64(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisArgminI64I64Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}

int affon_metal_reduce_axis_argmax_i64_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisArgmaxI64F32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}
int affon_metal_reduce_axis_argmax_i64_i64(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_f32_pipeline(ctx ? ctx.reduceAxisArgmaxI64I64Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}



int affon_metal_add_layer_norm_nd_f32(void *a_handle, void *b_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *input_strides, size_t axis, size_t axis_size, size_t len, float eps) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.addLayerNormNdF32Pipeline || !a_handle || !b_handle || !out_handle || !shape || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim || axis_size == 0) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *b = (__bridge AffonMetalBuffer *)b_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.addLayerNormNdF32Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:b.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:5];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:6];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:7];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:8];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:9];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.addLayerNormNdF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}
int affon_metal_layer_norm_nd_f32(void *a_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *input_strides, size_t axis, size_t axis_size, size_t len, float eps) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.layerNormNdF32Pipeline || !a_handle || !out_handle || !shape || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim || axis_size == 0) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.layerNormNdF32Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:2];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:4];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:5];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:6];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:7];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:8];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.layerNormNdF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}
int affon_metal_rms_norm_nd_f32(void *a_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *input_strides, size_t axis, size_t axis_size, size_t len, float eps) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.rmsNormNdF32Pipeline || !a_handle || !out_handle || !shape || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim || axis_size == 0) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.rmsNormNdF32Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:2];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:4];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:5];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:6];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:7];
    [encoder setBytes:&eps length:sizeof(eps) atIndex:8];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.rmsNormNdF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}
int affon_metal_reduce_axis_nd_f32(
    void *a_handle,
    void *out_handle,
    size_t ndim,
    const uint32_t *out_shape,
    const uint32_t *out_strides,
    const uint32_t *input_strides,
    size_t axis,
    size_t axis_size,
    size_t reduce_kind,
    size_t out_len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.reduceAxisNdF32Pipeline || !a_handle || !out_handle || !out_shape || !out_strides || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim || axis_size == 0) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t kind32 = (uint32_t)reduce_kind;
    uint32_t outLen32 = (uint32_t)out_len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.reduceAxisNdF32Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:out_shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:2];
    [encoder setBytes:out_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:5];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:6];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:7];
    [encoder setBytes:&kind32 length:sizeof(kind32) atIndex:8];
    [encoder setBytes:&outLen32 length:sizeof(outLen32) atIndex:9];

    MTLSize gridSize = MTLSizeMake(out_len, 1, 1);
    NSUInteger threadWidth = ctx.reduceAxisNdF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > out_len && out_len > 0) threadWidth = out_len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_reduce_axis_nd_i64(
    void *a_handle,
    void *out_handle,
    size_t ndim,
    const uint32_t *out_shape,
    const uint32_t *out_strides,
    const uint32_t *input_strides,
    size_t axis,
    size_t axis_size,
    size_t reduce_kind,
    size_t out_len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.reduceAxisNdI64Pipeline || !a_handle || !out_handle || !out_shape || !out_strides || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim || axis_size == 0) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t kind32 = (uint32_t)reduce_kind;
    uint32_t outLen32 = (uint32_t)out_len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.reduceAxisNdI64Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:out_shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:2];
    [encoder setBytes:out_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:5];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:6];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:7];
    [encoder setBytes:&kind32 length:sizeof(kind32) atIndex:8];
    [encoder setBytes:&outLen32 length:sizeof(outLen32) atIndex:9];

    MTLSize gridSize = MTLSizeMake(out_len, 1, 1);
    NSUInteger threadWidth = ctx.reduceAxisNdI64Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > out_len && out_len > 0) threadWidth = out_len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_reduce_axis_nd_arg_i64_f32(
    void *a_handle,
    void *out_handle,
    size_t ndim,
    const uint32_t *out_shape,
    const uint32_t *out_strides,
    const uint32_t *input_strides,
    size_t axis,
    size_t axis_size,
    size_t choose_max,
    size_t out_len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.reduceAxisNdArgI64F32Pipeline || !a_handle || !out_handle || !out_shape || !out_strides || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim || axis_size == 0) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t chooseMax32 = (uint32_t)choose_max;
    uint32_t outLen32 = (uint32_t)out_len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.reduceAxisNdArgI64F32Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:out_shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:2];
    [encoder setBytes:out_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:5];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:6];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:7];
    [encoder setBytes:&chooseMax32 length:sizeof(chooseMax32) atIndex:8];
    [encoder setBytes:&outLen32 length:sizeof(outLen32) atIndex:9];

    MTLSize gridSize = MTLSizeMake(out_len, 1, 1);
    NSUInteger threadWidth = ctx.reduceAxisNdArgI64F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > out_len && out_len > 0) threadWidth = out_len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_reduce_axis_nd_arg_i64_i64(
    void *a_handle,
    void *out_handle,
    size_t ndim,
    const uint32_t *out_shape,
    const uint32_t *out_strides,
    const uint32_t *input_strides,
    size_t axis,
    size_t axis_size,
    size_t choose_max,
    size_t out_len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.reduceAxisNdArgI64I64Pipeline || !a_handle || !out_handle || !out_shape || !out_strides || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim || axis_size == 0) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t chooseMax32 = (uint32_t)choose_max;
    uint32_t outLen32 = (uint32_t)out_len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.reduceAxisNdArgI64I64Pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:out_shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:2];
    [encoder setBytes:out_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:5];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:6];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:7];
    [encoder setBytes:&chooseMax32 length:sizeof(chooseMax32) atIndex:8];
    [encoder setBytes:&outLen32 length:sizeof(outLen32) atIndex:9];

    MTLSize gridSize = MTLSizeMake(out_len, 1, 1);
    NSUInteger threadWidth = ctx.reduceAxisNdArgI64I64Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > out_len && out_len > 0) threadWidth = out_len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

static int affon_run_reduce_parallel_f32(
    id<MTLComputePipelineState> pipeline,
    void *a_handle,
    void *out_handle,
    size_t len
) {
    @autoreleasepool {
    NSUInteger tgsize = (NSUInteger)affon_config_metal_threadgroup_size();
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !a_handle || !out_handle || len == 0) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:2];
    [encoder setThreadgroupMemoryLength:tgsize * sizeof(float) atIndex:0];

    MTLSize threadgroupSize = MTLSizeMake(tgsize, 1, 1);
    MTLSize gridSize = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_reduce_sum_parallel_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_parallel_f32(ctx ? ctx.reduceSumParallelF32Pipeline : nil, a_handle, out_handle, len);
}

int affon_metal_reduce_mean_parallel_f32(void *a_handle, void *out_handle, size_t len) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_parallel_f32(ctx ? ctx.reduceMeanParallelF32Pipeline : nil, a_handle, out_handle, len);
}

static int affon_run_reduce_axis_parallel_f32(
    id<MTLComputePipelineState> pipeline,
    void *a_handle,
    void *out_handle,
    size_t rows,
    size_t cols,
    size_t axis
) {
    @autoreleasepool {
    NSUInteger tgsize = (NSUInteger)affon_config_metal_threadgroup_size();
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !pipeline || !a_handle || !out_handle || rows == 0 || cols == 0 || axis > 1) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t cols32 = (uint32_t)cols;
    uint32_t axis32 = (uint32_t)axis;
    size_t num_tgs = (axis == 0) ? cols : rows;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:2];
    [encoder setBytes:&cols32 length:sizeof(cols32) atIndex:3];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:4];
    [encoder setThreadgroupMemoryLength:tgsize * sizeof(float) atIndex:0];

    MTLSize threadgroupSize = MTLSizeMake(tgsize, 1, 1);
    MTLSize gridSize = MTLSizeMake(num_tgs, 1, 1);
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_reduce_axis_sum_parallel_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_parallel_f32(ctx ? ctx.reduceAxisSumParallelF32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}

int affon_metal_reduce_axis_mean_parallel_f32(void *a_handle, void *out_handle, size_t rows, size_t cols, size_t axis) {
    AffonMetalContext *ctx = affon_get_context();
    return affon_run_reduce_axis_parallel_f32(ctx ? ctx.reduceAxisMeanParallelF32Pipeline : nil, a_handle, out_handle, rows, cols, axis);
}

int affon_metal_matmul_f32(void *a_handle, void *b_handle, void *out_handle, size_t m, size_t n, size_t k) {
    return affon_metal_matmul_offset_f32(a_handle, b_handle, out_handle, 0, 0, 0, m, n, k);
}

int affon_metal_matmul_i64(void *a_handle, void *b_handle, void *out_handle, size_t m, size_t n, size_t k) {
    return affon_metal_matmul_offset_i64(a_handle, b_handle, out_handle, 0, 0, 0, m, n, k);
}

int affon_metal_matmul_add_f32(void *a_handle, void *b_handle, void *bias_handle, void *out_handle, size_t m, size_t n, size_t k);
int affon_metal_matmul_add_i64(void *a_handle, void *b_handle, void *bias_handle, void *out_handle, size_t m, size_t n, size_t k);
int affon_metal_matmul_add_offset_f32(void *a_handle, void *b_handle, void *bias_handle, void *out_handle, size_t a_offset_bytes, size_t b_offset_bytes, size_t bias_offset_bytes, size_t out_offset_bytes, size_t m, size_t n, size_t k);
int affon_metal_matmul_add_offset_i64(void *a_handle, void *b_handle, void *bias_handle, void *out_handle, size_t a_offset_bytes, size_t b_offset_bytes, size_t bias_offset_bytes, size_t out_offset_bytes, size_t m, size_t n, size_t k);
int affon_metal_matmul_add_gelu_f32(void *a_handle, void *b_handle, void *bias_handle, void *out_handle, size_t m, size_t n, size_t k);
int affon_metal_matmul_add_gelu_offset_f32(void *a_handle, void *b_handle, void *bias_handle, void *out_handle, size_t a_offset_bytes, size_t b_offset_bytes, size_t bias_offset_bytes, size_t out_offset_bytes, size_t m, size_t n, size_t k);

int affon_metal_matmul_add_f32(void *a_handle, void *b_handle, void *bias_handle, void *out_handle, size_t m, size_t n, size_t k) {
    return affon_metal_matmul_add_offset_f32(a_handle, b_handle, bias_handle, out_handle, 0, 0, 0, 0, m, n, k);
}

int affon_metal_matmul_add_i64(void *a_handle, void *b_handle, void *bias_handle, void *out_handle, size_t m, size_t n, size_t k) {
    return affon_metal_matmul_add_offset_i64(a_handle, b_handle, bias_handle, out_handle, 0, 0, 0, 0, m, n, k);
}

int affon_metal_matmul_add_gelu_f32(void *a_handle, void *b_handle, void *bias_handle, void *out_handle, size_t m, size_t n, size_t k) {
    return affon_metal_matmul_add_gelu_offset_f32(a_handle, b_handle, bias_handle, out_handle, 0, 0, 0, 0, m, n, k);
}

int affon_metal_attention_scores_f32(void *q_handle, void *k_t_handle, void *scale_handle, void *mask_handle, void *out_handle, float mask_fill_value, size_t m, size_t n, size_t k);
int affon_metal_attention_scores_offset_f32(void *q_handle, void *k_t_handle, void *scale_handle, void *mask_handle, void *out_handle, size_t q_offset_bytes, size_t k_t_offset_bytes, size_t scale_offset_bytes, size_t mask_offset_bytes, size_t out_offset_bytes, float mask_fill_value, size_t m, size_t n, size_t k);

int affon_metal_attention_scores_f32(void *q_handle, void *k_t_handle, void *scale_handle, void *mask_handle, void *out_handle, float mask_fill_value, size_t m, size_t n, size_t k) {
    return affon_metal_attention_scores_offset_f32(q_handle, k_t_handle, scale_handle, mask_handle, out_handle, 0, 0, 0, 0, 0, mask_fill_value, m, n, k);
}

int affon_metal_matmul_add_offset_f32(
    void *a_handle,
    void *b_handle,
    void *bias_handle,
    void *out_handle,
    size_t a_offset_bytes,
    size_t b_offset_bytes,
    size_t bias_offset_bytes,
    size_t out_offset_bytes,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.matmulAddF32Pipeline || !a_handle || !b_handle || !bias_handle || !out_handle) return -1;
    id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
    id<MTLBuffer> b = affon_metal_borrow_buffer(b_handle);
    id<MTLBuffer> bias = affon_metal_borrow_buffer(bias_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!a || !b || !bias || !out) return -1;
    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;
    [encoder setComputePipelineState:ctx.matmulAddF32Pipeline];
    [encoder setBuffer:a offset:a_offset_bytes atIndex:0];
    [encoder setBuffer:b offset:b_offset_bytes atIndex:1];
    [encoder setBuffer:bias offset:bias_offset_bytes atIndex:2];
    [encoder setBuffer:out offset:out_offset_bytes atIndex:3];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:4];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:5];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:6];
    NSUInteger w = ctx.matmulAddF32Pipeline.threadExecutionWidth;
    if (w == 0) w = 8;
    NSUInteger h = ctx.matmulAddF32Pipeline.maxTotalThreadsPerThreadgroup / w;
    if (h == 0) h = 1;
    if (h > 8) h = 8;
    [encoder dispatchThreads:MTLSizeMake(n, m, 1) threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
    [encoder endEncoding];
    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

int affon_metal_attention_scores_offset_f32(
    void *q_handle,
    void *k_t_handle,
    void *scale_handle,
    void *mask_handle,
    void *out_handle,
    size_t q_offset_bytes,
    size_t k_t_offset_bytes,
    size_t scale_offset_bytes,
    size_t mask_offset_bytes,
    size_t out_offset_bytes,
    float mask_fill_value,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.attentionScoresF32Pipeline || !q_handle || !k_t_handle || !scale_handle || !mask_handle || !out_handle) return -1;
    id<MTLBuffer> q = affon_metal_borrow_buffer(q_handle);
    id<MTLBuffer> k_t = affon_metal_borrow_buffer(k_t_handle);
    id<MTLBuffer> scale = affon_metal_borrow_buffer(scale_handle);
    id<MTLBuffer> mask = affon_metal_borrow_buffer(mask_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!q || !k_t || !scale || !mask || !out) return -1;
    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;
    [encoder setComputePipelineState:ctx.attentionScoresF32Pipeline];
    [encoder setBuffer:q offset:q_offset_bytes atIndex:0];
    [encoder setBuffer:k_t offset:k_t_offset_bytes atIndex:1];
    [encoder setBuffer:scale offset:scale_offset_bytes atIndex:2];
    [encoder setBuffer:mask offset:mask_offset_bytes atIndex:3];
    [encoder setBuffer:out offset:out_offset_bytes atIndex:4];
    [encoder setBytes:&mask_fill_value length:sizeof(mask_fill_value) atIndex:5];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:6];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:7];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:8];

    NSUInteger w = ctx.attentionScoresF32Pipeline.threadExecutionWidth;
    if (w == 0) w = 32;
    [encoder dispatchThreads:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    [encoder endEncoding];
    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

int affon_metal_matmul_add_offset_i64(
    void *a_handle,
    void *b_handle,
    void *bias_handle,
    void *out_handle,
    size_t a_offset_bytes,
    size_t b_offset_bytes,
    size_t bias_offset_bytes,
    size_t out_offset_bytes,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.matmulAddI64Pipeline || !a_handle || !b_handle || !bias_handle || !out_handle) return -1;
    id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
    id<MTLBuffer> b = affon_metal_borrow_buffer(b_handle);
    id<MTLBuffer> bias = affon_metal_borrow_buffer(bias_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!a || !b || !bias || !out) return -1;
    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;
    [encoder setComputePipelineState:ctx.matmulAddI64Pipeline];
    [encoder setBuffer:a offset:a_offset_bytes atIndex:0];
    [encoder setBuffer:b offset:b_offset_bytes atIndex:1];
    [encoder setBuffer:bias offset:bias_offset_bytes atIndex:2];
    [encoder setBuffer:out offset:out_offset_bytes atIndex:3];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:4];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:5];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:6];
    NSUInteger w = ctx.matmulAddI64Pipeline.threadExecutionWidth;
    if (w == 0) w = 8;
    NSUInteger h = ctx.matmulAddI64Pipeline.maxTotalThreadsPerThreadgroup / w;
    if (h == 0) h = 1;
    if (h > 8) h = 8;
    [encoder dispatchThreads:MTLSizeMake(n, m, 1) threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
    [encoder endEncoding];
    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

int affon_metal_matmul_add_gelu_offset_f32(
    void *a_handle,
    void *b_handle,
    void *bias_handle,
    void *out_handle,
    size_t a_offset_bytes,
    size_t b_offset_bytes,
    size_t bias_offset_bytes,
    size_t out_offset_bytes,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.matmulAddGeluF32Pipeline || !a_handle || !b_handle || !bias_handle || !out_handle) return -1;
    id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
    id<MTLBuffer> b = affon_metal_borrow_buffer(b_handle);
    id<MTLBuffer> bias = affon_metal_borrow_buffer(bias_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!a || !b || !bias || !out) return -1;
    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;
    [encoder setComputePipelineState:ctx.matmulAddGeluF32Pipeline];
    [encoder setBuffer:a offset:a_offset_bytes atIndex:0];
    [encoder setBuffer:b offset:b_offset_bytes atIndex:1];
    [encoder setBuffer:bias offset:bias_offset_bytes atIndex:2];
    [encoder setBuffer:out offset:out_offset_bytes atIndex:3];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:4];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:5];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:6];
    NSUInteger w = ctx.matmulAddGeluF32Pipeline.threadExecutionWidth;
    if (w == 0) w = 8;
    NSUInteger h = ctx.matmulAddGeluF32Pipeline.maxTotalThreadsPerThreadgroup / w;
    if (h == 0) h = 1;
    if (h > 8) h = 8;
    [encoder dispatchThreads:MTLSizeMake(n, m, 1) threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
    [encoder endEncoding];
    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

int affon_metal_embedding_i64_f32(void *table_handle, void *index_handle, void *out_handle, size_t vocab, size_t emb_dim, size_t index_count) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.embeddingI64F32Pipeline || !table_handle || !index_handle || !out_handle) return -1;
    uint32_t vocab32 = (uint32_t)vocab;
    uint32_t embDim32 = (uint32_t)emb_dim;
    uint32_t indexCount32 = (uint32_t)index_count;
    uint32_t len32 = (uint32_t)(index_count * emb_dim);
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    if (!cb) return -3;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return -4;
    [enc setComputePipelineState:ctx.embeddingI64F32Pipeline];
    [enc setBuffer:affon_metal_borrow_buffer(table_handle) offset:0 atIndex:0];
    [enc setBuffer:affon_metal_borrow_buffer(index_handle) offset:0 atIndex:1];
    [enc setBuffer:affon_metal_borrow_buffer(out_handle) offset:0 atIndex:2];
    [enc setBytes:&vocab32 length:sizeof(vocab32) atIndex:3];
    [enc setBytes:&embDim32 length:sizeof(embDim32) atIndex:4];
    [enc setBytes:&indexCount32 length:sizeof(indexCount32) atIndex:5];
    [enc setBuffer:statusBuffer offset:0 atIndex:6];
    NSUInteger w = ctx.embeddingI64F32Pipeline.threadExecutionWidth;
    if (w == 0) w = 64;
    [enc dispatchThreads:MTLSizeMake(len32, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    [enc endEncoding];
    return affon_metal_finish_command_buffer_with_status(cb, statusBuffer, -5, -6);
    }
}

int affon_metal_embedding_i64_i64(void *table_handle, void *index_handle, void *out_handle, size_t vocab, size_t emb_dim, size_t index_count) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.embeddingI64I64Pipeline || !table_handle || !index_handle || !out_handle) return -1;
    uint32_t vocab32 = (uint32_t)vocab;
    uint32_t embDim32 = (uint32_t)emb_dim;
    uint32_t indexCount32 = (uint32_t)index_count;
    uint32_t len32 = (uint32_t)(index_count * emb_dim);
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    if (!cb) return -3;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return -4;
    [enc setComputePipelineState:ctx.embeddingI64I64Pipeline];
    [enc setBuffer:affon_metal_borrow_buffer(table_handle) offset:0 atIndex:0];
    [enc setBuffer:affon_metal_borrow_buffer(index_handle) offset:0 atIndex:1];
    [enc setBuffer:affon_metal_borrow_buffer(out_handle) offset:0 atIndex:2];
    [enc setBytes:&vocab32 length:sizeof(vocab32) atIndex:3];
    [enc setBytes:&embDim32 length:sizeof(embDim32) atIndex:4];
    [enc setBytes:&indexCount32 length:sizeof(indexCount32) atIndex:5];
    [enc setBuffer:statusBuffer offset:0 atIndex:6];
    NSUInteger w = ctx.embeddingI64I64Pipeline.threadExecutionWidth;
    if (w == 0) w = 64;
    [enc dispatchThreads:MTLSizeMake(len32, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    [enc endEncoding];
    return affon_metal_finish_command_buffer_with_status(cb, statusBuffer, -5, -6);
    }
}


int affon_metal_matmul_offset_f32(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    size_t a_offset_bytes,
    size_t b_offset_bytes,
    size_t out_offset_bytes,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.matmulF32Pipeline || !a_handle || !b_handle || !out_handle) return -1;

    AffonMetalBuffer *a = (__bridge AffonMetalBuffer *)a_handle;
    AffonMetalBuffer *b = (__bridge AffonMetalBuffer *)b_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;

    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.matmulF32Pipeline];
    [encoder setBuffer:a.buffer offset:a_offset_bytes atIndex:0];
    [encoder setBuffer:b.buffer offset:b_offset_bytes atIndex:1];
    [encoder setBuffer:out.buffer offset:out_offset_bytes atIndex:2];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:3];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:4];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:5];

    NSUInteger w = ctx.matmulF32Pipeline.threadExecutionWidth;
    if (w == 0) w = 8;
    NSUInteger h = ctx.matmulF32Pipeline.maxTotalThreadsPerThreadgroup / w;
    if (h == 0) h = 1;
    if (h > 8) h = 8;

    MTLSize gridSize = MTLSizeMake(n, m, 1);
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_matmul_offset_i64(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    size_t a_offset_bytes,
    size_t b_offset_bytes,
    size_t out_offset_bytes,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.matmulI64Pipeline || !a_handle || !b_handle || !out_handle) return -1;

    id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
    id<MTLBuffer> b = affon_metal_borrow_buffer(b_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!a || !b || !out) return -1;

    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.matmulI64Pipeline];
    [encoder setBuffer:a offset:a_offset_bytes atIndex:0];
    [encoder setBuffer:b offset:b_offset_bytes atIndex:1];
    [encoder setBuffer:out offset:out_offset_bytes atIndex:2];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:3];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:4];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:5];

    NSUInteger w = ctx.matmulI64Pipeline.threadExecutionWidth;
    if (w == 0) w = 8;
    NSUInteger h = ctx.matmulI64Pipeline.maxTotalThreadsPerThreadgroup / w;
    if (h == 0) h = 1;
    if (h > 8) h = 8;

    MTLSize gridSize = MTLSizeMake(n, m, 1);
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}

int affon_metal_matmul_offset_many_i64(
    void *a_handle,
    void *b_handle,
    void *out_handle,
    const size_t *a_offsets_bytes,
    const size_t *b_offsets_bytes,
    const size_t *out_offsets_bytes,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.matmulI64Pipeline || !a_handle || !b_handle || !out_handle) return -1;
    if ((batch_count > 0) && (!a_offsets_bytes || !b_offsets_bytes || !out_offsets_bytes)) return -1;
    if (batch_count == 0) return 0;

    id<MTLBuffer> a = affon_metal_borrow_buffer(a_handle);
    id<MTLBuffer> b = affon_metal_borrow_buffer(b_handle);
    id<MTLBuffer> out = affon_metal_borrow_buffer(out_handle);
    if (!a || !b || !out) return -1;

    uint32_t m32 = (uint32_t)m;
    uint32_t n32 = (uint32_t)n;
    uint32_t k32 = (uint32_t)k;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.matmulI64Pipeline];
    [encoder setBytes:&m32 length:sizeof(m32) atIndex:3];
    [encoder setBytes:&n32 length:sizeof(n32) atIndex:4];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:5];

    NSUInteger w = ctx.matmulI64Pipeline.threadExecutionWidth;
    if (w == 0) w = 8;
    NSUInteger h = ctx.matmulI64Pipeline.maxTotalThreadsPerThreadgroup / w;
    if (h == 0) h = 1;
    if (h > 8) h = 8;

    MTLSize gridSize = MTLSizeMake(n, m, 1);
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    for (size_t i = 0; i < batch_count; i++) {
        [encoder setBuffer:a offset:a_offsets_bytes[i] atIndex:0];
        [encoder setBuffer:b offset:b_offsets_bytes[i] atIndex:1];
        [encoder setBuffer:out offset:out_offsets_bytes[i] atIndex:2];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
    [encoder endEncoding];

    return affon_metal_finish_command_buffer(commandBuffer, -4);
    }
}


int affon_metal_gather_f32(void *input_handle, void *index_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *input_strides, size_t axis, size_t axis_size, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.gatherF32Pipeline || !input_handle || !index_handle || !out_handle || !shape || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.gatherF32Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:index.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:5];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:6];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:7];
    [encoder setBuffer:statusBuffer offset:0 atIndex:8];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:9];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.gatherF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_gather_i64_f32(void *input_handle, void *index_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *input_strides, size_t axis, size_t axis_size, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.gatherI64F32Pipeline || !input_handle || !index_handle || !out_handle || !shape || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.gatherI64F32Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:index.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:5];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:6];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:7];
    [encoder setBuffer:statusBuffer offset:0 atIndex:8];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:9];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.gatherI64F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_gather_i64_i64(void *input_handle, void *index_handle, void *out_handle, size_t ndim, const uint32_t *shape, const uint32_t *input_strides, size_t axis, size_t axis_size, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.gatherI64I64Pipeline || !input_handle || !index_handle || !out_handle || !shape || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;
    uint32_t len32 = (uint32_t)len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.gatherI64I64Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:index.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:5];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:6];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:7];
    [encoder setBuffer:statusBuffer offset:0 atIndex:8];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:9];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.gatherI64I64Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_index_select_axis0_f32(void *input_handle, void *index_handle, void *out_handle, size_t rows, size_t width) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.indexSelectAxis0F32Pipeline || !input_handle || !index_handle || !out_handle) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t width32 = (uint32_t)width;
    uint32_t len32 = (uint32_t)(rows * width);
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.indexSelectAxis0F32Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:index.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:3];
    [encoder setBytes:&width32 length:sizeof(width32) atIndex:4];
    [encoder setBuffer:statusBuffer offset:0 atIndex:5];

    MTLSize gridSize = MTLSizeMake(len32, 1, 1);
    NSUInteger threadWidth = ctx.indexSelectAxis0F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len32 && len32 > 0) threadWidth = len32;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_index_select_axis0_i64_f32(void *input_handle, void *index_handle, void *out_handle, size_t rows, size_t width) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.indexSelectAxis0I64F32Pipeline || !input_handle || !index_handle || !out_handle) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t width32 = (uint32_t)width;
    uint32_t len32 = (uint32_t)(rows * width);
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.indexSelectAxis0I64F32Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:index.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:3];
    [encoder setBytes:&width32 length:sizeof(width32) atIndex:4];
    [encoder setBuffer:statusBuffer offset:0 atIndex:5];

    MTLSize gridSize = MTLSizeMake(len32, 1, 1);
    NSUInteger threadWidth = ctx.indexSelectAxis0I64F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len32 && len32 > 0) threadWidth = len32;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_index_select_i64_f32(void *input_handle, void *index_handle, void *out_handle, size_t ndim, const uint32_t *out_shape, const uint32_t *input_strides, size_t axis, size_t index_len, size_t axis_size, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.indexSelectI64F32Pipeline || !input_handle || !index_handle || !out_handle || !out_shape || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t indexLen32 = (uint32_t)index_len;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t len32 = (uint32_t)len;
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.indexSelectI64F32Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:index.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:out_shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:5];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:6];
    [encoder setBytes:&indexLen32 length:sizeof(indexLen32) atIndex:7];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:8];
    [encoder setBuffer:statusBuffer offset:0 atIndex:9];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:10];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.indexSelectI64F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_index_select_axis0_i64_i64(void *input_handle, void *index_handle, void *out_handle, size_t rows, size_t width) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.indexSelectAxis0I64I64Pipeline || !input_handle || !index_handle || !out_handle) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t width32 = (uint32_t)width;
    uint32_t len32 = (uint32_t)(rows * width);
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.indexSelectAxis0I64I64Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:index.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:3];
    [encoder setBytes:&width32 length:sizeof(width32) atIndex:4];
    [encoder setBuffer:statusBuffer offset:0 atIndex:5];

    MTLSize gridSize = MTLSizeMake(len32, 1, 1);
    NSUInteger threadWidth = ctx.indexSelectAxis0I64I64Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len32 && len32 > 0) threadWidth = len32;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_index_select_i64_i64(void *input_handle, void *index_handle, void *out_handle, size_t ndim, const uint32_t *out_shape, const uint32_t *input_strides, size_t axis, size_t index_len, size_t axis_size, size_t len) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.indexSelectI64I64Pipeline || !input_handle || !index_handle || !out_handle || !out_shape || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t indexLen32 = (uint32_t)index_len;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t len32 = (uint32_t)len;
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.indexSelectI64I64Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:index.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:out_shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:5];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:6];
    [encoder setBytes:&indexLen32 length:sizeof(indexLen32) atIndex:7];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:8];
    [encoder setBuffer:statusBuffer offset:0 atIndex:9];
    [encoder setBytes:&len32 length:sizeof(len32) atIndex:10];

    MTLSize gridSize = MTLSizeMake(len, 1, 1);
    NSUInteger threadWidth = ctx.indexSelectI64I64Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len && len > 0) threadWidth = len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_index_select_axis0_scatter_add_f32(void *index_handle, void *src_handle, void *out_handle, size_t rows, size_t width, size_t dst_rows) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.indexSelectAxis0ScatterAddF32Pipeline || !index_handle || !src_handle || !out_handle) return -1;

    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *src = (__bridge AffonMetalBuffer *)src_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t width32 = (uint32_t)width;
    uint32_t dstRows32 = (uint32_t)dst_rows;
    uint32_t len32 = (uint32_t)(rows * width);
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.indexSelectAxis0ScatterAddF32Pipeline];
    [encoder setBuffer:index.buffer offset:0 atIndex:0];
    [encoder setBuffer:src.buffer offset:0 atIndex:1];
    [encoder setBuffer:out.buffer offset:0 atIndex:2];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:3];
    [encoder setBytes:&width32 length:sizeof(width32) atIndex:4];
    [encoder setBytes:&dstRows32 length:sizeof(dstRows32) atIndex:5];
    [encoder setBuffer:statusBuffer offset:0 atIndex:6];

    MTLSize gridSize = MTLSizeMake(len32, 1, 1);
    NSUInteger threadWidth = ctx.indexSelectAxis0ScatterAddF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > len32 && len32 > 0) threadWidth = len32;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}

int affon_metal_topk_f32(void *input_handle, void *values_handle, void *indices_handle, size_t rows, size_t cols, size_t axis, size_t k) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.topkF32Pipeline || !input_handle || !values_handle || !indices_handle) return -1;
    if (rows == 0 || cols == 0 || axis > 1 || k == 0 || k > 64) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *values = (__bridge AffonMetalBuffer *)values_handle;
    AffonMetalBuffer *indices = (__bridge AffonMetalBuffer *)indices_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t cols32 = (uint32_t)cols;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t k32 = (uint32_t)k;
    size_t work_items = axis == 0 ? cols : rows;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.topkF32Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:values.buffer offset:0 atIndex:1];
    [encoder setBuffer:indices.buffer offset:0 atIndex:2];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:3];
    [encoder setBytes:&cols32 length:sizeof(cols32) atIndex:4];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:5];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:6];

    MTLSize gridSize = MTLSizeMake(work_items, 1, 1);
    NSUInteger threadWidth = ctx.topkF32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > work_items && work_items > 0) threadWidth = work_items;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_topk_i64_f32(void *input_handle, void *values_handle, void *indices_handle, size_t rows, size_t cols, size_t axis, size_t k) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.topkI64F32Pipeline || !input_handle || !values_handle || !indices_handle) return -1;
    if (rows == 0 || cols == 0 || axis > 1 || k == 0 || k > 64) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *values = (__bridge AffonMetalBuffer *)values_handle;
    AffonMetalBuffer *indices = (__bridge AffonMetalBuffer *)indices_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t cols32 = (uint32_t)cols;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t k32 = (uint32_t)k;
    size_t work_items = axis == 0 ? cols : rows;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.topkI64F32Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:values.buffer offset:0 atIndex:1];
    [encoder setBuffer:indices.buffer offset:0 atIndex:2];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:3];
    [encoder setBytes:&cols32 length:sizeof(cols32) atIndex:4];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:5];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:6];

    MTLSize gridSize = MTLSizeMake(work_items, 1, 1);
    NSUInteger threadWidth = ctx.topkI64F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > work_items && work_items > 0) threadWidth = work_items;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_topk_dir_i64_f32(void *input_handle, void *values_handle, void *indices_handle, size_t rows, size_t cols, size_t axis, size_t k, size_t largest) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.topkDirI64F32Pipeline || !input_handle || !values_handle || !indices_handle) return -1;
    if (rows == 0 || cols == 0 || axis > 1 || k == 0 || k > 64) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *values = (__bridge AffonMetalBuffer *)values_handle;
    AffonMetalBuffer *indices = (__bridge AffonMetalBuffer *)indices_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t cols32 = (uint32_t)cols;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t k32 = (uint32_t)k;
    uint32_t largest32 = largest != 0 ? 1u : 0u;
    size_t work_items = axis == 0 ? cols : rows;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.topkDirI64F32Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:values.buffer offset:0 atIndex:1];
    [encoder setBuffer:indices.buffer offset:0 atIndex:2];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:3];
    [encoder setBytes:&cols32 length:sizeof(cols32) atIndex:4];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:5];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:6];
    [encoder setBytes:&largest32 length:sizeof(largest32) atIndex:7];

    MTLSize gridSize = MTLSizeMake(work_items, 1, 1);
    NSUInteger threadWidth = ctx.topkDirI64F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > work_items && work_items > 0) threadWidth = work_items;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_topk_nd_i64_f32(
    void *input_handle,
    void *values_handle,
    void *indices_handle,
    size_t ndim,
    const uint32_t *out_shape,
    const uint32_t *out_strides,
    const uint32_t *input_strides,
    size_t axis,
    size_t axis_size,
    size_t k,
    size_t largest,
    size_t out_len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.topkNdI64F32Pipeline || !input_handle || !values_handle || !indices_handle || !out_shape || !out_strides || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim || axis_size == 0 || k == 0 || k > 64) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *values = (__bridge AffonMetalBuffer *)values_handle;
    AffonMetalBuffer *indices = (__bridge AffonMetalBuffer *)indices_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t k32 = (uint32_t)k;
    uint32_t largest32 = largest != 0 ? 1u : 0u;
    uint32_t outLen32 = (uint32_t)out_len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.topkNdI64F32Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:values.buffer offset:0 atIndex:1];
    [encoder setBuffer:indices.buffer offset:0 atIndex:2];
    [encoder setBytes:out_shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:out_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:6];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:7];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:8];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:9];
    [encoder setBytes:&largest32 length:sizeof(largest32) atIndex:10];
    [encoder setBytes:&outLen32 length:sizeof(outLen32) atIndex:11];

    MTLSize gridSize = MTLSizeMake(out_len, 1, 1);
    NSUInteger threadWidth = ctx.topkNdI64F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > out_len && out_len > 0) threadWidth = out_len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_topk_dir_i64_i64(void *input_handle, void *values_handle, void *indices_handle, size_t rows, size_t cols, size_t axis, size_t k, size_t largest) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.topkDirI64I64Pipeline || !input_handle || !values_handle || !indices_handle) return -1;
    if (rows == 0 || cols == 0 || axis > 1 || k == 0 || k > 64) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *values = (__bridge AffonMetalBuffer *)values_handle;
    AffonMetalBuffer *indices = (__bridge AffonMetalBuffer *)indices_handle;
    uint32_t rows32 = (uint32_t)rows;
    uint32_t cols32 = (uint32_t)cols;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t k32 = (uint32_t)k;
    uint32_t largest32 = largest != 0 ? 1u : 0u;
    size_t work_items = axis == 0 ? cols : rows;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.topkDirI64I64Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:values.buffer offset:0 atIndex:1];
    [encoder setBuffer:indices.buffer offset:0 atIndex:2];
    [encoder setBytes:&rows32 length:sizeof(rows32) atIndex:3];
    [encoder setBytes:&cols32 length:sizeof(cols32) atIndex:4];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:5];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:6];
    [encoder setBytes:&largest32 length:sizeof(largest32) atIndex:7];

    MTLSize gridSize = MTLSizeMake(work_items, 1, 1);
    NSUInteger threadWidth = ctx.topkDirI64I64Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > work_items && work_items > 0) threadWidth = work_items;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_topk_nd_i64_i64(
    void *input_handle,
    void *values_handle,
    void *indices_handle,
    size_t ndim,
    const uint32_t *out_shape,
    const uint32_t *out_strides,
    const uint32_t *input_strides,
    size_t axis,
    size_t axis_size,
    size_t k,
    size_t largest,
    size_t out_len
) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.topkNdI64I64Pipeline || !input_handle || !values_handle || !indices_handle || !out_shape || !out_strides || !input_strides) return -1;
    if (ndim == 0 || ndim > affon_max_broadcast_dims || axis >= ndim || axis_size == 0 || k == 0 || k > 64) return -1;

    AffonMetalBuffer *input = (__bridge AffonMetalBuffer *)input_handle;
    AffonMetalBuffer *values = (__bridge AffonMetalBuffer *)values_handle;
    AffonMetalBuffer *indices = (__bridge AffonMetalBuffer *)indices_handle;
    uint32_t ndim32 = (uint32_t)ndim;
    uint32_t axis32 = (uint32_t)axis;
    uint32_t axisSize32 = (uint32_t)axis_size;
    uint32_t k32 = (uint32_t)k;
    uint32_t largest32 = largest != 0 ? 1u : 0u;
    uint32_t outLen32 = (uint32_t)out_len;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -2;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -3;

    [encoder setComputePipelineState:ctx.topkNdI64I64Pipeline];
    [encoder setBuffer:input.buffer offset:0 atIndex:0];
    [encoder setBuffer:values.buffer offset:0 atIndex:1];
    [encoder setBuffer:indices.buffer offset:0 atIndex:2];
    [encoder setBytes:out_shape length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:out_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:input_strides length:affon_max_broadcast_dims * sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&ndim32 length:sizeof(ndim32) atIndex:6];
    [encoder setBytes:&axis32 length:sizeof(axis32) atIndex:7];
    [encoder setBytes:&axisSize32 length:sizeof(axisSize32) atIndex:8];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:9];
    [encoder setBytes:&largest32 length:sizeof(largest32) atIndex:10];
    [encoder setBytes:&outLen32 length:sizeof(outLen32) atIndex:11];

    MTLSize gridSize = MTLSizeMake(out_len, 1, 1);
    NSUInteger threadWidth = ctx.topkNdI64I64Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > out_len && out_len > 0) threadWidth = out_len;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error) return -4;
    return 0;
    }
}

int affon_metal_one_hot_i64_f32(void *index_handle, void *out_handle, size_t num_indices, size_t num_classes) {
    @autoreleasepool {
    AffonMetalContext *ctx = affon_get_context();
    if (!ctx || !ctx.oneHotI64F32Pipeline || !index_handle || !out_handle || num_classes == 0) return -1;

    AffonMetalBuffer *index = (__bridge AffonMetalBuffer *)index_handle;
    AffonMetalBuffer *out = (__bridge AffonMetalBuffer *)out_handle;
    uint32_t numIndices32 = (uint32_t)num_indices;
    uint32_t numClasses32 = (uint32_t)num_classes;
    uint32_t outLen32 = (uint32_t)(num_indices * num_classes);
    id<MTLBuffer> statusBuffer = affon_metal_create_status_buffer(ctx);
    if (!statusBuffer) return -2;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (!commandBuffer) return -3;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) return -4;

    [encoder setComputePipelineState:ctx.oneHotI64F32Pipeline];
    [encoder setBuffer:index.buffer offset:0 atIndex:0];
    [encoder setBuffer:out.buffer offset:0 atIndex:1];
    [encoder setBytes:&numIndices32 length:sizeof(numIndices32) atIndex:2];
    [encoder setBytes:&numClasses32 length:sizeof(numClasses32) atIndex:3];
    [encoder setBuffer:statusBuffer offset:0 atIndex:4];

    MTLSize gridSize = MTLSizeMake(outLen32, 1, 1);
    NSUInteger threadWidth = ctx.oneHotI64F32Pipeline.maxTotalThreadsPerThreadgroup;
    if (threadWidth > outLen32 && outLen32 > 0) threadWidth = outLen32;
    if (threadWidth == 0) threadWidth = 1;
    MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    return affon_metal_finish_command_buffer_with_status(commandBuffer, statusBuffer, -5, -6);
    }
}
