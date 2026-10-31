# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `a9066d6e6b6e0237e2a11aee4da783ef35d28e370c459f52cc435100eb91e52f`  
Binary SHA-256: `af1b758d59cd4f42a812cf3dcaae67d3b0f01abca19878e39bc7300e1ae67e35`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 192
- Numerical capability passed: 192
- Operator parity and timing passed: 192
- Program parity and timing passed: 192
- Probe failures or incorrect: 0
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/eq/strided/r3_c5` | correct | [passed](operator_peer/metal__f32__eq__strided__r3_c5.json) | 0.2284 / 0.2202 | [passed](program_peer/metal__f32__eq__strided__r3_c5.json) | 0.2536 / 0.2413 |
| `metal/f32/eq/strided/r3_c32` | correct | [passed](operator_peer/metal__f32__eq__strided__r3_c32.json) | 0.2404 / 0.2206 | [passed](program_peer/metal__f32__eq__strided__r3_c32.json) | 0.2396 / 0.2253 |
| `metal/f32/eq/strided/r3_c65` | correct | [passed](operator_peer/metal__f32__eq__strided__r3_c65.json) | 0.2372 / 0.2173 | [passed](program_peer/metal__f32__eq__strided__r3_c65.json) | 0.2584 / 0.2472 |
| `metal/f32/eq/strided/r3_c129` | correct | [passed](operator_peer/metal__f32__eq__strided__r3_c129.json) | 0.244 / 0.2187 | [passed](program_peer/metal__f32__eq__strided__r3_c129.json) | 0.2312 / 0.2268 |
| `metal/f32/eq/strided/r16_c5` | correct | [passed](operator_peer/metal__f32__eq__strided__r16_c5.json) | 0.2482 / 0.2278 | [passed](program_peer/metal__f32__eq__strided__r16_c5.json) | 0.2164 / 0.2428 |
| `metal/f32/eq/strided/r16_c32` | correct | [passed](operator_peer/metal__f32__eq__strided__r16_c32.json) | 0.2502 / 0.2337 | [passed](program_peer/metal__f32__eq__strided__r16_c32.json) | 0.2554 / 0.2382 |
| `metal/f32/eq/strided/r16_c65` | correct | [passed](operator_peer/metal__f32__eq__strided__r16_c65.json) | 0.2682 / 0.2308 | [passed](program_peer/metal__f32__eq__strided__r16_c65.json) | 0.2958 / 0.2351 |
| `metal/f32/eq/strided/r16_c129` | correct | [passed](operator_peer/metal__f32__eq__strided__r16_c129.json) | 0.2804 / 0.2408 | [passed](program_peer/metal__f32__eq__strided__r16_c129.json) | 0.2886 / 0.2396 |
| `metal/f32/eq/strided/r33_c5` | correct | [passed](operator_peer/metal__f32__eq__strided__r33_c5.json) | 0.2506 / 0.2225 | [passed](program_peer/metal__f32__eq__strided__r33_c5.json) | 0.2454 / 0.2198 |
| `metal/f32/eq/strided/r33_c32` | correct | [passed](operator_peer/metal__f32__eq__strided__r33_c32.json) | 0.2758 / 0.2275 | [passed](program_peer/metal__f32__eq__strided__r33_c32.json) | 0.3104 / 0.2329 |
| `metal/f32/eq/strided/r33_c65` | correct | [passed](operator_peer/metal__f32__eq__strided__r33_c65.json) | 0.2916 / 0.2372 | [passed](program_peer/metal__f32__eq__strided__r33_c65.json) | 0.3324 / 0.2429 |
| `metal/f32/eq/strided/r33_c129` | correct | [passed](operator_peer/metal__f32__eq__strided__r33_c129.json) | 0.302 / 0.2486 | [passed](program_peer/metal__f32__eq__strided__r33_c129.json) | 0.3386 / 0.2524 |
| `metal/f32/eq/strided/r128_c5` | correct | [passed](operator_peer/metal__f32__eq__strided__r128_c5.json) | 0.2794 / 0.2297 | [passed](program_peer/metal__f32__eq__strided__r128_c5.json) | 0.3102 / 0.2336 |
| `metal/f32/eq/strided/r128_c32` | correct | [passed](operator_peer/metal__f32__eq__strided__r128_c32.json) | 0.3416 / 0.2457 | [passed](program_peer/metal__f32__eq__strided__r128_c32.json) | 0.3754 / 0.2518 |
| `metal/f32/eq/strided/r128_c65` | correct | [passed](operator_peer/metal__f32__eq__strided__r128_c65.json) | 0.3816 / 0.2652 | [passed](program_peer/metal__f32__eq__strided__r128_c65.json) | 0.4202 / 0.3194 |
| `metal/f32/eq/strided/r128_c129` | correct | [passed](operator_peer/metal__f32__eq__strided__r128_c129.json) | 0.454 / 0.3466 | [passed](program_peer/metal__f32__eq__strided__r128_c129.json) | 0.4952 / 0.358 |
| `metal/f32/eq/offset/r3_c5` | correct | [passed](operator_peer/metal__f32__eq__offset__r3_c5.json) | 0.2438 / 0.2296 | [passed](program_peer/metal__f32__eq__offset__r3_c5.json) | 0.2586 / 0.2462 |
| `metal/f32/eq/offset/r3_c32` | correct | [passed](operator_peer/metal__f32__eq__offset__r3_c32.json) | 0.2374 / 0.2226 | [passed](program_peer/metal__f32__eq__offset__r3_c32.json) | 0.2582 / 0.2439 |
| `metal/f32/eq/offset/r3_c65` | correct | [passed](operator_peer/metal__f32__eq__offset__r3_c65.json) | 0.2402 / 0.2216 | [passed](program_peer/metal__f32__eq__offset__r3_c65.json) | 0.2612 / 0.2337 |
| `metal/f32/eq/offset/r3_c129` | correct | [passed](operator_peer/metal__f32__eq__offset__r3_c129.json) | 0.2466 / 0.2229 | [passed](program_peer/metal__f32__eq__offset__r3_c129.json) | 0.262 / 0.2214 |
| `metal/f32/eq/offset/r16_c5` | correct | [passed](operator_peer/metal__f32__eq__offset__r16_c5.json) | 0.2494 / 0.2239 | [passed](program_peer/metal__f32__eq__offset__r16_c5.json) | 0.2624 / 0.219 |
| `metal/f32/eq/offset/r16_c32` | correct | [passed](operator_peer/metal__f32__eq__offset__r16_c32.json) | 0.2528 / 0.2264 | [passed](program_peer/metal__f32__eq__offset__r16_c32.json) | 0.2726 / 0.231 |
| `metal/f32/eq/offset/r16_c65` | correct | [passed](operator_peer/metal__f32__eq__offset__r16_c65.json) | 0.2534 / 0.2283 | [passed](program_peer/metal__f32__eq__offset__r16_c65.json) | 0.314 / 0.2356 |
| `metal/f32/eq/offset/r16_c129` | correct | [passed](operator_peer/metal__f32__eq__offset__r16_c129.json) | 0.2782 / 0.241 | [passed](program_peer/metal__f32__eq__offset__r16_c129.json) | 0.3236 / 0.2408 |
| `metal/f32/eq/offset/r33_c5` | correct | [passed](operator_peer/metal__f32__eq__offset__r33_c5.json) | 0.2508 / 0.2239 | [passed](program_peer/metal__f32__eq__offset__r33_c5.json) | 0.2688 / 0.2301 |
| `metal/f32/eq/offset/r33_c32` | correct | [passed](operator_peer/metal__f32__eq__offset__r33_c32.json) | 0.2784 / 0.2304 | [passed](program_peer/metal__f32__eq__offset__r33_c32.json) | 0.3158 / 0.2366 |
| `metal/f32/eq/offset/r33_c65` | correct | [passed](operator_peer/metal__f32__eq__offset__r33_c65.json) | 0.2928 / 0.2402 | [passed](program_peer/metal__f32__eq__offset__r33_c65.json) | 0.325 / 0.2245 |
| `metal/f32/eq/offset/r33_c129` | correct | [passed](operator_peer/metal__f32__eq__offset__r33_c129.json) | 0.307 / 0.2634 | [passed](program_peer/metal__f32__eq__offset__r33_c129.json) | 0.3436 / 0.2833 |
| `metal/f32/eq/offset/r128_c5` | correct | [passed](operator_peer/metal__f32__eq__offset__r128_c5.json) | 0.2918 / 0.2462 | [passed](program_peer/metal__f32__eq__offset__r128_c5.json) | 0.2746 / 0.2317 |
| `metal/f32/eq/offset/r128_c32` | correct | [passed](operator_peer/metal__f32__eq__offset__r128_c32.json) | 0.3162 / 0.2465 | [passed](program_peer/metal__f32__eq__offset__r128_c32.json) | 0.374 / 0.2479 |
| `metal/f32/eq/offset/r128_c65` | correct | [passed](operator_peer/metal__f32__eq__offset__r128_c65.json) | 0.3624 / 0.2678 | [passed](program_peer/metal__f32__eq__offset__r128_c65.json) | 0.4126 / 0.3154 |
| `metal/f32/eq/offset/r128_c129` | correct | [passed](operator_peer/metal__f32__eq__offset__r128_c129.json) | 0.453 / 0.3412 | [passed](program_peer/metal__f32__eq__offset__r128_c129.json) | 0.5154 / 0.3554 |
| `metal/f32/lt/strided/r3_c5` | correct | [passed](operator_peer/metal__f32__lt__strided__r3_c5.json) | 0.242 / 0.2492 | [passed](program_peer/metal__f32__lt__strided__r3_c5.json) | 0.2526 / 0.2212 |
| `metal/f32/lt/strided/r3_c32` | correct | [passed](operator_peer/metal__f32__lt__strided__r3_c32.json) | 0.241 / 0.2218 | [passed](program_peer/metal__f32__lt__strided__r3_c32.json) | 0.2584 / 0.2272 |
| `metal/f32/lt/strided/r3_c65` | correct | [passed](operator_peer/metal__f32__lt__strided__r3_c65.json) | 0.2424 / 0.2217 | [passed](program_peer/metal__f32__lt__strided__r3_c65.json) | 0.2594 / 0.2215 |
| `metal/f32/lt/strided/r3_c129` | correct | [passed](operator_peer/metal__f32__lt__strided__r3_c129.json) | 0.2514 / 0.2252 | [passed](program_peer/metal__f32__lt__strided__r3_c129.json) | 0.26 / 0.2329 |
| `metal/f32/lt/strided/r16_c5` | correct | [passed](operator_peer/metal__f32__lt__strided__r16_c5.json) | 0.2464 / 0.2228 | [passed](program_peer/metal__f32__lt__strided__r16_c5.json) | 0.2602 / 0.2276 |
| `metal/f32/lt/strided/r16_c32` | correct | [passed](operator_peer/metal__f32__lt__strided__r16_c32.json) | 0.2102 / 0.2224 | [passed](program_peer/metal__f32__lt__strided__r16_c32.json) | 0.2712 / 0.2352 |
| `metal/f32/lt/strided/r16_c65` | correct | [passed](operator_peer/metal__f32__lt__strided__r16_c65.json) | 0.2358 / 0.2329 | [passed](program_peer/metal__f32__lt__strided__r16_c65.json) | 0.3156 / 0.2345 |
| `metal/f32/lt/strided/r16_c129` | correct | [passed](operator_peer/metal__f32__lt__strided__r16_c129.json) | 0.2776 / 0.2375 | [passed](program_peer/metal__f32__lt__strided__r16_c129.json) | 0.322 / 0.2411 |
| `metal/f32/lt/strided/r33_c5` | correct | [passed](operator_peer/metal__f32__lt__strided__r33_c5.json) | 0.2462 / 0.2258 | [passed](program_peer/metal__f32__lt__strided__r33_c5.json) | 0.2678 / 0.2327 |
| `metal/f32/lt/strided/r33_c32` | correct | [passed](operator_peer/metal__f32__lt__strided__r33_c32.json) | 0.2762 / 0.2582 | [passed](program_peer/metal__f32__lt__strided__r33_c32.json) | 0.3094 / 0.2348 |
| `metal/f32/lt/strided/r33_c65` | correct | [passed](operator_peer/metal__f32__lt__strided__r33_c65.json) | 0.2856 / 0.2391 | [passed](program_peer/metal__f32__lt__strided__r33_c65.json) | 0.3324 / 0.2391 |
| `metal/f32/lt/strided/r33_c129` | correct | [passed](operator_peer/metal__f32__lt__strided__r33_c129.json) | 0.2756 / 0.2454 | [passed](program_peer/metal__f32__lt__strided__r33_c129.json) | 0.3388 / 0.249 |
| `metal/f32/lt/strided/r128_c5` | correct | [passed](operator_peer/metal__f32__lt__strided__r128_c5.json) | 0.291 / 0.2353 | [passed](program_peer/metal__f32__lt__strided__r128_c5.json) | 0.3106 / 0.2318 |
| `metal/f32/lt/strided/r128_c32` | correct | [passed](operator_peer/metal__f32__lt__strided__r128_c32.json) | 0.311 / 0.2454 | [passed](program_peer/metal__f32__lt__strided__r128_c32.json) | 0.3748 / 0.2506 |
| `metal/f32/lt/strided/r128_c65` | correct | [passed](operator_peer/metal__f32__lt__strided__r128_c65.json) | 0.3798 / 0.2996 | [passed](program_peer/metal__f32__lt__strided__r128_c65.json) | 0.402 / 0.326 |
| `metal/f32/lt/strided/r128_c129` | correct | [passed](operator_peer/metal__f32__lt__strided__r128_c129.json) | 0.4576 / 0.3493 | [passed](program_peer/metal__f32__lt__strided__r128_c129.json) | 0.4884 / 0.3436 |
| `metal/f32/lt/offset/r3_c5` | correct | [passed](operator_peer/metal__f32__lt__offset__r3_c5.json) | 0.2446 / 0.2274 | [passed](program_peer/metal__f32__lt__offset__r3_c5.json) | 0.257 / 0.2378 |
| `metal/f32/lt/offset/r3_c32` | correct | [passed](operator_peer/metal__f32__lt__offset__r3_c32.json) | 0.2438 / 0.2454 | [passed](program_peer/metal__f32__lt__offset__r3_c32.json) | 0.2586 / 0.214 |
| `metal/f32/lt/offset/r3_c65` | correct | [passed](operator_peer/metal__f32__lt__offset__r3_c65.json) | 0.2454 / 0.2298 | [passed](program_peer/metal__f32__lt__offset__r3_c65.json) | 0.2548 / 0.2211 |
| `metal/f32/lt/offset/r3_c129` | correct | [passed](operator_peer/metal__f32__lt__offset__r3_c129.json) | 0.2392 / 0.2288 | [passed](program_peer/metal__f32__lt__offset__r3_c129.json) | 0.2566 / 0.2165 |
| `metal/f32/lt/offset/r16_c5` | correct | [passed](operator_peer/metal__f32__lt__offset__r16_c5.json) | 0.2052 / 0.2275 | [passed](program_peer/metal__f32__lt__offset__r16_c5.json) | 0.2296 / 0.2145 |
| `metal/f32/lt/offset/r16_c32` | correct | [passed](operator_peer/metal__f32__lt__offset__r16_c32.json) | 0.2484 / 0.2319 | [passed](program_peer/metal__f32__lt__offset__r16_c32.json) | 0.265 / 0.2263 |
| `metal/f32/lt/offset/r16_c65` | correct | [passed](operator_peer/metal__f32__lt__offset__r16_c65.json) | 0.2692 / 0.2298 | [passed](program_peer/metal__f32__lt__offset__r16_c65.json) | 0.3044 / 0.2303 |
| `metal/f32/lt/offset/r16_c129` | correct | [passed](operator_peer/metal__f32__lt__offset__r16_c129.json) | 0.278 / 0.2381 | [passed](program_peer/metal__f32__lt__offset__r16_c129.json) | 0.325 / 0.2425 |
| `metal/f32/lt/offset/r33_c5` | correct | [passed](operator_peer/metal__f32__lt__offset__r33_c5.json) | 0.252 / 0.2218 | [passed](program_peer/metal__f32__lt__offset__r33_c5.json) | 0.2656 / 0.2282 |
| `metal/f32/lt/offset/r33_c32` | correct | [passed](operator_peer/metal__f32__lt__offset__r33_c32.json) | 0.2782 / 0.2484 | [passed](program_peer/metal__f32__lt__offset__r33_c32.json) | 0.319 / 0.2212 |
| `metal/f32/lt/offset/r33_c65` | correct | [passed](operator_peer/metal__f32__lt__offset__r33_c65.json) | 0.291 / 0.2354 | [passed](program_peer/metal__f32__lt__offset__r33_c65.json) | 0.2952 / 0.261 |
| `metal/f32/lt/offset/r33_c129` | correct | [passed](operator_peer/metal__f32__lt__offset__r33_c129.json) | 0.2988 / 0.2801 | [passed](program_peer/metal__f32__lt__offset__r33_c129.json) | 0.331 / 0.2409 |
| `metal/f32/lt/offset/r128_c5` | correct | [passed](operator_peer/metal__f32__lt__offset__r128_c5.json) | 0.2932 / 0.2362 | [passed](program_peer/metal__f32__lt__offset__r128_c5.json) | 0.3102 / 0.2287 |
| `metal/f32/lt/offset/r128_c32` | correct | [passed](operator_peer/metal__f32__lt__offset__r128_c32.json) | 0.3424 / 0.2443 | [passed](program_peer/metal__f32__lt__offset__r128_c32.json) | 0.374 / 0.2452 |
| `metal/f32/lt/offset/r128_c65` | correct | [passed](operator_peer/metal__f32__lt__offset__r128_c65.json) | 0.3828 / 0.2847 | [passed](program_peer/metal__f32__lt__offset__r128_c65.json) | 0.3944 / 0.3247 |
| `metal/f32/lt/offset/r128_c129` | correct | [passed](operator_peer/metal__f32__lt__offset__r128_c129.json) | 0.451 / 0.3403 | [passed](program_peer/metal__f32__lt__offset__r128_c129.json) | 0.5114 / 0.3535 |
| `metal/f32/gt/strided/r3_c5` | correct | [passed](operator_peer/metal__f32__gt__strided__r3_c5.json) | 0.2412 / 0.2605 | [passed](program_peer/metal__f32__gt__strided__r3_c5.json) | 0.2434 / 0.2279 |
| `metal/f32/gt/strided/r3_c32` | correct | [passed](operator_peer/metal__f32__gt__strided__r3_c32.json) | 0.1906 / 0.2161 | [passed](program_peer/metal__f32__gt__strided__r3_c32.json) | 0.2322 / 0.2147 |
| `metal/f32/gt/strided/r3_c65` | correct | [passed](operator_peer/metal__f32__gt__strided__r3_c65.json) | 0.241 / 0.2458 | [passed](program_peer/metal__f32__gt__strided__r3_c65.json) | 0.1922 / 0.2176 |
| `metal/f32/gt/strided/r3_c129` | correct | [passed](operator_peer/metal__f32__gt__strided__r3_c129.json) | 0.2586 / 0.2319 | [passed](program_peer/metal__f32__gt__strided__r3_c129.json) | 0.2546 / 0.2293 |
| `metal/f32/gt/strided/r16_c5` | correct | [passed](operator_peer/metal__f32__gt__strided__r16_c5.json) | 0.2496 / 0.2268 | [passed](program_peer/metal__f32__gt__strided__r16_c5.json) | 0.2456 / 0.2381 |
| `metal/f32/gt/strided/r16_c32` | correct | [passed](operator_peer/metal__f32__gt__strided__r16_c32.json) | 0.2586 / 0.2216 | [passed](program_peer/metal__f32__gt__strided__r16_c32.json) | 0.2342 / 0.226 |
| `metal/f32/gt/strided/r16_c65` | correct | [passed](operator_peer/metal__f32__gt__strided__r16_c65.json) | 0.2674 / 0.2352 | [passed](program_peer/metal__f32__gt__strided__r16_c65.json) | 0.2856 / 0.2382 |
| `metal/f32/gt/strided/r16_c129` | correct | [passed](operator_peer/metal__f32__gt__strided__r16_c129.json) | 0.2832 / 0.241 | [passed](program_peer/metal__f32__gt__strided__r16_c129.json) | 0.3088 / 0.2357 |
| `metal/f32/gt/strided/r33_c5` | correct | [passed](operator_peer/metal__f32__gt__strided__r33_c5.json) | 0.2558 / 0.2236 | [passed](program_peer/metal__f32__gt__strided__r33_c5.json) | 0.2678 / 0.2197 |
| `metal/f32/gt/strided/r33_c32` | correct | [passed](operator_peer/metal__f32__gt__strided__r33_c32.json) | 0.2726 / 0.2316 | [passed](program_peer/metal__f32__gt__strided__r33_c32.json) | 0.3192 / 0.2355 |
| `metal/f32/gt/strided/r33_c65` | correct | [passed](operator_peer/metal__f32__gt__strided__r33_c65.json) | 0.283 / 0.2499 | [passed](program_peer/metal__f32__gt__strided__r33_c65.json) | 0.328 / 0.2437 |
| `metal/f32/gt/strided/r33_c129` | correct | [passed](operator_peer/metal__f32__gt__strided__r33_c129.json) | 0.3062 / 0.2433 | [passed](program_peer/metal__f32__gt__strided__r33_c129.json) | 0.3418 / 0.2521 |
| `metal/f32/gt/strided/r128_c5` | correct | [passed](operator_peer/metal__f32__gt__strided__r128_c5.json) | 0.2916 / 0.2352 | [passed](program_peer/metal__f32__gt__strided__r128_c5.json) | 0.3128 / 0.2338 |
| `metal/f32/gt/strided/r128_c32` | correct | [passed](operator_peer/metal__f32__gt__strided__r128_c32.json) | 0.313 / 0.2551 | [passed](program_peer/metal__f32__gt__strided__r128_c32.json) | 0.3732 / 0.2491 |
| `metal/f32/gt/strided/r128_c65` | correct | [passed](operator_peer/metal__f32__gt__strided__r128_c65.json) | 0.3814 / 0.3023 | [passed](program_peer/metal__f32__gt__strided__r128_c65.json) | 0.4228 / 0.295 |
| `metal/f32/gt/strided/r128_c129` | correct | [passed](operator_peer/metal__f32__gt__strided__r128_c129.json) | 0.4532 / 0.3463 | [passed](program_peer/metal__f32__gt__strided__r128_c129.json) | 0.5082 / 0.3604 |
| `metal/f32/gt/offset/r3_c5` | correct | [passed](operator_peer/metal__f32__gt__offset__r3_c5.json) | 0.2428 / 0.2196 | [passed](program_peer/metal__f32__gt__offset__r3_c5.json) | 0.2564 / 0.2234 |
| `metal/f32/gt/offset/r3_c32` | correct | [passed](operator_peer/metal__f32__gt__offset__r3_c32.json) | 0.2172 / 0.2395 | [passed](program_peer/metal__f32__gt__offset__r3_c32.json) | 0.257 / 0.2265 |
| `metal/f32/gt/offset/r3_c65` | correct | [passed](operator_peer/metal__f32__gt__offset__r3_c65.json) | 0.2124 / 0.2285 | [passed](program_peer/metal__f32__gt__offset__r3_c65.json) | 0.2576 / 0.232 |
| `metal/f32/gt/offset/r3_c129` | correct | [passed](operator_peer/metal__f32__gt__offset__r3_c129.json) | 0.248 / 0.2232 | [passed](program_peer/metal__f32__gt__offset__r3_c129.json) | 0.2592 / 0.2304 |
| `metal/f32/gt/offset/r16_c5` | correct | [passed](operator_peer/metal__f32__gt__offset__r16_c5.json) | 0.2478 / 0.2171 | [passed](program_peer/metal__f32__gt__offset__r16_c5.json) | 0.2632 / 0.2609 |
| `metal/f32/gt/offset/r16_c32` | correct | [passed](operator_peer/metal__f32__gt__offset__r16_c32.json) | 0.2538 / 0.2306 | [passed](program_peer/metal__f32__gt__offset__r16_c32.json) | 0.27 / 0.2291 |
| `metal/f32/gt/offset/r16_c65` | correct | [passed](operator_peer/metal__f32__gt__offset__r16_c65.json) | 0.2484 / 0.2301 | [passed](program_peer/metal__f32__gt__offset__r16_c65.json) | 0.3072 / 0.2349 |
| `metal/f32/gt/offset/r16_c129` | correct | [passed](operator_peer/metal__f32__gt__offset__r16_c129.json) | 0.2564 / 0.223 | [passed](program_peer/metal__f32__gt__offset__r16_c129.json) | 0.3094 / 0.2398 |
| `metal/f32/gt/offset/r33_c5` | correct | [passed](operator_peer/metal__f32__gt__offset__r33_c5.json) | 0.2512 / 0.2202 | [passed](program_peer/metal__f32__gt__offset__r33_c5.json) | 0.2496 / 0.2323 |
| `metal/f32/gt/offset/r33_c32` | correct | [passed](operator_peer/metal__f32__gt__offset__r33_c32.json) | 0.275 / 0.2336 | [passed](program_peer/metal__f32__gt__offset__r33_c32.json) | 0.3086 / 0.2515 |
| `metal/f32/gt/offset/r33_c65` | correct | [passed](operator_peer/metal__f32__gt__offset__r33_c65.json) | 0.2902 / 0.2396 | [passed](program_peer/metal__f32__gt__offset__r33_c65.json) | 0.333 / 0.2374 |
| `metal/f32/gt/offset/r33_c129` | correct | [passed](operator_peer/metal__f32__gt__offset__r33_c129.json) | 0.3034 / 0.2467 | [passed](program_peer/metal__f32__gt__offset__r33_c129.json) | 0.3396 / 0.2487 |
| `metal/f32/gt/offset/r128_c5` | correct | [passed](operator_peer/metal__f32__gt__offset__r128_c5.json) | 0.2926 / 0.232 | [passed](program_peer/metal__f32__gt__offset__r128_c5.json) | 0.3106 / 0.2302 |
| `metal/f32/gt/offset/r128_c32` | correct | [passed](operator_peer/metal__f32__gt__offset__r128_c32.json) | 0.3424 / 0.2447 | [passed](program_peer/metal__f32__gt__offset__r128_c32.json) | 0.3712 / 0.2488 |
| `metal/f32/gt/offset/r128_c65` | correct | [passed](operator_peer/metal__f32__gt__offset__r128_c65.json) | 0.3866 / 0.2994 | [passed](program_peer/metal__f32__gt__offset__r128_c65.json) | 0.4172 / 0.2858 |
| `metal/f32/gt/offset/r128_c129` | correct | [passed](operator_peer/metal__f32__gt__offset__r128_c129.json) | 0.4454 / 0.3323 | [passed](program_peer/metal__f32__gt__offset__r128_c129.json) | 0.5136 / 0.3554 |
| `metal/bf16/eq/strided/r3_c5` | correct | [passed](operator_peer/metal__bf16__eq__strided__r3_c5.json) | 0.2478 / 0.2281 | [passed](program_peer/metal__bf16__eq__strided__r3_c5.json) | 0.2572 / 0.2284 |
| `metal/bf16/eq/strided/r3_c32` | correct | [passed](operator_peer/metal__bf16__eq__strided__r3_c32.json) | 0.2448 / 0.227 | [passed](program_peer/metal__bf16__eq__strided__r3_c32.json) | 0.2554 / 0.2533 |
| `metal/bf16/eq/strided/r3_c65` | correct | [passed](operator_peer/metal__bf16__eq__strided__r3_c65.json) | 0.2516 / 0.2311 | [passed](program_peer/metal__bf16__eq__strided__r3_c65.json) | 0.2596 / 0.2405 |
| `metal/bf16/eq/strided/r3_c129` | correct | [passed](operator_peer/metal__bf16__eq__strided__r3_c129.json) | 0.2498 / 0.2261 | [passed](program_peer/metal__bf16__eq__strided__r3_c129.json) | 0.263 / 0.2347 |
| `metal/bf16/eq/strided/r16_c5` | correct | [passed](operator_peer/metal__bf16__eq__strided__r16_c5.json) | 0.2544 / 0.2193 | [passed](program_peer/metal__bf16__eq__strided__r16_c5.json) | 0.2716 / 0.2243 |
| `metal/bf16/eq/strided/r16_c32` | correct | [passed](operator_peer/metal__bf16__eq__strided__r16_c32.json) | 0.2378 / 0.2291 | [passed](program_peer/metal__bf16__eq__strided__r16_c32.json) | 0.2736 / 0.2173 |
| `metal/bf16/eq/strided/r16_c65` | correct | [passed](operator_peer/metal__bf16__eq__strided__r16_c65.json) | 0.2658 / 0.2328 | [passed](program_peer/metal__bf16__eq__strided__r16_c65.json) | 0.3026 / 0.257 |
| `metal/bf16/eq/strided/r16_c129` | correct | [passed](operator_peer/metal__bf16__eq__strided__r16_c129.json) | 0.2828 / 0.2372 | [passed](program_peer/metal__bf16__eq__strided__r16_c129.json) | 0.3034 / 0.2451 |
| `metal/bf16/eq/strided/r33_c5` | correct | [passed](operator_peer/metal__bf16__eq__strided__r33_c5.json) | 0.2504 / 0.2147 | [passed](program_peer/metal__bf16__eq__strided__r33_c5.json) | 0.267 / 0.216 |
| `metal/bf16/eq/strided/r33_c32` | correct | [passed](operator_peer/metal__bf16__eq__strided__r33_c32.json) | 0.2656 / 0.2281 | [passed](program_peer/metal__bf16__eq__strided__r33_c32.json) | 0.3146 / 0.2354 |
| `metal/bf16/eq/strided/r33_c65` | correct | [passed](operator_peer/metal__bf16__eq__strided__r33_c65.json) | 0.3272 / 0.2286 | [passed](program_peer/metal__bf16__eq__strided__r33_c65.json) | 0.3346 / 0.2401 |
| `metal/bf16/eq/strided/r33_c129` | correct | [passed](operator_peer/metal__bf16__eq__strided__r33_c129.json) | 0.3046 / 0.2538 | [passed](program_peer/metal__bf16__eq__strided__r33_c129.json) | 0.3432 / 0.249 |
| `metal/bf16/eq/strided/r128_c5` | correct | [passed](operator_peer/metal__bf16__eq__strided__r128_c5.json) | 0.3004 / 0.2251 | [passed](program_peer/metal__bf16__eq__strided__r128_c5.json) | 0.3304 / 0.2326 |
| `metal/bf16/eq/strided/r128_c32` | correct | [passed](operator_peer/metal__bf16__eq__strided__r128_c32.json) | 0.3258 / 0.2467 | [passed](program_peer/metal__bf16__eq__strided__r128_c32.json) | 0.3726 / 0.2474 |
| `metal/bf16/eq/strided/r128_c65` | correct | [passed](operator_peer/metal__bf16__eq__strided__r128_c65.json) | 0.3792 / 0.2768 | [passed](program_peer/metal__bf16__eq__strided__r128_c65.json) | 0.421 / 0.3235 |
| `metal/bf16/eq/strided/r128_c129` | correct | [passed](operator_peer/metal__bf16__eq__strided__r128_c129.json) | 0.4514 / 0.3294 | [passed](program_peer/metal__bf16__eq__strided__r128_c129.json) | 0.5172 / 0.3547 |
| `metal/bf16/eq/offset/r3_c5` | correct | [passed](operator_peer/metal__bf16__eq__offset__r3_c5.json) | 0.249 / 0.2258 | [passed](program_peer/metal__bf16__eq__offset__r3_c5.json) | 0.2626 / 0.2194 |
| `metal/bf16/eq/offset/r3_c32` | correct | [passed](operator_peer/metal__bf16__eq__offset__r3_c32.json) | 0.2438 / 0.2258 | [passed](program_peer/metal__bf16__eq__offset__r3_c32.json) | 0.2564 / 0.246 |
| `metal/bf16/eq/offset/r3_c65` | correct | [passed](operator_peer/metal__bf16__eq__offset__r3_c65.json) | 0.243 / 0.2275 | [passed](program_peer/metal__bf16__eq__offset__r3_c65.json) | 0.2594 / 0.2217 |
| `metal/bf16/eq/offset/r3_c129` | correct | [passed](operator_peer/metal__bf16__eq__offset__r3_c129.json) | 0.2508 / 0.2268 | [passed](program_peer/metal__bf16__eq__offset__r3_c129.json) | 0.2656 / 0.22 |
| `metal/bf16/eq/offset/r16_c5` | correct | [passed](operator_peer/metal__bf16__eq__offset__r16_c5.json) | 0.252 / 0.2362 | [passed](program_peer/metal__bf16__eq__offset__r16_c5.json) | 0.2658 / 0.223 |
| `metal/bf16/eq/offset/r16_c32` | correct | [passed](operator_peer/metal__bf16__eq__offset__r16_c32.json) | 0.2588 / 0.2326 | [passed](program_peer/metal__bf16__eq__offset__r16_c32.json) | 0.272 / 0.2507 |
| `metal/bf16/eq/offset/r16_c65` | correct | [passed](operator_peer/metal__bf16__eq__offset__r16_c65.json) | 0.269 / 0.2305 | [passed](program_peer/metal__bf16__eq__offset__r16_c65.json) | 0.3084 / 0.2332 |
| `metal/bf16/eq/offset/r16_c129` | correct | [passed](operator_peer/metal__bf16__eq__offset__r16_c129.json) | 0.2824 / 0.2401 | [passed](program_peer/metal__bf16__eq__offset__r16_c129.json) | 0.319 / 0.2409 |
| `metal/bf16/eq/offset/r33_c5` | correct | [passed](operator_peer/metal__bf16__eq__offset__r33_c5.json) | 0.26 / 0.2242 | [passed](program_peer/metal__bf16__eq__offset__r33_c5.json) | 0.272 / 0.2263 |
| `metal/bf16/eq/offset/r33_c32` | correct | [passed](operator_peer/metal__bf16__eq__offset__r33_c32.json) | 0.2736 / 0.2521 | [passed](program_peer/metal__bf16__eq__offset__r33_c32.json) | 0.3078 / 0.2303 |
| `metal/bf16/eq/offset/r33_c65` | correct | [passed](operator_peer/metal__bf16__eq__offset__r33_c65.json) | 0.299 / 0.2356 | [passed](program_peer/metal__bf16__eq__offset__r33_c65.json) | 0.3318 / 0.237 |
| `metal/bf16/eq/offset/r33_c129` | correct | [passed](operator_peer/metal__bf16__eq__offset__r33_c129.json) | 0.3074 / 0.245 | [passed](program_peer/metal__bf16__eq__offset__r33_c129.json) | 0.3438 / 0.2421 |
| `metal/bf16/eq/offset/r128_c5` | correct | [passed](operator_peer/metal__bf16__eq__offset__r128_c5.json) | 0.2922 / 0.2307 | [passed](program_peer/metal__bf16__eq__offset__r128_c5.json) | 0.3122 / 0.2362 |
| `metal/bf16/eq/offset/r128_c32` | correct | [passed](operator_peer/metal__bf16__eq__offset__r128_c32.json) | 0.3458 / 0.2571 | [passed](program_peer/metal__bf16__eq__offset__r128_c32.json) | 0.3792 / 0.2467 |
| `metal/bf16/eq/offset/r128_c65` | correct | [passed](operator_peer/metal__bf16__eq__offset__r128_c65.json) | 0.3758 / 0.2588 | [passed](program_peer/metal__bf16__eq__offset__r128_c65.json) | 0.4206 / 0.322 |
| `metal/bf16/eq/offset/r128_c129` | correct | [passed](operator_peer/metal__bf16__eq__offset__r128_c129.json) | 0.4554 / 0.3346 | [passed](program_peer/metal__bf16__eq__offset__r128_c129.json) | 0.5232 / 0.3543 |
| `metal/bf16/lt/strided/r3_c5` | correct | [passed](operator_peer/metal__bf16__lt__strided__r3_c5.json) | 0.2454 / 0.2317 | [passed](program_peer/metal__bf16__lt__strided__r3_c5.json) | 0.2638 / 0.2291 |
| `metal/bf16/lt/strided/r3_c32` | correct | [passed](operator_peer/metal__bf16__lt__strided__r3_c32.json) | 0.245 / 0.2281 | [passed](program_peer/metal__bf16__lt__strided__r3_c32.json) | 0.2596 / 0.2165 |
| `metal/bf16/lt/strided/r3_c65` | correct | [passed](operator_peer/metal__bf16__lt__strided__r3_c65.json) | 0.244 / 0.259 | [passed](program_peer/metal__bf16__lt__strided__r3_c65.json) | 0.2594 / 0.2274 |
| `metal/bf16/lt/strided/r3_c129` | correct | [passed](operator_peer/metal__bf16__lt__strided__r3_c129.json) | 0.2418 / 0.2237 | [passed](program_peer/metal__bf16__lt__strided__r3_c129.json) | 0.2598 / 0.2419 |
| `metal/bf16/lt/strided/r16_c5` | correct | [passed](operator_peer/metal__bf16__lt__strided__r16_c5.json) | 0.2512 / 0.2177 | [passed](program_peer/metal__bf16__lt__strided__r16_c5.json) | 0.2874 / 0.2269 |
| `metal/bf16/lt/strided/r16_c32` | correct | [passed](operator_peer/metal__bf16__lt__strided__r16_c32.json) | 0.2586 / 0.2318 | [passed](program_peer/metal__bf16__lt__strided__r16_c32.json) | 0.2738 / 0.2337 |
| `metal/bf16/lt/strided/r16_c65` | correct | [passed](operator_peer/metal__bf16__lt__strided__r16_c65.json) | 0.2698 / 0.2338 | [passed](program_peer/metal__bf16__lt__strided__r16_c65.json) | 0.3116 / 0.2312 |
| `metal/bf16/lt/strided/r16_c129` | correct | [passed](operator_peer/metal__bf16__lt__strided__r16_c129.json) | 0.2794 / 0.238 | [passed](program_peer/metal__bf16__lt__strided__r16_c129.json) | 0.3284 / 0.2387 |
| `metal/bf16/lt/strided/r33_c5` | correct | [passed](operator_peer/metal__bf16__lt__strided__r33_c5.json) | 0.2572 / 0.2215 | [passed](program_peer/metal__bf16__lt__strided__r33_c5.json) | 0.2728 / 0.2333 |
| `metal/bf16/lt/strided/r33_c32` | correct | [passed](operator_peer/metal__bf16__lt__strided__r33_c32.json) | 0.2802 / 0.2279 | [passed](program_peer/metal__bf16__lt__strided__r33_c32.json) | 0.3204 / 0.2316 |
| `metal/bf16/lt/strided/r33_c65` | correct | [passed](operator_peer/metal__bf16__lt__strided__r33_c65.json) | 0.2914 / 0.2382 | [passed](program_peer/metal__bf16__lt__strided__r33_c65.json) | 0.331 / 0.2462 |
| `metal/bf16/lt/strided/r33_c129` | correct | [passed](operator_peer/metal__bf16__lt__strided__r33_c129.json) | 0.3034 / 0.2467 | [passed](program_peer/metal__bf16__lt__strided__r33_c129.json) | 0.3428 / 0.2498 |
| `metal/bf16/lt/strided/r128_c5` | correct | [passed](operator_peer/metal__bf16__lt__strided__r128_c5.json) | 0.2934 / 0.2367 | [passed](program_peer/metal__bf16__lt__strided__r128_c5.json) | 0.3182 / 0.2434 |
| `metal/bf16/lt/strided/r128_c32` | correct | [passed](operator_peer/metal__bf16__lt__strided__r128_c32.json) | 0.3552 / 0.2448 | [passed](program_peer/metal__bf16__lt__strided__r128_c32.json) | 0.348 / 0.2485 |
| `metal/bf16/lt/strided/r128_c65` | correct | [passed](operator_peer/metal__bf16__lt__strided__r128_c65.json) | 0.3512 / 0.2903 | [passed](program_peer/metal__bf16__lt__strided__r128_c65.json) | 0.4126 / 0.3207 |
| `metal/bf16/lt/strided/r128_c129` | correct | [passed](operator_peer/metal__bf16__lt__strided__r128_c129.json) | 0.4464 / 0.33 | [passed](program_peer/metal__bf16__lt__strided__r128_c129.json) | 0.4822 / 0.3555 |
| `metal/bf16/lt/offset/r3_c5` | correct | [passed](operator_peer/metal__bf16__lt__offset__r3_c5.json) | 0.2488 / 0.2496 | [passed](program_peer/metal__bf16__lt__offset__r3_c5.json) | 0.2596 / 0.2268 |
| `metal/bf16/lt/offset/r3_c32` | correct | [passed](operator_peer/metal__bf16__lt__offset__r3_c32.json) | 0.2484 / 0.2194 | [passed](program_peer/metal__bf16__lt__offset__r3_c32.json) | 0.2678 / 0.2364 |
| `metal/bf16/lt/offset/r3_c65` | correct | [passed](operator_peer/metal__bf16__lt__offset__r3_c65.json) | 0.2458 / 0.2229 | [passed](program_peer/metal__bf16__lt__offset__r3_c65.json) | 0.2518 / 0.2361 |
| `metal/bf16/lt/offset/r3_c129` | correct | [passed](operator_peer/metal__bf16__lt__offset__r3_c129.json) | 0.2506 / 0.2307 | [passed](program_peer/metal__bf16__lt__offset__r3_c129.json) | 0.261 / 0.2168 |
| `metal/bf16/lt/offset/r16_c5` | correct | [passed](operator_peer/metal__bf16__lt__offset__r16_c5.json) | 0.2502 / 0.2257 | [passed](program_peer/metal__bf16__lt__offset__r16_c5.json) | 0.2586 / 0.2213 |
| `metal/bf16/lt/offset/r16_c32` | correct | [passed](operator_peer/metal__bf16__lt__offset__r16_c32.json) | 0.2608 / 0.223 | [passed](program_peer/metal__bf16__lt__offset__r16_c32.json) | 0.2456 / 0.2205 |
| `metal/bf16/lt/offset/r16_c65` | correct | [passed](operator_peer/metal__bf16__lt__offset__r16_c65.json) | 0.2306 / 0.2292 | [passed](program_peer/metal__bf16__lt__offset__r16_c65.json) | 0.2776 / 0.2312 |
| `metal/bf16/lt/offset/r16_c129` | correct | [passed](operator_peer/metal__bf16__lt__offset__r16_c129.json) | 0.2802 / 0.2366 | [passed](program_peer/metal__bf16__lt__offset__r16_c129.json) | 0.3286 / 0.2405 |
| `metal/bf16/lt/offset/r33_c5` | correct | [passed](operator_peer/metal__bf16__lt__offset__r33_c5.json) | 0.2578 / 0.234 | [passed](program_peer/metal__bf16__lt__offset__r33_c5.json) | 0.2716 / 0.2163 |
| `metal/bf16/lt/offset/r33_c32` | correct | [passed](operator_peer/metal__bf16__lt__offset__r33_c32.json) | 0.279 / 0.2394 | [passed](program_peer/metal__bf16__lt__offset__r33_c32.json) | 0.3204 / 0.2358 |
| `metal/bf16/lt/offset/r33_c65` | correct | [passed](operator_peer/metal__bf16__lt__offset__r33_c65.json) | 0.2878 / 0.2561 | [passed](program_peer/metal__bf16__lt__offset__r33_c65.json) | 0.2966 / 0.2407 |
| `metal/bf16/lt/offset/r33_c129` | correct | [passed](operator_peer/metal__bf16__lt__offset__r33_c129.json) | 0.306 / 0.2463 | [passed](program_peer/metal__bf16__lt__offset__r33_c129.json) | 0.3322 / 0.2471 |
| `metal/bf16/lt/offset/r128_c5` | correct | [passed](operator_peer/metal__bf16__lt__offset__r128_c5.json) | 0.293 / 0.2219 | [passed](program_peer/metal__bf16__lt__offset__r128_c5.json) | 0.3172 / 0.2258 |
| `metal/bf16/lt/offset/r128_c32` | correct | [passed](operator_peer/metal__bf16__lt__offset__r128_c32.json) | 0.3428 / 0.2472 | [passed](program_peer/metal__bf16__lt__offset__r128_c32.json) | 0.372 / 0.2449 |
| `metal/bf16/lt/offset/r128_c65` | correct | [passed](operator_peer/metal__bf16__lt__offset__r128_c65.json) | 0.3802 / 0.2968 | [passed](program_peer/metal__bf16__lt__offset__r128_c65.json) | 0.419 / 0.266 |
| `metal/bf16/lt/offset/r128_c129` | correct | [passed](operator_peer/metal__bf16__lt__offset__r128_c129.json) | 0.4534 / 0.34 | [passed](program_peer/metal__bf16__lt__offset__r128_c129.json) | 0.53 / 0.3559 |
| `metal/bf16/gt/strided/r3_c5` | correct | [passed](operator_peer/metal__bf16__gt__strided__r3_c5.json) | 0.2518 / 0.2622 | [passed](program_peer/metal__bf16__gt__strided__r3_c5.json) | 0.2584 / 0.2252 |
| `metal/bf16/gt/strided/r3_c32` | correct | [passed](operator_peer/metal__bf16__gt__strided__r3_c32.json) | 0.2428 / 0.2197 | [passed](program_peer/metal__bf16__gt__strided__r3_c32.json) | 0.2276 / 0.2365 |
| `metal/bf16/gt/strided/r3_c65` | correct | [passed](operator_peer/metal__bf16__gt__strided__r3_c65.json) | 0.2438 / 0.2392 | [passed](program_peer/metal__bf16__gt__strided__r3_c65.json) | 0.2426 / 0.2311 |
| `metal/bf16/gt/strided/r3_c129` | correct | [passed](operator_peer/metal__bf16__gt__strided__r3_c129.json) | 0.2502 / 0.2305 | [passed](program_peer/metal__bf16__gt__strided__r3_c129.json) | 0.2648 / 0.2264 |
| `metal/bf16/gt/strided/r16_c5` | correct | [passed](operator_peer/metal__bf16__gt__strided__r16_c5.json) | 0.259 / 0.2287 | [passed](program_peer/metal__bf16__gt__strided__r16_c5.json) | 0.2658 / 0.2276 |
| `metal/bf16/gt/strided/r16_c32` | correct | [passed](operator_peer/metal__bf16__gt__strided__r16_c32.json) | 0.2596 / 0.2334 | [passed](program_peer/metal__bf16__gt__strided__r16_c32.json) | 0.2942 / 0.247 |
| `metal/bf16/gt/strided/r16_c65` | correct | [passed](operator_peer/metal__bf16__gt__strided__r16_c65.json) | 0.2682 / 0.2394 | [passed](program_peer/metal__bf16__gt__strided__r16_c65.json) | 0.3178 / 0.2339 |
| `metal/bf16/gt/strided/r16_c129` | correct | [passed](operator_peer/metal__bf16__gt__strided__r16_c129.json) | 0.2524 / 0.2407 | [passed](program_peer/metal__bf16__gt__strided__r16_c129.json) | 0.3212 / 0.2418 |
| `metal/bf16/gt/strided/r33_c5` | correct | [passed](operator_peer/metal__bf16__gt__strided__r33_c5.json) | 0.2562 / 0.2418 | [passed](program_peer/metal__bf16__gt__strided__r33_c5.json) | 0.27 / 0.2213 |
| `metal/bf16/gt/strided/r33_c32` | correct | [passed](operator_peer/metal__bf16__gt__strided__r33_c32.json) | 0.2798 / 0.2312 | [passed](program_peer/metal__bf16__gt__strided__r33_c32.json) | 0.3192 / 0.2366 |
| `metal/bf16/gt/strided/r33_c65` | correct | [passed](operator_peer/metal__bf16__gt__strided__r33_c65.json) | 0.2906 / 0.2307 | [passed](program_peer/metal__bf16__gt__strided__r33_c65.json) | 0.3198 / 0.2395 |
| `metal/bf16/gt/strided/r33_c129` | correct | [passed](operator_peer/metal__bf16__gt__strided__r33_c129.json) | 0.303 / 0.2485 | [passed](program_peer/metal__bf16__gt__strided__r33_c129.json) | 0.3408 / 0.2504 |
| `metal/bf16/gt/strided/r128_c5` | correct | [passed](operator_peer/metal__bf16__gt__strided__r128_c5.json) | 0.2698 / 0.234 | [passed](program_peer/metal__bf16__gt__strided__r128_c5.json) | 0.3232 / 0.2318 |
| `metal/bf16/gt/strided/r128_c32` | correct | [passed](operator_peer/metal__bf16__gt__strided__r128_c32.json) | 0.3468 / 0.2448 | [passed](program_peer/metal__bf16__gt__strided__r128_c32.json) | 0.3726 / 0.2504 |
| `metal/bf16/gt/strided/r128_c65` | correct | [passed](operator_peer/metal__bf16__gt__strided__r128_c65.json) | 0.3856 / 0.2602 | [passed](program_peer/metal__bf16__gt__strided__r128_c65.json) | 0.393 / 0.295 |
| `metal/bf16/gt/strided/r128_c129` | correct | [passed](operator_peer/metal__bf16__gt__strided__r128_c129.json) | 0.4608 / 0.3433 | [passed](program_peer/metal__bf16__gt__strided__r128_c129.json) | 0.533 / 0.3541 |
| `metal/bf16/gt/offset/r3_c5` | correct | [passed](operator_peer/metal__bf16__gt__offset__r3_c5.json) | 0.2424 / 0.2317 | [passed](program_peer/metal__bf16__gt__offset__r3_c5.json) | 0.2628 / 0.2252 |
| `metal/bf16/gt/offset/r3_c32` | correct | [passed](operator_peer/metal__bf16__gt__offset__r3_c32.json) | 0.2474 / 0.2304 | [passed](program_peer/metal__bf16__gt__offset__r3_c32.json) | 0.258 / 0.2241 |
| `metal/bf16/gt/offset/r3_c65` | correct | [passed](operator_peer/metal__bf16__gt__offset__r3_c65.json) | 0.2504 / 0.2192 | [passed](program_peer/metal__bf16__gt__offset__r3_c65.json) | 0.2578 / 0.2283 |
| `metal/bf16/gt/offset/r3_c129` | correct | [passed](operator_peer/metal__bf16__gt__offset__r3_c129.json) | 0.2502 / 0.2573 | [passed](program_peer/metal__bf16__gt__offset__r3_c129.json) | 0.2642 / 0.2269 |
| `metal/bf16/gt/offset/r16_c5` | correct | [passed](operator_peer/metal__bf16__gt__offset__r16_c5.json) | 0.254 / 0.2503 | [passed](program_peer/metal__bf16__gt__offset__r16_c5.json) | 0.2572 / 0.2366 |
| `metal/bf16/gt/offset/r16_c32` | correct | [passed](operator_peer/metal__bf16__gt__offset__r16_c32.json) | 0.2588 / 0.2096 | [passed](program_peer/metal__bf16__gt__offset__r16_c32.json) | 0.2728 / 0.228 |
| `metal/bf16/gt/offset/r16_c65` | correct | [passed](operator_peer/metal__bf16__gt__offset__r16_c65.json) | 0.2676 / 0.2334 | [passed](program_peer/metal__bf16__gt__offset__r16_c65.json) | 0.3022 / 0.2364 |
| `metal/bf16/gt/offset/r16_c129` | correct | [passed](operator_peer/metal__bf16__gt__offset__r16_c129.json) | 0.2442 / 0.2371 | [passed](program_peer/metal__bf16__gt__offset__r16_c129.json) | 0.2828 / 0.2385 |
| `metal/bf16/gt/offset/r33_c5` | correct | [passed](operator_peer/metal__bf16__gt__offset__r33_c5.json) | 0.2522 / 0.2222 | [passed](program_peer/metal__bf16__gt__offset__r33_c5.json) | 0.2708 / 0.2311 |
| `metal/bf16/gt/offset/r33_c32` | correct | [passed](operator_peer/metal__bf16__gt__offset__r33_c32.json) | 0.278 / 0.2297 | [passed](program_peer/metal__bf16__gt__offset__r33_c32.json) | 0.32 / 0.2362 |
| `metal/bf16/gt/offset/r33_c65` | correct | [passed](operator_peer/metal__bf16__gt__offset__r33_c65.json) | 0.2932 / 0.2369 | [passed](program_peer/metal__bf16__gt__offset__r33_c65.json) | 0.317 / 0.2406 |
| `metal/bf16/gt/offset/r33_c129` | correct | [passed](operator_peer/metal__bf16__gt__offset__r33_c129.json) | 0.3046 / 0.2459 | [passed](program_peer/metal__bf16__gt__offset__r33_c129.json) | 0.3456 / 0.2456 |
| `metal/bf16/gt/offset/r128_c5` | correct | [passed](operator_peer/metal__bf16__gt__offset__r128_c5.json) | 0.2952 / 0.2275 | [passed](program_peer/metal__bf16__gt__offset__r128_c5.json) | 0.3084 / 0.2301 |
| `metal/bf16/gt/offset/r128_c32` | correct | [passed](operator_peer/metal__bf16__gt__offset__r128_c32.json) | 0.3302 / 0.2472 | [passed](program_peer/metal__bf16__gt__offset__r128_c32.json) | 0.3784 / 0.249 |
| `metal/bf16/gt/offset/r128_c65` | correct | [passed](operator_peer/metal__bf16__gt__offset__r128_c65.json) | 0.3814 / 0.2613 | [passed](program_peer/metal__bf16__gt__offset__r128_c65.json) | 0.4188 / 0.2743 |
| `metal/bf16/gt/offset/r128_c129` | correct | [passed](operator_peer/metal__bf16__gt__offset__r128_c129.json) | 0.455 / 0.3334 | [passed](program_peer/metal__bf16__gt__offset__r128_c129.json) | 0.4552 / 0.336 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
