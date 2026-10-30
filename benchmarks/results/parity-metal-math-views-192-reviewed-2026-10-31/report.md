# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7f07fe6aa842ce0791d10fa8bf8c43bab9f74a0117f0c62fa05022a2386cbde7`  
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
| `metal/f32/exp/strided/r3_c5` | correct | [passed](operator_peer/metal__f32__exp__strided__r3_c5.json) | 0.2676 / 0.1968 | [passed](program_peer/metal__f32__exp__strided__r3_c5.json) | 0.2526 / 0.2347 |
| `metal/f32/exp/strided/r3_c32` | correct | [passed](operator_peer/metal__f32__exp__strided__r3_c32.json) | 0.2416 / 0.2039 | [passed](program_peer/metal__f32__exp__strided__r3_c32.json) | 0.2516 / 0.1934 |
| `metal/f32/exp/strided/r3_c65` | correct | [passed](operator_peer/metal__f32__exp__strided__r3_c65.json) | 0.2438 / 0.1897 | [passed](program_peer/metal__f32__exp__strided__r3_c65.json) | 0.253 / 0.1929 |
| `metal/f32/exp/strided/r3_c129` | correct | [passed](operator_peer/metal__f32__exp__strided__r3_c129.json) | 0.2362 / 0.19 | [passed](program_peer/metal__f32__exp__strided__r3_c129.json) | 0.2538 / 0.1967 |
| `metal/f32/exp/strided/r16_c5` | correct | [passed](operator_peer/metal__f32__exp__strided__r16_c5.json) | 0.2488 / 0.1908 | [passed](program_peer/metal__f32__exp__strided__r16_c5.json) | 0.2586 / 0.2101 |
| `metal/f32/exp/strided/r16_c32` | correct | [passed](operator_peer/metal__f32__exp__strided__r16_c32.json) | 0.2578 / 0.1946 | [passed](program_peer/metal__f32__exp__strided__r16_c32.json) | 0.2712 / 0.203 |
| `metal/f32/exp/strided/r16_c65` | correct | [passed](operator_peer/metal__f32__exp__strided__r16_c65.json) | 0.2628 / 0.2017 | [passed](program_peer/metal__f32__exp__strided__r16_c65.json) | 0.222 / 0.2061 |
| `metal/f32/exp/strided/r16_c129` | correct | [passed](operator_peer/metal__f32__exp__strided__r16_c129.json) | 0.2806 / 0.2221 | [passed](program_peer/metal__f32__exp__strided__r16_c129.json) | 0.2816 / 0.2258 |
| `metal/f32/exp/strided/r33_c5` | correct | [passed](operator_peer/metal__f32__exp__strided__r33_c5.json) | 0.2556 / 0.2046 | [passed](program_peer/metal__f32__exp__strided__r33_c5.json) | 0.2588 / 0.183 |
| `metal/f32/exp/strided/r33_c32` | correct | [passed](operator_peer/metal__f32__exp__strided__r33_c32.json) | 0.271 / 0.2098 | [passed](program_peer/metal__f32__exp__strided__r33_c32.json) | 0.2806 / 0.1989 |
| `metal/f32/exp/strided/r33_c65` | correct | [passed](operator_peer/metal__f32__exp__strided__r33_c65.json) | 0.282 / 0.2081 | [passed](program_peer/metal__f32__exp__strided__r33_c65.json) | 0.2952 / 0.2346 |
| `metal/f32/exp/strided/r33_c129` | correct | [passed](operator_peer/metal__f32__exp__strided__r33_c129.json) | 0.3058 / 0.2375 | [passed](program_peer/metal__f32__exp__strided__r33_c129.json) | 0.3106 / 0.2404 |
| `metal/f32/exp/strided/r128_c5` | correct | [passed](operator_peer/metal__f32__exp__strided__r128_c5.json) | 0.2902 / 0.2062 | [passed](program_peer/metal__f32__exp__strided__r128_c5.json) | 0.2982 / 0.2081 |
| `metal/f32/exp/strided/r128_c32` | correct | [passed](operator_peer/metal__f32__exp__strided__r128_c32.json) | 0.3416 / 0.2633 | [passed](program_peer/metal__f32__exp__strided__r128_c32.json) | 0.3426 / 0.2426 |
| `metal/f32/exp/strided/r128_c65` | correct | [passed](operator_peer/metal__f32__exp__strided__r128_c65.json) | 0.3818 / 0.2822 | [passed](program_peer/metal__f32__exp__strided__r128_c65.json) | 0.3838 / 0.285 |
| `metal/f32/exp/strided/r128_c129` | correct | [passed](operator_peer/metal__f32__exp__strided__r128_c129.json) | 0.4496 / 0.4161 | [passed](program_peer/metal__f32__exp__strided__r128_c129.json) | 0.4638 / 0.4296 |
| `metal/f32/exp/offset/r3_c5` | correct | [passed](operator_peer/metal__f32__exp__offset__r3_c5.json) | 0.2452 / 0.2251 | [passed](program_peer/metal__f32__exp__offset__r3_c5.json) | 0.2082 / 0.2021 |
| `metal/f32/exp/offset/r3_c32` | correct | [passed](operator_peer/metal__f32__exp__offset__r3_c32.json) | 0.2442 / 0.1933 | [passed](program_peer/metal__f32__exp__offset__r3_c32.json) | 0.2492 / 0.2006 |
| `metal/f32/exp/offset/r3_c65` | correct | [passed](operator_peer/metal__f32__exp__offset__r3_c65.json) | 0.2462 / 0.197 | [passed](program_peer/metal__f32__exp__offset__r3_c65.json) | 0.2466 / 0.1992 |
| `metal/f32/exp/offset/r3_c129` | correct | [passed](operator_peer/metal__f32__exp__offset__r3_c129.json) | 0.1928 / 0.1941 | [passed](program_peer/metal__f32__exp__offset__r3_c129.json) | 0.2546 / 0.1991 |
| `metal/f32/exp/offset/r16_c5` | correct | [passed](operator_peer/metal__f32__exp__offset__r16_c5.json) | 0.2472 / 0.2 | [passed](program_peer/metal__f32__exp__offset__r16_c5.json) | 0.2242 / 0.193 |
| `metal/f32/exp/offset/r16_c32` | correct | [passed](operator_peer/metal__f32__exp__offset__r16_c32.json) | 0.2578 / 0.2074 | [passed](program_peer/metal__f32__exp__offset__r16_c32.json) | 0.2536 / 0.2252 |
| `metal/f32/exp/offset/r16_c65` | correct | [passed](operator_peer/metal__f32__exp__offset__r16_c65.json) | 0.2646 / 0.21 | [passed](program_peer/metal__f32__exp__offset__r16_c65.json) | 0.2724 / 0.2085 |
| `metal/f32/exp/offset/r16_c129` | correct | [passed](operator_peer/metal__f32__exp__offset__r16_c129.json) | 0.2794 / 0.2179 | [passed](program_peer/metal__f32__exp__offset__r16_c129.json) | 0.2818 / 0.2206 |
| `metal/f32/exp/offset/r33_c5` | correct | [passed](operator_peer/metal__f32__exp__offset__r33_c5.json) | 0.2554 / 0.1973 | [passed](program_peer/metal__f32__exp__offset__r33_c5.json) | 0.2556 / 0.1868 |
| `metal/f32/exp/offset/r33_c32` | correct | [passed](operator_peer/metal__f32__exp__offset__r33_c32.json) | 0.276 / 0.2067 | [passed](program_peer/metal__f32__exp__offset__r33_c32.json) | 0.2814 / 0.2084 |
| `metal/f32/exp/offset/r33_c65` | correct | [passed](operator_peer/metal__f32__exp__offset__r33_c65.json) | 0.282 / 0.2119 | [passed](program_peer/metal__f32__exp__offset__r33_c65.json) | 0.2934 / 0.223 |
| `metal/f32/exp/offset/r33_c129` | correct | [passed](operator_peer/metal__f32__exp__offset__r33_c129.json) | 0.2776 / 0.2384 | [passed](program_peer/metal__f32__exp__offset__r33_c129.json) | 0.2996 / 0.2396 |
| `metal/f32/exp/offset/r128_c5` | correct | [passed](operator_peer/metal__f32__exp__offset__r128_c5.json) | 0.291 / 0.2043 | [passed](program_peer/metal__f32__exp__offset__r128_c5.json) | 0.2764 / 0.2143 |
| `metal/f32/exp/offset/r128_c32` | correct | [passed](operator_peer/metal__f32__exp__offset__r128_c32.json) | 0.343 / 0.2395 | [passed](program_peer/metal__f32__exp__offset__r128_c32.json) | 0.322 / 0.2297 |
| `metal/f32/exp/offset/r128_c65` | correct | [passed](operator_peer/metal__f32__exp__offset__r128_c65.json) | 0.3518 / 0.2975 | [passed](program_peer/metal__f32__exp__offset__r128_c65.json) | 0.385 / 0.3258 |
| `metal/f32/exp/offset/r128_c129` | correct | [passed](operator_peer/metal__f32__exp__offset__r128_c129.json) | 0.4388 / 0.4127 | [passed](program_peer/metal__f32__exp__offset__r128_c129.json) | 0.4622 / 0.4276 |
| `metal/f32/log/strided/r3_c5` | correct | [passed](operator_peer/metal__f32__log__strided__r3_c5.json) | 0.2466 / 0.1902 | [passed](program_peer/metal__f32__log__strided__r3_c5.json) | 0.2528 / 0.197 |
| `metal/f32/log/strided/r3_c32` | correct | [passed](operator_peer/metal__f32__log__strided__r3_c32.json) | 0.2402 / 0.1969 | [passed](program_peer/metal__f32__log__strided__r3_c32.json) | 0.2512 / 0.2106 |
| `metal/f32/log/strided/r3_c65` | correct | [passed](operator_peer/metal__f32__log__strided__r3_c65.json) | 0.2444 / 0.2048 | [passed](program_peer/metal__f32__log__strided__r3_c65.json) | 0.2516 / 0.2131 |
| `metal/f32/log/strided/r3_c129` | correct | [passed](operator_peer/metal__f32__log__strided__r3_c129.json) | 0.25 / 0.1919 | [passed](program_peer/metal__f32__log__strided__r3_c129.json) | 0.2568 / 0.2384 |
| `metal/f32/log/strided/r16_c5` | correct | [passed](operator_peer/metal__f32__log__strided__r16_c5.json) | 0.2514 / 0.1957 | [passed](program_peer/metal__f32__log__strided__r16_c5.json) | 0.262 / 0.1887 |
| `metal/f32/log/strided/r16_c32` | correct | [passed](operator_peer/metal__f32__log__strided__r16_c32.json) | 0.2654 / 0.2158 | [passed](program_peer/metal__f32__log__strided__r16_c32.json) | 0.2628 / 0.2041 |
| `metal/f32/log/strided/r16_c65` | correct | [passed](operator_peer/metal__f32__log__strided__r16_c65.json) | 0.265 / 0.207 | [passed](program_peer/metal__f32__log__strided__r16_c65.json) | 0.272 / 0.2065 |
| `metal/f32/log/strided/r16_c129` | correct | [passed](operator_peer/metal__f32__log__strided__r16_c129.json) | 0.263 / 0.2205 | [passed](program_peer/metal__f32__log__strided__r16_c129.json) | 0.2842 / 0.2297 |
| `metal/f32/log/strided/r33_c5` | correct | [passed](operator_peer/metal__f32__log__strided__r33_c5.json) | 0.2514 / 0.2004 | [passed](program_peer/metal__f32__log__strided__r33_c5.json) | 0.2596 / 0.2062 |
| `metal/f32/log/strided/r33_c32` | correct | [passed](operator_peer/metal__f32__log__strided__r33_c32.json) | 0.2782 / 0.2049 | [passed](program_peer/metal__f32__log__strided__r33_c32.json) | 0.2828 / 0.2168 |
| `metal/f32/log/strided/r33_c65` | correct | [passed](operator_peer/metal__f32__log__strided__r33_c65.json) | 0.287 / 0.2192 | [passed](program_peer/metal__f32__log__strided__r33_c65.json) | 0.2554 / 0.2225 |
| `metal/f32/log/strided/r33_c129` | correct | [passed](operator_peer/metal__f32__log__strided__r33_c129.json) | 0.2792 / 0.2407 | [passed](program_peer/metal__f32__log__strided__r33_c129.json) | 0.3146 / 0.2419 |
| `metal/f32/log/strided/r128_c5` | correct | [passed](operator_peer/metal__f32__log__strided__r128_c5.json) | 0.2792 / 0.2003 | [passed](program_peer/metal__f32__log__strided__r128_c5.json) | 0.2946 / 0.1962 |
| `metal/f32/log/strided/r128_c32` | correct | [passed](operator_peer/metal__f32__log__strided__r128_c32.json) | 0.3504 / 0.2377 | [passed](program_peer/metal__f32__log__strided__r128_c32.json) | 0.352 / 0.2416 |
| `metal/f32/log/strided/r128_c65` | correct | [passed](operator_peer/metal__f32__log__strided__r128_c65.json) | 0.378 / 0.2822 | [passed](program_peer/metal__f32__log__strided__r128_c65.json) | 0.3912 / 0.3407 |
| `metal/f32/log/strided/r128_c129` | correct | [passed](operator_peer/metal__f32__log__strided__r128_c129.json) | 0.4514 / 0.4156 | [passed](program_peer/metal__f32__log__strided__r128_c129.json) | 0.4606 / 0.4311 |
| `metal/f32/log/offset/r3_c5` | correct | [passed](operator_peer/metal__f32__log__offset__r3_c5.json) | 0.249 / 0.1965 | [passed](program_peer/metal__f32__log__offset__r3_c5.json) | 0.2538 / 0.2038 |
| `metal/f32/log/offset/r3_c32` | correct | [passed](operator_peer/metal__f32__log__offset__r3_c32.json) | 0.2458 / 0.2 | [passed](program_peer/metal__f32__log__offset__r3_c32.json) | 0.2482 / 0.2199 |
| `metal/f32/log/offset/r3_c65` | correct | [passed](operator_peer/metal__f32__log__offset__r3_c65.json) | 0.195 / 0.1872 | [passed](program_peer/metal__f32__log__offset__r3_c65.json) | 0.2514 / 0.1904 |
| `metal/f32/log/offset/r3_c129` | correct | [passed](operator_peer/metal__f32__log__offset__r3_c129.json) | 0.2346 / 0.2108 | [passed](program_peer/metal__f32__log__offset__r3_c129.json) | 0.2486 / 0.2028 |
| `metal/f32/log/offset/r16_c5` | correct | [passed](operator_peer/metal__f32__log__offset__r16_c5.json) | 0.2554 / 0.1804 | [passed](program_peer/metal__f32__log__offset__r16_c5.json) | 0.2568 / 0.1935 |
| `metal/f32/log/offset/r16_c32` | correct | [passed](operator_peer/metal__f32__log__offset__r16_c32.json) | 0.2568 / 0.2002 | [passed](program_peer/metal__f32__log__offset__r16_c32.json) | 0.2628 / 0.2057 |
| `metal/f32/log/offset/r16_c65` | correct | [passed](operator_peer/metal__f32__log__offset__r16_c65.json) | 0.253 / 0.2056 | [passed](program_peer/metal__f32__log__offset__r16_c65.json) | 0.2312 / 0.2182 |
| `metal/f32/log/offset/r16_c129` | correct | [passed](operator_peer/metal__f32__log__offset__r16_c129.json) | 0.2788 / 0.2191 | [passed](program_peer/metal__f32__log__offset__r16_c129.json) | 0.2792 / 0.2219 |
| `metal/f32/log/offset/r33_c5` | correct | [passed](operator_peer/metal__f32__log__offset__r33_c5.json) | 0.2534 / 0.1929 | [passed](program_peer/metal__f32__log__offset__r33_c5.json) | 0.242 / 0.1885 |
| `metal/f32/log/offset/r33_c32` | correct | [passed](operator_peer/metal__f32__log__offset__r33_c32.json) | 0.2772 / 0.2195 | [passed](program_peer/metal__f32__log__offset__r33_c32.json) | 0.2744 / 0.2147 |
| `metal/f32/log/offset/r33_c65` | correct | [passed](operator_peer/metal__f32__log__offset__r33_c65.json) | 0.2598 / 0.2223 | [passed](program_peer/metal__f32__log__offset__r33_c65.json) | 0.2908 / 0.2234 |
| `metal/f32/log/offset/r33_c129` | correct | [passed](operator_peer/metal__f32__log__offset__r33_c129.json) | 0.2954 / 0.2403 | [passed](program_peer/metal__f32__log__offset__r33_c129.json) | 0.2876 / 0.2411 |
| `metal/f32/log/offset/r128_c5` | correct | [passed](operator_peer/metal__f32__log__offset__r128_c5.json) | 0.294 / 0.2008 | [passed](program_peer/metal__f32__log__offset__r128_c5.json) | 0.2956 / 0.2178 |
| `metal/f32/log/offset/r128_c32` | correct | [passed](operator_peer/metal__f32__log__offset__r128_c32.json) | 0.3356 / 0.2401 | [passed](program_peer/metal__f32__log__offset__r128_c32.json) | 0.3414 / 0.2428 |
| `metal/f32/log/offset/r128_c65` | correct | [passed](operator_peer/metal__f32__log__offset__r128_c65.json) | 0.3766 / 0.2999 | [passed](program_peer/metal__f32__log__offset__r128_c65.json) | 0.385 / 0.3425 |
| `metal/f32/log/offset/r128_c129` | correct | [passed](operator_peer/metal__f32__log__offset__r128_c129.json) | 0.4236 / 0.4137 | [passed](program_peer/metal__f32__log__offset__r128_c129.json) | 0.4644 / 0.4133 |
| `metal/f32/sqrt/strided/r3_c5` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r3_c5.json) | 0.2356 / 0.2023 | [passed](program_peer/metal__f32__sqrt__strided__r3_c5.json) | 0.2492 / 0.1949 |
| `metal/f32/sqrt/strided/r3_c32` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r3_c32.json) | 0.2306 / 0.1917 | [passed](program_peer/metal__f32__sqrt__strided__r3_c32.json) | 0.2598 / 0.1933 |
| `metal/f32/sqrt/strided/r3_c65` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r3_c65.json) | 0.2484 / 0.1948 | [passed](program_peer/metal__f32__sqrt__strided__r3_c65.json) | 0.2502 / 0.1986 |
| `metal/f32/sqrt/strided/r3_c129` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r3_c129.json) | 0.246 / 0.2089 | [passed](program_peer/metal__f32__sqrt__strided__r3_c129.json) | 0.252 / 0.2042 |
| `metal/f32/sqrt/strided/r16_c5` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r16_c5.json) | 0.249 / 0.1791 | [passed](program_peer/metal__f32__sqrt__strided__r16_c5.json) | 0.2532 / 0.1905 |
| `metal/f32/sqrt/strided/r16_c32` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r16_c32.json) | 0.2552 / 0.1969 | [passed](program_peer/metal__f32__sqrt__strided__r16_c32.json) | 0.2648 / 0.217 |
| `metal/f32/sqrt/strided/r16_c65` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r16_c65.json) | 0.2068 / 0.2101 | [passed](program_peer/metal__f32__sqrt__strided__r16_c65.json) | 0.2694 / 0.2152 |
| `metal/f32/sqrt/strided/r16_c129` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r16_c129.json) | 0.2734 / 0.2183 | [passed](program_peer/metal__f32__sqrt__strided__r16_c129.json) | 0.286 / 0.2233 |
| `metal/f32/sqrt/strided/r33_c5` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r33_c5.json) | 0.257 / 0.1944 | [passed](program_peer/metal__f32__sqrt__strided__r33_c5.json) | 0.251 / 0.2159 |
| `metal/f32/sqrt/strided/r33_c32` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r33_c32.json) | 0.2804 / 0.2088 | [passed](program_peer/metal__f32__sqrt__strided__r33_c32.json) | 0.2644 / 0.2732 |
| `metal/f32/sqrt/strided/r33_c65` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r33_c65.json) | 0.2936 / 0.2213 | [passed](program_peer/metal__f32__sqrt__strided__r33_c65.json) | 0.2914 / 0.2341 |
| `metal/f32/sqrt/strided/r33_c129` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r33_c129.json) | 0.3052 / 0.2556 | [passed](program_peer/metal__f32__sqrt__strided__r33_c129.json) | 0.3128 / 0.2423 |
| `metal/f32/sqrt/strided/r128_c5` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r128_c5.json) | 0.2842 / 0.2008 | [passed](program_peer/metal__f32__sqrt__strided__r128_c5.json) | 0.2946 / 0.2569 |
| `metal/f32/sqrt/strided/r128_c32` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r128_c32.json) | 0.3502 / 0.24 | [passed](program_peer/metal__f32__sqrt__strided__r128_c32.json) | 0.3496 / 0.2439 |
| `metal/f32/sqrt/strided/r128_c65` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r128_c65.json) | 0.3824 / 0.3057 | [passed](program_peer/metal__f32__sqrt__strided__r128_c65.json) | 0.3674 / 0.3431 |
| `metal/f32/sqrt/strided/r128_c129` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r128_c129.json) | 0.4496 / 0.4098 | [passed](program_peer/metal__f32__sqrt__strided__r128_c129.json) | 0.4292 / 0.429 |
| `metal/f32/sqrt/offset/r3_c5` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r3_c5.json) | 0.2394 / 0.2147 | [passed](program_peer/metal__f32__sqrt__offset__r3_c5.json) | 0.2532 / 0.1957 |
| `metal/f32/sqrt/offset/r3_c32` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r3_c32.json) | 0.2116 / 0.1945 | [passed](program_peer/metal__f32__sqrt__offset__r3_c32.json) | 0.2468 / 0.2108 |
| `metal/f32/sqrt/offset/r3_c65` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r3_c65.json) | 0.244 / 0.187 | [passed](program_peer/metal__f32__sqrt__offset__r3_c65.json) | 0.2522 / 0.2188 |
| `metal/f32/sqrt/offset/r3_c129` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r3_c129.json) | 0.2468 / 0.1991 | [passed](program_peer/metal__f32__sqrt__offset__r3_c129.json) | 0.257 / 0.197 |
| `metal/f32/sqrt/offset/r16_c5` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r16_c5.json) | 0.2502 / 0.2042 | [passed](program_peer/metal__f32__sqrt__offset__r16_c5.json) | 0.2054 / 0.1916 |
| `metal/f32/sqrt/offset/r16_c32` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r16_c32.json) | 0.2566 / 0.203 | [passed](program_peer/metal__f32__sqrt__offset__r16_c32.json) | 0.2624 / 0.2146 |
| `metal/f32/sqrt/offset/r16_c65` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r16_c65.json) | 0.2668 / 0.2007 | [passed](program_peer/metal__f32__sqrt__offset__r16_c65.json) | 0.2542 / 0.1993 |
| `metal/f32/sqrt/offset/r16_c129` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r16_c129.json) | 0.2766 / 0.2168 | [passed](program_peer/metal__f32__sqrt__offset__r16_c129.json) | 0.2382 / 0.224 |
| `metal/f32/sqrt/offset/r33_c5` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r33_c5.json) | 0.2562 / 0.1931 | [passed](program_peer/metal__f32__sqrt__offset__r33_c5.json) | 0.2588 / 0.1906 |
| `metal/f32/sqrt/offset/r33_c32` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r33_c32.json) | 0.2316 / 0.2027 | [passed](program_peer/metal__f32__sqrt__offset__r33_c32.json) | 0.2876 / 0.2144 |
| `metal/f32/sqrt/offset/r33_c65` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r33_c65.json) | 0.2854 / 0.2183 | [passed](program_peer/metal__f32__sqrt__offset__r33_c65.json) | 0.2936 / 0.2226 |
| `metal/f32/sqrt/offset/r33_c129` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r33_c129.json) | 0.2926 / 0.2396 | [passed](program_peer/metal__f32__sqrt__offset__r33_c129.json) | 0.2978 / 0.2432 |
| `metal/f32/sqrt/offset/r128_c5` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r128_c5.json) | 0.2866 / 0.1882 | [passed](program_peer/metal__f32__sqrt__offset__r128_c5.json) | 0.2894 / 0.2314 |
| `metal/f32/sqrt/offset/r128_c32` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r128_c32.json) | 0.3164 / 0.2393 | [passed](program_peer/metal__f32__sqrt__offset__r128_c32.json) | 0.3426 / 0.242 |
| `metal/f32/sqrt/offset/r128_c65` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r128_c65.json) | 0.386 / 0.2785 | [passed](program_peer/metal__f32__sqrt__offset__r128_c65.json) | 0.3878 / 0.3116 |
| `metal/f32/sqrt/offset/r128_c129` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r128_c129.json) | 0.4516 / 0.4171 | [passed](program_peer/metal__f32__sqrt__offset__r128_c129.json) | 0.461 / 0.4252 |
| `metal/bf16/exp/strided/r3_c5` | correct | [passed](operator_peer/metal__bf16__exp__strided__r3_c5.json) | 0.2212 / 0.2004 | [passed](program_peer/metal__bf16__exp__strided__r3_c5.json) | 0.2644 / 0.2314 |
| `metal/bf16/exp/strided/r3_c32` | correct | [passed](operator_peer/metal__bf16__exp__strided__r3_c32.json) | 0.2454 / 0.1888 | [passed](program_peer/metal__bf16__exp__strided__r3_c32.json) | 0.2454 / 0.1993 |
| `metal/bf16/exp/strided/r3_c65` | correct | [passed](operator_peer/metal__bf16__exp__strided__r3_c65.json) | 0.2512 / 0.1951 | [passed](program_peer/metal__bf16__exp__strided__r3_c65.json) | 0.2686 / 0.1868 |
| `metal/bf16/exp/strided/r3_c129` | correct | [passed](operator_peer/metal__bf16__exp__strided__r3_c129.json) | 0.2526 / 0.197 | [passed](program_peer/metal__bf16__exp__strided__r3_c129.json) | 0.2548 / 0.1903 |
| `metal/bf16/exp/strided/r16_c5` | correct | [passed](operator_peer/metal__bf16__exp__strided__r16_c5.json) | 0.246 / 0.1912 | [passed](program_peer/metal__bf16__exp__strided__r16_c5.json) | 0.2838 / 0.2033 |
| `metal/bf16/exp/strided/r16_c32` | correct | [passed](operator_peer/metal__bf16__exp__strided__r16_c32.json) | 0.2562 / 0.2077 | [passed](program_peer/metal__bf16__exp__strided__r16_c32.json) | 0.2726 / 0.1989 |
| `metal/bf16/exp/strided/r16_c65` | correct | [passed](operator_peer/metal__bf16__exp__strided__r16_c65.json) | 0.2668 / 0.205 | [passed](program_peer/metal__bf16__exp__strided__r16_c65.json) | 0.3104 / 0.205 |
| `metal/bf16/exp/strided/r16_c129` | correct | [passed](operator_peer/metal__bf16__exp__strided__r16_c129.json) | 0.2792 / 0.2134 | [passed](program_peer/metal__bf16__exp__strided__r16_c129.json) | 0.313 / 0.2161 |
| `metal/bf16/exp/strided/r33_c5` | correct | [passed](operator_peer/metal__bf16__exp__strided__r33_c5.json) | 0.241 / 0.1927 | [passed](program_peer/metal__bf16__exp__strided__r33_c5.json) | 0.2452 / 0.1925 |
| `metal/bf16/exp/strided/r33_c32` | correct | [passed](operator_peer/metal__bf16__exp__strided__r33_c32.json) | 0.2376 / 0.2037 | [passed](program_peer/metal__bf16__exp__strided__r33_c32.json) | 0.2632 / 0.2034 |
| `metal/bf16/exp/strided/r33_c65` | correct | [passed](operator_peer/metal__bf16__exp__strided__r33_c65.json) | 0.2894 / 0.2137 | [passed](program_peer/metal__bf16__exp__strided__r33_c65.json) | 0.3276 / 0.2224 |
| `metal/bf16/exp/strided/r33_c129` | correct | [passed](operator_peer/metal__bf16__exp__strided__r33_c129.json) | 0.3016 / 0.2356 | [passed](program_peer/metal__bf16__exp__strided__r33_c129.json) | 0.3502 / 0.2423 |
| `metal/bf16/exp/strided/r128_c5` | correct | [passed](operator_peer/metal__bf16__exp__strided__r128_c5.json) | 0.2916 / 0.2923 | [passed](program_peer/metal__bf16__exp__strided__r128_c5.json) | 0.275 / 0.2049 |
| `metal/bf16/exp/strided/r128_c32` | correct | [passed](operator_peer/metal__bf16__exp__strided__r128_c32.json) | 0.345 / 0.2414 | [passed](program_peer/metal__bf16__exp__strided__r128_c32.json) | 0.381 / 0.2431 |
| `metal/bf16/exp/strided/r128_c65` | correct | [passed](operator_peer/metal__bf16__exp__strided__r128_c65.json) | 0.3892 / 0.3153 | [passed](program_peer/metal__bf16__exp__strided__r128_c65.json) | 0.4206 / 0.3388 |
| `metal/bf16/exp/strided/r128_c129` | correct | [passed](operator_peer/metal__bf16__exp__strided__r128_c129.json) | 0.5018 / 0.3994 | [passed](program_peer/metal__bf16__exp__strided__r128_c129.json) | 0.518 / 0.4221 |
| `metal/bf16/exp/offset/r3_c5` | correct | [passed](operator_peer/metal__bf16__exp__offset__r3_c5.json) | 0.2468 / 0.1892 | [passed](program_peer/metal__bf16__exp__offset__r3_c5.json) | 0.2606 / 0.2195 |
| `metal/bf16/exp/offset/r3_c32` | correct | [passed](operator_peer/metal__bf16__exp__offset__r3_c32.json) | 0.2476 / 0.1965 | [passed](program_peer/metal__bf16__exp__offset__r3_c32.json) | 0.2564 / 0.1902 |
| `metal/bf16/exp/offset/r3_c65` | correct | [passed](operator_peer/metal__bf16__exp__offset__r3_c65.json) | 0.2482 / 0.2002 | [passed](program_peer/metal__bf16__exp__offset__r3_c65.json) | 0.2638 / 0.1947 |
| `metal/bf16/exp/offset/r3_c129` | correct | [passed](operator_peer/metal__bf16__exp__offset__r3_c129.json) | 0.2506 / 0.1959 | [passed](program_peer/metal__bf16__exp__offset__r3_c129.json) | 0.2634 / 0.2018 |
| `metal/bf16/exp/offset/r16_c5` | correct | [passed](operator_peer/metal__bf16__exp__offset__r16_c5.json) | 0.235 / 0.1814 | [passed](program_peer/metal__bf16__exp__offset__r16_c5.json) | 0.2638 / 0.1952 |
| `metal/bf16/exp/offset/r16_c32` | correct | [passed](operator_peer/metal__bf16__exp__offset__r16_c32.json) | 0.258 / 0.1948 | [passed](program_peer/metal__bf16__exp__offset__r16_c32.json) | 0.2712 / 0.2064 |
| `metal/bf16/exp/offset/r16_c65` | correct | [passed](operator_peer/metal__bf16__exp__offset__r16_c65.json) | 0.2244 / 0.212 | [passed](program_peer/metal__bf16__exp__offset__r16_c65.json) | 0.251 / 0.2022 |
| `metal/bf16/exp/offset/r16_c129` | correct | [passed](operator_peer/metal__bf16__exp__offset__r16_c129.json) | 0.2892 / 0.208 | [passed](program_peer/metal__bf16__exp__offset__r16_c129.json) | 0.309 / 0.219 |
| `metal/bf16/exp/offset/r33_c5` | correct | [passed](operator_peer/metal__bf16__exp__offset__r33_c5.json) | 0.2584 / 0.1866 | [passed](program_peer/metal__bf16__exp__offset__r33_c5.json) | 0.27 / 0.1979 |
| `metal/bf16/exp/offset/r33_c32` | correct | [passed](operator_peer/metal__bf16__exp__offset__r33_c32.json) | 0.2804 / 0.2039 | [passed](program_peer/metal__bf16__exp__offset__r33_c32.json) | 0.2868 / 0.2117 |
| `metal/bf16/exp/offset/r33_c65` | correct | [passed](operator_peer/metal__bf16__exp__offset__r33_c65.json) | 0.2608 / 0.2076 | [passed](program_peer/metal__bf16__exp__offset__r33_c65.json) | 0.2752 / 0.2261 |
| `metal/bf16/exp/offset/r33_c129` | correct | [passed](operator_peer/metal__bf16__exp__offset__r33_c129.json) | 0.3018 / 0.2384 | [passed](program_peer/metal__bf16__exp__offset__r33_c129.json) | 0.3342 / 0.24 |
| `metal/bf16/exp/offset/r128_c5` | correct | [passed](operator_peer/metal__bf16__exp__offset__r128_c5.json) | 0.2942 / 0.1986 | [passed](program_peer/metal__bf16__exp__offset__r128_c5.json) | 0.316 / 0.1975 |
| `metal/bf16/exp/offset/r128_c32` | correct | [passed](operator_peer/metal__bf16__exp__offset__r128_c32.json) | 0.3428 / 0.2419 | [passed](program_peer/metal__bf16__exp__offset__r128_c32.json) | 0.38 / 0.2415 |
| `metal/bf16/exp/offset/r128_c65` | correct | [passed](operator_peer/metal__bf16__exp__offset__r128_c65.json) | 0.3678 / 0.3107 | [passed](program_peer/metal__bf16__exp__offset__r128_c65.json) | 0.3664 / 0.3061 |
| `metal/bf16/exp/offset/r128_c129` | correct | [passed](operator_peer/metal__bf16__exp__offset__r128_c129.json) | 0.4554 / 0.42 | [passed](program_peer/metal__bf16__exp__offset__r128_c129.json) | 0.5064 / 0.4333 |
| `metal/bf16/log/strided/r3_c5` | correct | [passed](operator_peer/metal__bf16__log__strided__r3_c5.json) | 0.2462 / 0.195 | [passed](program_peer/metal__bf16__log__strided__r3_c5.json) | 0.2626 / 0.2107 |
| `metal/bf16/log/strided/r3_c32` | correct | [passed](operator_peer/metal__bf16__log__strided__r3_c32.json) | 0.2474 / 0.1946 | [passed](program_peer/metal__bf16__log__strided__r3_c32.json) | 0.225 / 0.1936 |
| `metal/bf16/log/strided/r3_c65` | correct | [passed](operator_peer/metal__bf16__log__strided__r3_c65.json) | 0.2468 / 0.2036 | [passed](program_peer/metal__bf16__log__strided__r3_c65.json) | 0.2776 / 0.194 |
| `metal/bf16/log/strided/r3_c129` | correct | [passed](operator_peer/metal__bf16__log__strided__r3_c129.json) | 0.2222 / 0.2044 | [passed](program_peer/metal__bf16__log__strided__r3_c129.json) | 0.2714 / 0.2058 |
| `metal/bf16/log/strided/r16_c5` | correct | [passed](operator_peer/metal__bf16__log__strided__r16_c5.json) | 0.2558 / 0.202 | [passed](program_peer/metal__bf16__log__strided__r16_c5.json) | 0.2226 / 0.1926 |
| `metal/bf16/log/strided/r16_c32` | correct | [passed](operator_peer/metal__bf16__log__strided__r16_c32.json) | 0.2556 / 0.2049 | [passed](program_peer/metal__bf16__log__strided__r16_c32.json) | 0.2724 / 0.1915 |
| `metal/bf16/log/strided/r16_c65` | correct | [passed](operator_peer/metal__bf16__log__strided__r16_c65.json) | 0.2114 / 0.206 | [passed](program_peer/metal__bf16__log__strided__r16_c65.json) | 0.3076 / 0.2043 |
| `metal/bf16/log/strided/r16_c129` | correct | [passed](operator_peer/metal__bf16__log__strided__r16_c129.json) | 0.2702 / 0.249 | [passed](program_peer/metal__bf16__log__strided__r16_c129.json) | 0.3102 / 0.2193 |
| `metal/bf16/log/strided/r33_c5` | correct | [passed](operator_peer/metal__bf16__log__strided__r33_c5.json) | 0.2152 / 0.192 | [passed](program_peer/metal__bf16__log__strided__r33_c5.json) | 0.3012 / 0.1971 |
| `metal/bf16/log/strided/r33_c32` | correct | [passed](operator_peer/metal__bf16__log__strided__r33_c32.json) | 0.2798 / 0.2006 | [passed](program_peer/metal__bf16__log__strided__r33_c32.json) | 0.2998 / 0.2048 |
| `metal/bf16/log/strided/r33_c65` | correct | [passed](operator_peer/metal__bf16__log__strided__r33_c65.json) | 0.288 / 0.1975 | [passed](program_peer/metal__bf16__log__strided__r33_c65.json) | 0.3274 / 0.2247 |
| `metal/bf16/log/strided/r33_c129` | correct | [passed](operator_peer/metal__bf16__log__strided__r33_c129.json) | 0.3006 / 0.2418 | [passed](program_peer/metal__bf16__log__strided__r33_c129.json) | 0.3692 / 0.2433 |
| `metal/bf16/log/strided/r128_c5` | correct | [passed](operator_peer/metal__bf16__log__strided__r128_c5.json) | 0.2912 / 0.2081 | [passed](program_peer/metal__bf16__log__strided__r128_c5.json) | 0.316 / 0.2077 |
| `metal/bf16/log/strided/r128_c32` | correct | [passed](operator_peer/metal__bf16__log__strided__r128_c32.json) | 0.3382 / 0.2372 | [passed](program_peer/metal__bf16__log__strided__r128_c32.json) | 0.3826 / 0.2417 |
| `metal/bf16/log/strided/r128_c65` | correct | [passed](operator_peer/metal__bf16__log__strided__r128_c65.json) | 0.361 / 0.3413 | [passed](program_peer/metal__bf16__log__strided__r128_c65.json) | 0.3876 / 0.3233 |
| `metal/bf16/log/strided/r128_c129` | correct | [passed](operator_peer/metal__bf16__log__strided__r128_c129.json) | 0.4454 / 0.4134 | [passed](program_peer/metal__bf16__log__strided__r128_c129.json) | 0.511 / 0.4202 |
| `metal/bf16/log/offset/r3_c5` | correct | [passed](operator_peer/metal__bf16__log__offset__r3_c5.json) | 0.2606 / 0.1966 | [passed](program_peer/metal__bf16__log__offset__r3_c5.json) | 0.2836 / 0.1933 |
| `metal/bf16/log/offset/r3_c32` | correct | [passed](operator_peer/metal__bf16__log__offset__r3_c32.json) | 0.2526 / 0.2011 | [passed](program_peer/metal__bf16__log__offset__r3_c32.json) | 0.2332 / 0.1967 |
| `metal/bf16/log/offset/r3_c65` | correct | [passed](operator_peer/metal__bf16__log__offset__r3_c65.json) | 0.2036 / 0.1989 | [passed](program_peer/metal__bf16__log__offset__r3_c65.json) | 0.2272 / 0.1834 |
| `metal/bf16/log/offset/r3_c129` | correct | [passed](operator_peer/metal__bf16__log__offset__r3_c129.json) | 0.253 / 0.1961 | [passed](program_peer/metal__bf16__log__offset__r3_c129.json) | 0.2628 / 0.1887 |
| `metal/bf16/log/offset/r16_c5` | correct | [passed](operator_peer/metal__bf16__log__offset__r16_c5.json) | 0.2506 / 0.2139 | [passed](program_peer/metal__bf16__log__offset__r16_c5.json) | 0.2338 / 0.1953 |
| `metal/bf16/log/offset/r16_c32` | correct | [passed](operator_peer/metal__bf16__log__offset__r16_c32.json) | 0.2566 / 0.1968 | [passed](program_peer/metal__bf16__log__offset__r16_c32.json) | 0.2708 / 0.2035 |
| `metal/bf16/log/offset/r16_c65` | correct | [passed](operator_peer/metal__bf16__log__offset__r16_c65.json) | 0.261 / 0.2168 | [passed](program_peer/metal__bf16__log__offset__r16_c65.json) | 0.2718 / 0.2252 |
| `metal/bf16/log/offset/r16_c129` | correct | [passed](operator_peer/metal__bf16__log__offset__r16_c129.json) | 0.2766 / 0.2178 | [passed](program_peer/metal__bf16__log__offset__r16_c129.json) | 0.265 / 0.2215 |
| `metal/bf16/log/offset/r33_c5` | correct | [passed](operator_peer/metal__bf16__log__offset__r33_c5.json) | 0.223 / 0.2045 | [passed](program_peer/metal__bf16__log__offset__r33_c5.json) | 0.2704 / 0.1946 |
| `metal/bf16/log/offset/r33_c32` | correct | [passed](operator_peer/metal__bf16__log__offset__r33_c32.json) | 0.2754 / 0.2302 | [passed](program_peer/metal__bf16__log__offset__r33_c32.json) | 0.2996 / 0.2069 |
| `metal/bf16/log/offset/r33_c65` | correct | [passed](operator_peer/metal__bf16__log__offset__r33_c65.json) | 0.288 / 0.2077 | [passed](program_peer/metal__bf16__log__offset__r33_c65.json) | 0.3262 / 0.2235 |
| `metal/bf16/log/offset/r33_c129` | correct | [passed](operator_peer/metal__bf16__log__offset__r33_c129.json) | 0.3004 / 0.2403 | [passed](program_peer/metal__bf16__log__offset__r33_c129.json) | 0.3468 / 0.2422 |
| `metal/bf16/log/offset/r128_c5` | correct | [passed](operator_peer/metal__bf16__log__offset__r128_c5.json) | 0.297 / 0.2069 | [passed](program_peer/metal__bf16__log__offset__r128_c5.json) | 0.3244 / 0.2006 |
| `metal/bf16/log/offset/r128_c32` | correct | [passed](operator_peer/metal__bf16__log__offset__r128_c32.json) | 0.3386 / 0.24 | [passed](program_peer/metal__bf16__log__offset__r128_c32.json) | 0.3728 / 0.2395 |
| `metal/bf16/log/offset/r128_c65` | correct | [passed](operator_peer/metal__bf16__log__offset__r128_c65.json) | 0.3824 / 0.3372 | [passed](program_peer/metal__bf16__log__offset__r128_c65.json) | 0.4146 / 0.2879 |
| `metal/bf16/log/offset/r128_c129` | correct | [passed](operator_peer/metal__bf16__log__offset__r128_c129.json) | 0.4362 / 0.4174 | [passed](program_peer/metal__bf16__log__offset__r128_c129.json) | 0.5248 / 0.4315 |
| `metal/bf16/sqrt/strided/r3_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r3_c5.json) | 0.2472 / 0.2016 | [passed](program_peer/metal__bf16__sqrt__strided__r3_c5.json) | 0.2634 / 0.1954 |
| `metal/bf16/sqrt/strided/r3_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r3_c32.json) | 0.2496 / 0.2032 | [passed](program_peer/metal__bf16__sqrt__strided__r3_c32.json) | 0.2578 / 0.1971 |
| `metal/bf16/sqrt/strided/r3_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r3_c65.json) | 0.2508 / 0.2024 | [passed](program_peer/metal__bf16__sqrt__strided__r3_c65.json) | 0.2848 / 0.1814 |
| `metal/bf16/sqrt/strided/r3_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r3_c129.json) | 0.235 / 0.1939 | [passed](program_peer/metal__bf16__sqrt__strided__r3_c129.json) | 0.261 / 0.1756 |
| `metal/bf16/sqrt/strided/r16_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r16_c5.json) | 0.2522 / 0.1957 | [passed](program_peer/metal__bf16__sqrt__strided__r16_c5.json) | 0.2482 / 0.1905 |
| `metal/bf16/sqrt/strided/r16_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r16_c32.json) | 0.2598 / 0.2061 | [passed](program_peer/metal__bf16__sqrt__strided__r16_c32.json) | 0.2376 / 0.1972 |
| `metal/bf16/sqrt/strided/r16_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r16_c65.json) | 0.242 / 0.205 | [passed](program_peer/metal__bf16__sqrt__strided__r16_c65.json) | 0.3112 / 0.1957 |
| `metal/bf16/sqrt/strided/r16_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r16_c129.json) | 0.2694 / 0.2164 | [passed](program_peer/metal__bf16__sqrt__strided__r16_c129.json) | 0.3124 / 0.2212 |
| `metal/bf16/sqrt/strided/r33_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r33_c5.json) | 0.2526 / 0.1986 | [passed](program_peer/metal__bf16__sqrt__strided__r33_c5.json) | 0.2424 / 0.2107 |
| `metal/bf16/sqrt/strided/r33_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r33_c32.json) | 0.2778 / 0.201 | [passed](program_peer/metal__bf16__sqrt__strided__r33_c32.json) | 0.3232 / 0.2076 |
| `metal/bf16/sqrt/strided/r33_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r33_c65.json) | 0.2782 / 0.2216 | [passed](program_peer/metal__bf16__sqrt__strided__r33_c65.json) | 0.316 / 0.2198 |
| `metal/bf16/sqrt/strided/r33_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r33_c129.json) | 0.308 / 0.2387 | [passed](program_peer/metal__bf16__sqrt__strided__r33_c129.json) | 0.34 / 0.2406 |
| `metal/bf16/sqrt/strided/r128_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r128_c5.json) | 0.2878 / 0.2074 | [passed](program_peer/metal__bf16__sqrt__strided__r128_c5.json) | 0.3328 / 0.2086 |
| `metal/bf16/sqrt/strided/r128_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r128_c32.json) | 0.3408 / 0.239 | [passed](program_peer/metal__bf16__sqrt__strided__r128_c32.json) | 0.3624 / 0.2414 |
| `metal/bf16/sqrt/strided/r128_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r128_c65.json) | 0.3874 / 0.2842 | [passed](program_peer/metal__bf16__sqrt__strided__r128_c65.json) | 0.4214 / 0.3399 |
| `metal/bf16/sqrt/strided/r128_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r128_c129.json) | 0.4452 / 0.4162 | [passed](program_peer/metal__bf16__sqrt__strided__r128_c129.json) | 0.5188 / 0.4284 |
| `metal/bf16/sqrt/offset/r3_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r3_c5.json) | 0.2468 / 0.1892 | [passed](program_peer/metal__bf16__sqrt__offset__r3_c5.json) | 0.2674 / 0.2001 |
| `metal/bf16/sqrt/offset/r3_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r3_c32.json) | 0.2522 / 0.1944 | [passed](program_peer/metal__bf16__sqrt__offset__r3_c32.json) | 0.2332 / 0.1965 |
| `metal/bf16/sqrt/offset/r3_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r3_c65.json) | 0.2458 / 0.1913 | [passed](program_peer/metal__bf16__sqrt__offset__r3_c65.json) | 0.2638 / 0.1789 |
| `metal/bf16/sqrt/offset/r3_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r3_c129.json) | 0.2206 / 0.1869 | [passed](program_peer/metal__bf16__sqrt__offset__r3_c129.json) | 0.263 / 0.1996 |
| `metal/bf16/sqrt/offset/r16_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r16_c5.json) | 0.254 / 0.2026 | [passed](program_peer/metal__bf16__sqrt__offset__r16_c5.json) | 0.2642 / 0.2038 |
| `metal/bf16/sqrt/offset/r16_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r16_c32.json) | 0.2554 / 0.2056 | [passed](program_peer/metal__bf16__sqrt__offset__r16_c32.json) | 0.2726 / 0.2081 |
| `metal/bf16/sqrt/offset/r16_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r16_c65.json) | 0.2696 / 0.2038 | [passed](program_peer/metal__bf16__sqrt__offset__r16_c65.json) | 0.26 / 0.2051 |
| `metal/bf16/sqrt/offset/r16_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r16_c129.json) | 0.269 / 0.2208 | [passed](program_peer/metal__bf16__sqrt__offset__r16_c129.json) | 0.2612 / 0.2226 |
| `metal/bf16/sqrt/offset/r33_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r33_c5.json) | 0.2612 / 0.2068 | [passed](program_peer/metal__bf16__sqrt__offset__r33_c5.json) | 0.2778 / 0.1897 |
| `metal/bf16/sqrt/offset/r33_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r33_c32.json) | 0.2778 / 0.2076 | [passed](program_peer/metal__bf16__sqrt__offset__r33_c32.json) | 0.324 / 0.2086 |
| `metal/bf16/sqrt/offset/r33_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r33_c65.json) | 0.286 / 0.2284 | [passed](program_peer/metal__bf16__sqrt__offset__r33_c65.json) | 0.2578 / 0.2157 |
| `metal/bf16/sqrt/offset/r33_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r33_c129.json) | 0.2584 / 0.2383 | [passed](program_peer/metal__bf16__sqrt__offset__r33_c129.json) | 0.3052 / 0.2411 |
| `metal/bf16/sqrt/offset/r128_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r128_c5.json) | 0.2942 / 0.2059 | [passed](program_peer/metal__bf16__sqrt__offset__r128_c5.json) | 0.308 / 0.2064 |
| `metal/bf16/sqrt/offset/r128_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r128_c32.json) | 0.348 / 0.2397 | [passed](program_peer/metal__bf16__sqrt__offset__r128_c32.json) | 0.3768 / 0.2426 |
| `metal/bf16/sqrt/offset/r128_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r128_c65.json) | 0.3764 / 0.2786 | [passed](program_peer/metal__bf16__sqrt__offset__r128_c65.json) | 0.421 / 0.3391 |
| `metal/bf16/sqrt/offset/r128_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r128_c129.json) | 0.461 / 0.4154 | [passed](program_peer/metal__bf16__sqrt__offset__r128_c129.json) | 0.5226 / 0.4165 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
