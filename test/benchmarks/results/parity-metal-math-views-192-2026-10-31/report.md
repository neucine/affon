# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7f07fe6aa842ce0791d10fa8bf8c43bab9f74a0117f0c62fa05022a2386cbde7`  
Binary SHA-256: `af1b758d59cd4f42a812cf3dcaae67d3b0f01abca19878e39bc7300e1ae67e35`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 192
- Numerical capability passed: 192
- Operator parity and timing passed: 187
- Program parity and timing passed: 187
- Probe failures or incorrect: 0
- Operator incorrect or error: 5
- Program incorrect or error: 5

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/exp/strided/r3_c5` | correct | [passed](operator_peer/metal__f32__exp__strided__r3_c5.json) | 0.263 / 0.2105 | [passed](program_peer/metal__f32__exp__strided__r3_c5.json) | 0.2508 / 0.1916 |
| `metal/f32/exp/strided/r3_c32` | correct | [passed](operator_peer/metal__f32__exp__strided__r3_c32.json) | 0.2398 / 0.1998 | [passed](program_peer/metal__f32__exp__strided__r3_c32.json) | 0.2504 / 0.1937 |
| `metal/f32/exp/strided/r3_c65` | correct | [passed](operator_peer/metal__f32__exp__strided__r3_c65.json) | 0.2402 / 0.197 | [passed](program_peer/metal__f32__exp__strided__r3_c65.json) | 0.2832 / 0.1883 |
| `metal/f32/exp/strided/r3_c129` | correct | [incorrect](operator_peer_failures/metal__f32__exp__strided__r3_c129.json) | 0.2006 / 0.1934 | [incorrect](program_peer_failures/metal__f32__exp__strided__r3_c129.json) | 0.2582 / 0.1976 |
| `metal/f32/exp/strided/r16_c5` | correct | [passed](operator_peer/metal__f32__exp__strided__r16_c5.json) | 0.2438 / 0.1976 | [passed](program_peer/metal__f32__exp__strided__r16_c5.json) | 0.2598 / 0.2346 |
| `metal/f32/exp/strided/r16_c32` | correct | [passed](operator_peer/metal__f32__exp__strided__r16_c32.json) | 0.253 / 0.2098 | [passed](program_peer/metal__f32__exp__strided__r16_c32.json) | 0.2228 / 0.208 |
| `metal/f32/exp/strided/r16_c65` | correct | [passed](operator_peer/metal__f32__exp__strided__r16_c65.json) | 0.2664 / 0.2022 | [passed](program_peer/metal__f32__exp__strided__r16_c65.json) | 0.2718 / 0.2132 |
| `metal/f32/exp/strided/r16_c129` | correct | [incorrect](operator_peer_failures/metal__f32__exp__strided__r16_c129.json) | 0.278 / 0.2179 | [incorrect](program_peer_failures/metal__f32__exp__strided__r16_c129.json) | 0.283 / 0.2231 |
| `metal/f32/exp/strided/r33_c5` | correct | [passed](operator_peer/metal__f32__exp__strided__r33_c5.json) | 0.2514 / 0.1947 | [passed](program_peer/metal__f32__exp__strided__r33_c5.json) | 0.261 / 0.2052 |
| `metal/f32/exp/strided/r33_c32` | correct | [passed](operator_peer/metal__f32__exp__strided__r33_c32.json) | 0.279 / 0.1985 | [passed](program_peer/metal__f32__exp__strided__r33_c32.json) | 0.276 / 0.2146 |
| `metal/f32/exp/strided/r33_c65` | correct | [passed](operator_peer/metal__f32__exp__strided__r33_c65.json) | 0.288 / 0.2173 | [passed](program_peer/metal__f32__exp__strided__r33_c65.json) | 0.2962 / 0.222 |
| `metal/f32/exp/strided/r33_c129` | correct | [incorrect](operator_peer_failures/metal__f32__exp__strided__r33_c129.json) | 0.3034 / 0.2407 | [incorrect](program_peer_failures/metal__f32__exp__strided__r33_c129.json) | 0.3118 / 0.2421 |
| `metal/f32/exp/strided/r128_c5` | correct | [passed](operator_peer/metal__f32__exp__strided__r128_c5.json) | 0.2908 / 0.1965 | [passed](program_peer/metal__f32__exp__strided__r128_c5.json) | 0.3 / 0.2083 |
| `metal/f32/exp/strided/r128_c32` | correct | [incorrect](operator_peer_failures/metal__f32__exp__strided__r128_c32.json) | 0.331 / 0.2784 | [incorrect](program_peer_failures/metal__f32__exp__strided__r128_c32.json) | 0.3514 / 0.2419 |
| `metal/f32/exp/strided/r128_c65` | correct | [passed](operator_peer/metal__f32__exp__strided__r128_c65.json) | 0.3798 / 0.3312 | [passed](program_peer/metal__f32__exp__strided__r128_c65.json) | 0.387 / 0.2827 |
| `metal/f32/exp/strided/r128_c129` | correct | [passed](operator_peer/metal__f32__exp__strided__r128_c129.json) | 0.453 / 0.4168 | [passed](program_peer/metal__f32__exp__strided__r128_c129.json) | 0.46 / 0.4378 |
| `metal/f32/exp/offset/r3_c5` | correct | [passed](operator_peer/metal__f32__exp__offset__r3_c5.json) | 0.225 / 0.2221 | [passed](program_peer/metal__f32__exp__offset__r3_c5.json) | 0.2506 / 0.1976 |
| `metal/f32/exp/offset/r3_c32` | correct | [passed](operator_peer/metal__f32__exp__offset__r3_c32.json) | 0.194 / 0.1981 | [passed](program_peer/metal__f32__exp__offset__r3_c32.json) | 0.2468 / 0.2297 |
| `metal/f32/exp/offset/r3_c65` | correct | [passed](operator_peer/metal__f32__exp__offset__r3_c65.json) | 0.1934 / 0.1831 | [passed](program_peer/metal__f32__exp__offset__r3_c65.json) | 0.2496 / 0.1883 |
| `metal/f32/exp/offset/r3_c129` | correct | [passed](operator_peer/metal__f32__exp__offset__r3_c129.json) | 0.2426 / 0.2113 | [passed](program_peer/metal__f32__exp__offset__r3_c129.json) | 0.2438 / 0.187 |
| `metal/f32/exp/offset/r16_c5` | correct | [passed](operator_peer/metal__f32__exp__offset__r16_c5.json) | 0.2552 / 0.202 | [passed](program_peer/metal__f32__exp__offset__r16_c5.json) | 0.2592 / 0.1865 |
| `metal/f32/exp/offset/r16_c32` | correct | [passed](operator_peer/metal__f32__exp__offset__r16_c32.json) | 0.2538 / 0.195 | [passed](program_peer/metal__f32__exp__offset__r16_c32.json) | 0.2606 / 0.1975 |
| `metal/f32/exp/offset/r16_c65` | correct | [passed](operator_peer/metal__f32__exp__offset__r16_c65.json) | 0.2664 / 0.1998 | [passed](program_peer/metal__f32__exp__offset__r16_c65.json) | 0.2698 / 0.1982 |
| `metal/f32/exp/offset/r16_c129` | correct | [passed](operator_peer/metal__f32__exp__offset__r16_c129.json) | 0.2546 / 0.22 | [passed](program_peer/metal__f32__exp__offset__r16_c129.json) | 0.2872 / 0.2213 |
| `metal/f32/exp/offset/r33_c5` | correct | [passed](operator_peer/metal__f32__exp__offset__r33_c5.json) | 0.2436 / 0.1978 | [passed](program_peer/metal__f32__exp__offset__r33_c5.json) | 0.2412 / 0.2045 |
| `metal/f32/exp/offset/r33_c32` | correct | [passed](operator_peer/metal__f32__exp__offset__r33_c32.json) | 0.2748 / 0.2414 | [passed](program_peer/metal__f32__exp__offset__r33_c32.json) | 0.2808 / 0.2128 |
| `metal/f32/exp/offset/r33_c65` | correct | [passed](operator_peer/metal__f32__exp__offset__r33_c65.json) | 0.2868 / 0.2211 | [passed](program_peer/metal__f32__exp__offset__r33_c65.json) | 0.2924 / 0.2243 |
| `metal/f32/exp/offset/r33_c129` | correct | [passed](operator_peer/metal__f32__exp__offset__r33_c129.json) | 0.3014 / 0.2421 | [passed](program_peer/metal__f32__exp__offset__r33_c129.json) | 0.3086 / 0.2455 |
| `metal/f32/exp/offset/r128_c5` | correct | [passed](operator_peer/metal__f32__exp__offset__r128_c5.json) | 0.2924 / 0.207 | [passed](program_peer/metal__f32__exp__offset__r128_c5.json) | 0.2966 / 0.2091 |
| `metal/f32/exp/offset/r128_c32` | correct | [passed](operator_peer/metal__f32__exp__offset__r128_c32.json) | 0.3422 / 0.2398 | [passed](program_peer/metal__f32__exp__offset__r128_c32.json) | 0.348 / 0.2428 |
| `metal/f32/exp/offset/r128_c65` | correct | [incorrect](operator_peer_failures/metal__f32__exp__offset__r128_c65.json) | 0.3828 / 0.3338 | [incorrect](program_peer_failures/metal__f32__exp__offset__r128_c65.json) | 0.3924 / 0.2815 |
| `metal/f32/exp/offset/r128_c129` | correct | [passed](operator_peer/metal__f32__exp__offset__r128_c129.json) | 0.4388 / 0.416 | [passed](program_peer/metal__f32__exp__offset__r128_c129.json) | 0.4482 / 0.4209 |
| `metal/f32/log/strided/r3_c5` | correct | [passed](operator_peer/metal__f32__log__strided__r3_c5.json) | 0.2444 / 0.1967 | [passed](program_peer/metal__f32__log__strided__r3_c5.json) | 0.2498 / 0.2004 |
| `metal/f32/log/strided/r3_c32` | correct | [passed](operator_peer/metal__f32__log__strided__r3_c32.json) | 0.2386 / 0.1878 | [passed](program_peer/metal__f32__log__strided__r3_c32.json) | 0.2466 / 0.2003 |
| `metal/f32/log/strided/r3_c65` | correct | [passed](operator_peer/metal__f32__log__strided__r3_c65.json) | 0.2338 / 0.1979 | [passed](program_peer/metal__f32__log__strided__r3_c65.json) | 0.252 / 0.188 |
| `metal/f32/log/strided/r3_c129` | correct | [passed](operator_peer/metal__f32__log__strided__r3_c129.json) | 0.2484 / 0.2187 | [passed](program_peer/metal__f32__log__strided__r3_c129.json) | 0.2504 / 0.2199 |
| `metal/f32/log/strided/r16_c5` | correct | [passed](operator_peer/metal__f32__log__strided__r16_c5.json) | 0.2454 / 0.1967 | [passed](program_peer/metal__f32__log__strided__r16_c5.json) | 0.2538 / 0.2178 |
| `metal/f32/log/strided/r16_c32` | correct | [passed](operator_peer/metal__f32__log__strided__r16_c32.json) | 0.2094 / 0.1975 | [passed](program_peer/metal__f32__log__strided__r16_c32.json) | 0.2592 / 0.1965 |
| `metal/f32/log/strided/r16_c65` | correct | [passed](operator_peer/metal__f32__log__strided__r16_c65.json) | 0.2684 / 0.2038 | [passed](program_peer/metal__f32__log__strided__r16_c65.json) | 0.2716 / 0.2023 |
| `metal/f32/log/strided/r16_c129` | correct | [passed](operator_peer/metal__f32__log__strided__r16_c129.json) | 0.2822 / 0.2203 | [passed](program_peer/metal__f32__log__strided__r16_c129.json) | 0.2838 / 0.2222 |
| `metal/f32/log/strided/r33_c5` | correct | [passed](operator_peer/metal__f32__log__strided__r33_c5.json) | 0.2544 / 0.2005 | [passed](program_peer/metal__f32__log__strided__r33_c5.json) | 0.2632 / 0.1941 |
| `metal/f32/log/strided/r33_c32` | correct | [passed](operator_peer/metal__f32__log__strided__r33_c32.json) | 0.2756 / 0.2053 | [passed](program_peer/metal__f32__log__strided__r33_c32.json) | 0.2814 / 0.2141 |
| `metal/f32/log/strided/r33_c65` | correct | [passed](operator_peer/metal__f32__log__strided__r33_c65.json) | 0.291 / 0.2209 | [passed](program_peer/metal__f32__log__strided__r33_c65.json) | 0.2912 / 0.2246 |
| `metal/f32/log/strided/r33_c129` | correct | [passed](operator_peer/metal__f32__log__strided__r33_c129.json) | 0.3066 / 0.2389 | [passed](program_peer/metal__f32__log__strided__r33_c129.json) | 0.3106 / 0.2401 |
| `metal/f32/log/strided/r128_c5` | correct | [passed](operator_peer/metal__f32__log__strided__r128_c5.json) | 0.2904 / 0.2352 | [passed](program_peer/metal__f32__log__strided__r128_c5.json) | 0.2946 / 0.2045 |
| `metal/f32/log/strided/r128_c32` | correct | [passed](operator_peer/metal__f32__log__strided__r128_c32.json) | 0.341 / 0.2419 | [passed](program_peer/metal__f32__log__strided__r128_c32.json) | 0.347 / 0.2411 |
| `metal/f32/log/strided/r128_c65` | correct | [passed](operator_peer/metal__f32__log__strided__r128_c65.json) | 0.371 / 0.3115 | [passed](program_peer/metal__f32__log__strided__r128_c65.json) | 0.3862 / 0.3365 |
| `metal/f32/log/strided/r128_c129` | correct | [passed](operator_peer/metal__f32__log__strided__r128_c129.json) | 0.4484 / 0.3863 | [passed](program_peer/metal__f32__log__strided__r128_c129.json) | 0.4662 / 0.425 |
| `metal/f32/log/offset/r3_c5` | correct | [passed](operator_peer/metal__f32__log__offset__r3_c5.json) | 0.2432 / 0.2224 | [passed](program_peer/metal__f32__log__offset__r3_c5.json) | 0.2502 / 0.2003 |
| `metal/f32/log/offset/r3_c32` | correct | [passed](operator_peer/metal__f32__log__offset__r3_c32.json) | 0.2402 / 0.1864 | [passed](program_peer/metal__f32__log__offset__r3_c32.json) | 0.249 / 0.1994 |
| `metal/f32/log/offset/r3_c65` | correct | [passed](operator_peer/metal__f32__log__offset__r3_c65.json) | 0.2408 / 0.1995 | [passed](program_peer/metal__f32__log__offset__r3_c65.json) | 0.2476 / 0.1954 |
| `metal/f32/log/offset/r3_c129` | correct | [passed](operator_peer/metal__f32__log__offset__r3_c129.json) | 0.2426 / 0.1925 | [passed](program_peer/metal__f32__log__offset__r3_c129.json) | 0.2506 / 0.1958 |
| `metal/f32/log/offset/r16_c5` | correct | [passed](operator_peer/metal__f32__log__offset__r16_c5.json) | 0.194 / 0.1774 | [passed](program_peer/metal__f32__log__offset__r16_c5.json) | 0.2554 / 0.2174 |
| `metal/f32/log/offset/r16_c32` | correct | [passed](operator_peer/metal__f32__log__offset__r16_c32.json) | 0.2428 / 0.1958 | [passed](program_peer/metal__f32__log__offset__r16_c32.json) | 0.2502 / 0.1963 |
| `metal/f32/log/offset/r16_c65` | correct | [passed](operator_peer/metal__f32__log__offset__r16_c65.json) | 0.2656 / 0.228 | [passed](program_peer/metal__f32__log__offset__r16_c65.json) | 0.2736 / 0.2087 |
| `metal/f32/log/offset/r16_c129` | correct | [passed](operator_peer/metal__f32__log__offset__r16_c129.json) | 0.275 / 0.22 | [passed](program_peer/metal__f32__log__offset__r16_c129.json) | 0.2818 / 0.2235 |
| `metal/f32/log/offset/r33_c5` | correct | [passed](operator_peer/metal__f32__log__offset__r33_c5.json) | 0.2548 / 0.2046 | [passed](program_peer/metal__f32__log__offset__r33_c5.json) | 0.2574 / 0.2193 |
| `metal/f32/log/offset/r33_c32` | correct | [passed](operator_peer/metal__f32__log__offset__r33_c32.json) | 0.274 / 0.2055 | [passed](program_peer/metal__f32__log__offset__r33_c32.json) | 0.282 / 0.2075 |
| `metal/f32/log/offset/r33_c65` | correct | [passed](operator_peer/metal__f32__log__offset__r33_c65.json) | 0.2884 / 0.2223 | [passed](program_peer/metal__f32__log__offset__r33_c65.json) | 0.2944 / 0.2244 |
| `metal/f32/log/offset/r33_c129` | correct | [passed](operator_peer/metal__f32__log__offset__r33_c129.json) | 0.3038 / 0.2415 | [passed](program_peer/metal__f32__log__offset__r33_c129.json) | 0.3032 / 0.2463 |
| `metal/f32/log/offset/r128_c5` | correct | [passed](operator_peer/metal__f32__log__offset__r128_c5.json) | 0.2964 / 0.1982 | [passed](program_peer/metal__f32__log__offset__r128_c5.json) | 0.297 / 0.2246 |
| `metal/f32/log/offset/r128_c32` | correct | [passed](operator_peer/metal__f32__log__offset__r128_c32.json) | 0.3398 / 0.2394 | [passed](program_peer/metal__f32__log__offset__r128_c32.json) | 0.3418 / 0.2428 |
| `metal/f32/log/offset/r128_c65` | correct | [passed](operator_peer/metal__f32__log__offset__r128_c65.json) | 0.3682 / 0.3426 | [passed](program_peer/metal__f32__log__offset__r128_c65.json) | 0.3916 / 0.2872 |
| `metal/f32/log/offset/r128_c129` | correct | [passed](operator_peer/metal__f32__log__offset__r128_c129.json) | 0.454 / 0.4198 | [passed](program_peer/metal__f32__log__offset__r128_c129.json) | 0.4654 / 0.4263 |
| `metal/f32/sqrt/strided/r3_c5` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r3_c5.json) | 0.1986 / 0.1845 | [passed](program_peer/metal__f32__sqrt__strided__r3_c5.json) | 0.2494 / 0.1937 |
| `metal/f32/sqrt/strided/r3_c32` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r3_c32.json) | 0.2414 / 0.1863 | [passed](program_peer/metal__f32__sqrt__strided__r3_c32.json) | 0.2396 / 0.2245 |
| `metal/f32/sqrt/strided/r3_c65` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r3_c65.json) | 0.2342 / 0.2042 | [passed](program_peer/metal__f32__sqrt__strided__r3_c65.json) | 0.2496 / 0.2048 |
| `metal/f32/sqrt/strided/r3_c129` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r3_c129.json) | 0.2498 / 0.209 | [passed](program_peer/metal__f32__sqrt__strided__r3_c129.json) | 0.2534 / 0.2023 |
| `metal/f32/sqrt/strided/r16_c5` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r16_c5.json) | 0.2332 / 0.2032 | [passed](program_peer/metal__f32__sqrt__strided__r16_c5.json) | 0.2156 / 0.1991 |
| `metal/f32/sqrt/strided/r16_c32` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r16_c32.json) | 0.2548 / 0.1972 | [passed](program_peer/metal__f32__sqrt__strided__r16_c32.json) | 0.262 / 0.1953 |
| `metal/f32/sqrt/strided/r16_c65` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r16_c65.json) | 0.265 / 0.208 | [passed](program_peer/metal__f32__sqrt__strided__r16_c65.json) | 0.27 / 0.2242 |
| `metal/f32/sqrt/strided/r16_c129` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r16_c129.json) | 0.2808 / 0.2404 | [passed](program_peer/metal__f32__sqrt__strided__r16_c129.json) | 0.2832 / 0.2199 |
| `metal/f32/sqrt/strided/r33_c5` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r33_c5.json) | 0.2504 / 0.2035 | [passed](program_peer/metal__f32__sqrt__strided__r33_c5.json) | 0.2612 / 0.1892 |
| `metal/f32/sqrt/strided/r33_c32` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r33_c32.json) | 0.2772 / 0.2014 | [passed](program_peer/metal__f32__sqrt__strided__r33_c32.json) | 0.28 / 0.2197 |
| `metal/f32/sqrt/strided/r33_c65` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r33_c65.json) | 0.2848 / 0.2186 | [passed](program_peer/metal__f32__sqrt__strided__r33_c65.json) | 0.2956 / 0.223 |
| `metal/f32/sqrt/strided/r33_c129` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r33_c129.json) | 0.3016 / 0.2418 | [passed](program_peer/metal__f32__sqrt__strided__r33_c129.json) | 0.3144 / 0.2439 |
| `metal/f32/sqrt/strided/r128_c5` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r128_c5.json) | 0.2884 / 0.229 | [passed](program_peer/metal__f32__sqrt__strided__r128_c5.json) | 0.2984 / 0.2072 |
| `metal/f32/sqrt/strided/r128_c32` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r128_c32.json) | 0.3452 / 0.241 | [passed](program_peer/metal__f32__sqrt__strided__r128_c32.json) | 0.349 / 0.2394 |
| `metal/f32/sqrt/strided/r128_c65` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r128_c65.json) | 0.3828 / 0.3378 | [passed](program_peer/metal__f32__sqrt__strided__r128_c65.json) | 0.365 / 0.3405 |
| `metal/f32/sqrt/strided/r128_c129` | correct | [passed](operator_peer/metal__f32__sqrt__strided__r128_c129.json) | 0.4518 / 0.4149 | [passed](program_peer/metal__f32__sqrt__strided__r128_c129.json) | 0.4636 / 0.4256 |
| `metal/f32/sqrt/offset/r3_c5` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r3_c5.json) | 0.2466 / 0.1834 | [passed](program_peer/metal__f32__sqrt__offset__r3_c5.json) | 0.2524 / 0.1926 |
| `metal/f32/sqrt/offset/r3_c32` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r3_c32.json) | 0.2404 / 0.1874 | [passed](program_peer/metal__f32__sqrt__offset__r3_c32.json) | 0.2468 / 0.1924 |
| `metal/f32/sqrt/offset/r3_c65` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r3_c65.json) | 0.244 / 0.2066 | [passed](program_peer/metal__f32__sqrt__offset__r3_c65.json) | 0.2498 / 0.2028 |
| `metal/f32/sqrt/offset/r3_c129` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r3_c129.json) | 0.2492 / 0.2011 | [passed](program_peer/metal__f32__sqrt__offset__r3_c129.json) | 0.2552 / 0.2061 |
| `metal/f32/sqrt/offset/r16_c5` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r16_c5.json) | 0.246 / 0.2025 | [passed](program_peer/metal__f32__sqrt__offset__r16_c5.json) | 0.253 / 0.1994 |
| `metal/f32/sqrt/offset/r16_c32` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r16_c32.json) | 0.1958 / 0.2051 | [passed](program_peer/metal__f32__sqrt__offset__r16_c32.json) | 0.2598 / 0.1985 |
| `metal/f32/sqrt/offset/r16_c65` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r16_c65.json) | 0.2684 / 0.2089 | [passed](program_peer/metal__f32__sqrt__offset__r16_c65.json) | 0.2522 / 0.21 |
| `metal/f32/sqrt/offset/r16_c129` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r16_c129.json) | 0.2734 / 0.2183 | [passed](program_peer/metal__f32__sqrt__offset__r16_c129.json) | 0.277 / 0.2154 |
| `metal/f32/sqrt/offset/r33_c5` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r33_c5.json) | 0.2542 / 0.203 | [passed](program_peer/metal__f32__sqrt__offset__r33_c5.json) | 0.259 / 0.2022 |
| `metal/f32/sqrt/offset/r33_c32` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r33_c32.json) | 0.2714 / 0.2103 | [passed](program_peer/metal__f32__sqrt__offset__r33_c32.json) | 0.279 / 0.2682 |
| `metal/f32/sqrt/offset/r33_c65` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r33_c65.json) | 0.2702 / 0.2217 | [passed](program_peer/metal__f32__sqrt__offset__r33_c65.json) | 0.295 / 0.2256 |
| `metal/f32/sqrt/offset/r33_c129` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r33_c129.json) | 0.2802 / 0.2427 | [passed](program_peer/metal__f32__sqrt__offset__r33_c129.json) | 0.3122 / 0.2479 |
| `metal/f32/sqrt/offset/r128_c5` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r128_c5.json) | 0.2864 / 0.1993 | [passed](program_peer/metal__f32__sqrt__offset__r128_c5.json) | 0.2956 / 0.2061 |
| `metal/f32/sqrt/offset/r128_c32` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r128_c32.json) | 0.3428 / 0.2418 | [passed](program_peer/metal__f32__sqrt__offset__r128_c32.json) | 0.3118 / 0.242 |
| `metal/f32/sqrt/offset/r128_c65` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r128_c65.json) | 0.398 / 0.3094 | [passed](program_peer/metal__f32__sqrt__offset__r128_c65.json) | 0.3888 / 0.3349 |
| `metal/f32/sqrt/offset/r128_c129` | correct | [passed](operator_peer/metal__f32__sqrt__offset__r128_c129.json) | 0.4174 / 0.4153 | [passed](program_peer/metal__f32__sqrt__offset__r128_c129.json) | 0.458 / 0.4255 |
| `metal/bf16/exp/strided/r3_c5` | correct | [passed](operator_peer/metal__bf16__exp__strided__r3_c5.json) | 0.249 / 0.227 | [passed](program_peer/metal__bf16__exp__strided__r3_c5.json) | 0.2602 / 0.1957 |
| `metal/bf16/exp/strided/r3_c32` | correct | [passed](operator_peer/metal__bf16__exp__strided__r3_c32.json) | 0.2216 / 0.1997 | [passed](program_peer/metal__bf16__exp__strided__r3_c32.json) | 0.2572 / 0.2336 |
| `metal/bf16/exp/strided/r3_c65` | correct | [passed](operator_peer/metal__bf16__exp__strided__r3_c65.json) | 0.2196 / 0.2044 | [passed](program_peer/metal__bf16__exp__strided__r3_c65.json) | 0.2652 / 0.1961 |
| `metal/bf16/exp/strided/r3_c129` | correct | [passed](operator_peer/metal__bf16__exp__strided__r3_c129.json) | 0.2502 / 0.2062 | [passed](program_peer/metal__bf16__exp__strided__r3_c129.json) | 0.2286 / 0.1965 |
| `metal/bf16/exp/strided/r16_c5` | correct | [passed](operator_peer/metal__bf16__exp__strided__r16_c5.json) | 0.2534 / 0.2115 | [passed](program_peer/metal__bf16__exp__strided__r16_c5.json) | 0.2558 / 0.1839 |
| `metal/bf16/exp/strided/r16_c32` | correct | [passed](operator_peer/metal__bf16__exp__strided__r16_c32.json) | 0.257 / 0.2104 | [passed](program_peer/metal__bf16__exp__strided__r16_c32.json) | 0.2732 / 0.2046 |
| `metal/bf16/exp/strided/r16_c65` | correct | [passed](operator_peer/metal__bf16__exp__strided__r16_c65.json) | 0.261 / 0.2455 | [passed](program_peer/metal__bf16__exp__strided__r16_c65.json) | 0.3078 / 0.2546 |
| `metal/bf16/exp/strided/r16_c129` | correct | [passed](operator_peer/metal__bf16__exp__strided__r16_c129.json) | 0.2772 / 0.2185 | [passed](program_peer/metal__bf16__exp__strided__r16_c129.json) | 0.3174 / 0.2202 |
| `metal/bf16/exp/strided/r33_c5` | correct | [passed](operator_peer/metal__bf16__exp__strided__r33_c5.json) | 0.254 / 0.1893 | [passed](program_peer/metal__bf16__exp__strided__r33_c5.json) | 0.2756 / 0.2047 |
| `metal/bf16/exp/strided/r33_c32` | correct | [passed](operator_peer/metal__bf16__exp__strided__r33_c32.json) | 0.276 / 0.2005 | [passed](program_peer/metal__bf16__exp__strided__r33_c32.json) | 0.2952 / 0.2007 |
| `metal/bf16/exp/strided/r33_c65` | correct | [passed](operator_peer/metal__bf16__exp__strided__r33_c65.json) | 0.2768 / 0.2188 | [passed](program_peer/metal__bf16__exp__strided__r33_c65.json) | 0.325 / 0.2159 |
| `metal/bf16/exp/strided/r33_c129` | correct | [passed](operator_peer/metal__bf16__exp__strided__r33_c129.json) | 0.3016 / 0.2393 | [passed](program_peer/metal__bf16__exp__strided__r33_c129.json) | 0.3458 / 0.2421 |
| `metal/bf16/exp/strided/r128_c5` | correct | [passed](operator_peer/metal__bf16__exp__strided__r128_c5.json) | 0.29 / 0.1979 | [passed](program_peer/metal__bf16__exp__strided__r128_c5.json) | 0.312 / 0.2097 |
| `metal/bf16/exp/strided/r128_c32` | correct | [passed](operator_peer/metal__bf16__exp__strided__r128_c32.json) | 0.3384 / 0.2357 | [passed](program_peer/metal__bf16__exp__strided__r128_c32.json) | 0.3736 / 0.2364 |
| `metal/bf16/exp/strided/r128_c65` | correct | [passed](operator_peer/metal__bf16__exp__strided__r128_c65.json) | 0.3762 / 0.3016 | [passed](program_peer/metal__bf16__exp__strided__r128_c65.json) | 0.3948 / 0.3242 |
| `metal/bf16/exp/strided/r128_c129` | correct | [passed](operator_peer/metal__bf16__exp__strided__r128_c129.json) | 0.451 / 0.4137 | [passed](program_peer/metal__bf16__exp__strided__r128_c129.json) | 0.51 / 0.4206 |
| `metal/bf16/exp/offset/r3_c5` | correct | [passed](operator_peer/metal__bf16__exp__offset__r3_c5.json) | 0.1946 / 0.1874 | [passed](program_peer/metal__bf16__exp__offset__r3_c5.json) | 0.253 / 0.1851 |
| `metal/bf16/exp/offset/r3_c32` | correct | [passed](operator_peer/metal__bf16__exp__offset__r3_c32.json) | 0.2424 / 0.1971 | [passed](program_peer/metal__bf16__exp__offset__r3_c32.json) | 0.263 / 0.1942 |
| `metal/bf16/exp/offset/r3_c65` | correct | [passed](operator_peer/metal__bf16__exp__offset__r3_c65.json) | 0.2482 / 0.2069 | [passed](program_peer/metal__bf16__exp__offset__r3_c65.json) | 0.257 / 0.186 |
| `metal/bf16/exp/offset/r3_c129` | correct | [passed](operator_peer/metal__bf16__exp__offset__r3_c129.json) | 0.2504 / 0.2088 | [passed](program_peer/metal__bf16__exp__offset__r3_c129.json) | 0.285 / 0.188 |
| `metal/bf16/exp/offset/r16_c5` | correct | [passed](operator_peer/metal__bf16__exp__offset__r16_c5.json) | 0.2476 / 0.2002 | [passed](program_peer/metal__bf16__exp__offset__r16_c5.json) | 0.2696 / 0.2008 |
| `metal/bf16/exp/offset/r16_c32` | correct | [passed](operator_peer/metal__bf16__exp__offset__r16_c32.json) | 0.2524 / 0.1995 | [passed](program_peer/metal__bf16__exp__offset__r16_c32.json) | 0.2642 / 0.1994 |
| `metal/bf16/exp/offset/r16_c65` | correct | [passed](operator_peer/metal__bf16__exp__offset__r16_c65.json) | 0.2682 / 0.2053 | [passed](program_peer/metal__bf16__exp__offset__r16_c65.json) | 0.3082 / 0.1994 |
| `metal/bf16/exp/offset/r16_c129` | correct | [passed](operator_peer/metal__bf16__exp__offset__r16_c129.json) | 0.2776 / 0.2208 | [passed](program_peer/metal__bf16__exp__offset__r16_c129.json) | 0.32 / 0.2219 |
| `metal/bf16/exp/offset/r33_c5` | correct | [passed](operator_peer/metal__bf16__exp__offset__r33_c5.json) | 0.2538 / 0.1875 | [passed](program_peer/metal__bf16__exp__offset__r33_c5.json) | 0.2712 / 0.1924 |
| `metal/bf16/exp/offset/r33_c32` | correct | [passed](operator_peer/metal__bf16__exp__offset__r33_c32.json) | 0.2758 / 0.214 | [passed](program_peer/metal__bf16__exp__offset__r33_c32.json) | 0.3038 / 0.2099 |
| `metal/bf16/exp/offset/r33_c65` | correct | [passed](operator_peer/metal__bf16__exp__offset__r33_c65.json) | 0.2794 / 0.2211 | [passed](program_peer/metal__bf16__exp__offset__r33_c65.json) | 0.331 / 0.2233 |
| `metal/bf16/exp/offset/r33_c129` | correct | [passed](operator_peer/metal__bf16__exp__offset__r33_c129.json) | 0.3038 / 0.2411 | [passed](program_peer/metal__bf16__exp__offset__r33_c129.json) | 0.369 / 0.2411 |
| `metal/bf16/exp/offset/r128_c5` | correct | [passed](operator_peer/metal__bf16__exp__offset__r128_c5.json) | 0.2918 / 0.2032 | [passed](program_peer/metal__bf16__exp__offset__r128_c5.json) | 0.3086 / 0.2041 |
| `metal/bf16/exp/offset/r128_c32` | correct | [passed](operator_peer/metal__bf16__exp__offset__r128_c32.json) | 0.3118 / 0.2401 | [passed](program_peer/metal__bf16__exp__offset__r128_c32.json) | 0.3796 / 0.243 |
| `metal/bf16/exp/offset/r128_c65` | correct | [passed](operator_peer/metal__bf16__exp__offset__r128_c65.json) | 0.3816 / 0.2771 | [passed](program_peer/metal__bf16__exp__offset__r128_c65.json) | 0.4194 / 0.3404 |
| `metal/bf16/exp/offset/r128_c129` | correct | [passed](operator_peer/metal__bf16__exp__offset__r128_c129.json) | 0.4514 / 0.4129 | [passed](program_peer/metal__bf16__exp__offset__r128_c129.json) | 0.5126 / 0.426 |
| `metal/bf16/log/strided/r3_c5` | correct | [passed](operator_peer/metal__bf16__log__strided__r3_c5.json) | 0.2474 / 0.2047 | [passed](program_peer/metal__bf16__log__strided__r3_c5.json) | 0.256 / 0.1987 |
| `metal/bf16/log/strided/r3_c32` | correct | [passed](operator_peer/metal__bf16__log__strided__r3_c32.json) | 0.2436 / 0.186 | [passed](program_peer/metal__bf16__log__strided__r3_c32.json) | 0.2576 / 0.2 |
| `metal/bf16/log/strided/r3_c65` | correct | [passed](operator_peer/metal__bf16__log__strided__r3_c65.json) | 0.2472 / 0.2031 | [passed](program_peer/metal__bf16__log__strided__r3_c65.json) | 0.2494 / 0.1995 |
| `metal/bf16/log/strided/r3_c129` | correct | [passed](operator_peer/metal__bf16__log__strided__r3_c129.json) | 0.2462 / 0.1976 | [passed](program_peer/metal__bf16__log__strided__r3_c129.json) | 0.2638 / 0.1923 |
| `metal/bf16/log/strided/r16_c5` | correct | [passed](operator_peer/metal__bf16__log__strided__r16_c5.json) | 0.2516 / 0.1933 | [passed](program_peer/metal__bf16__log__strided__r16_c5.json) | 0.2694 / 0.1916 |
| `metal/bf16/log/strided/r16_c32` | correct | [passed](operator_peer/metal__bf16__log__strided__r16_c32.json) | 0.2508 / 0.1909 | [passed](program_peer/metal__bf16__log__strided__r16_c32.json) | 0.2734 / 0.1967 |
| `metal/bf16/log/strided/r16_c65` | correct | [passed](operator_peer/metal__bf16__log__strided__r16_c65.json) | 0.2676 / 0.2006 | [passed](program_peer/metal__bf16__log__strided__r16_c65.json) | 0.3132 / 0.205 |
| `metal/bf16/log/strided/r16_c129` | correct | [passed](operator_peer/metal__bf16__log__strided__r16_c129.json) | 0.2796 / 0.2565 | [passed](program_peer/metal__bf16__log__strided__r16_c129.json) | 0.3078 / 0.2229 |
| `metal/bf16/log/strided/r33_c5` | correct | [passed](operator_peer/metal__bf16__log__strided__r33_c5.json) | 0.2568 / 0.2015 | [passed](program_peer/metal__bf16__log__strided__r33_c5.json) | 0.2714 / 0.1921 |
| `metal/bf16/log/strided/r33_c32` | correct | [passed](operator_peer/metal__bf16__log__strided__r33_c32.json) | 0.2788 / 0.2311 | [passed](program_peer/metal__bf16__log__strided__r33_c32.json) | 0.3174 / 0.2137 |
| `metal/bf16/log/strided/r33_c65` | correct | [passed](operator_peer/metal__bf16__log__strided__r33_c65.json) | 0.2886 / 0.2069 | [passed](program_peer/metal__bf16__log__strided__r33_c65.json) | 0.3318 / 0.2106 |
| `metal/bf16/log/strided/r33_c129` | correct | [passed](operator_peer/metal__bf16__log__strided__r33_c129.json) | 0.3086 / 0.2353 | [passed](program_peer/metal__bf16__log__strided__r33_c129.json) | 0.3314 / 0.243 |
| `metal/bf16/log/strided/r128_c5` | correct | [passed](operator_peer/metal__bf16__log__strided__r128_c5.json) | 0.291 / 0.2025 | [passed](program_peer/metal__bf16__log__strided__r128_c5.json) | 0.2792 / 0.2063 |
| `metal/bf16/log/strided/r128_c32` | correct | [passed](operator_peer/metal__bf16__log__strided__r128_c32.json) | 0.3364 / 0.2376 | [passed](program_peer/metal__bf16__log__strided__r128_c32.json) | 0.3794 / 0.2444 |
| `metal/bf16/log/strided/r128_c65` | correct | [passed](operator_peer/metal__bf16__log__strided__r128_c65.json) | 0.3836 / 0.2831 | [passed](program_peer/metal__bf16__log__strided__r128_c65.json) | 0.4216 / 0.2846 |
| `metal/bf16/log/strided/r128_c129` | correct | [passed](operator_peer/metal__bf16__log__strided__r128_c129.json) | 0.453 / 0.4195 | [passed](program_peer/metal__bf16__log__strided__r128_c129.json) | 0.4994 / 0.4042 |
| `metal/bf16/log/offset/r3_c5` | correct | [passed](operator_peer/metal__bf16__log__offset__r3_c5.json) | 0.2478 / 0.1994 | [passed](program_peer/metal__bf16__log__offset__r3_c5.json) | 0.2636 / 0.1971 |
| `metal/bf16/log/offset/r3_c32` | correct | [passed](operator_peer/metal__bf16__log__offset__r3_c32.json) | 0.2448 / 0.2017 | [passed](program_peer/metal__bf16__log__offset__r3_c32.json) | 0.2564 / 0.1879 |
| `metal/bf16/log/offset/r3_c65` | correct | [passed](operator_peer/metal__bf16__log__offset__r3_c65.json) | 0.2474 / 0.1971 | [passed](program_peer/metal__bf16__log__offset__r3_c65.json) | 0.2592 / 0.23 |
| `metal/bf16/log/offset/r3_c129` | correct | [passed](operator_peer/metal__bf16__log__offset__r3_c129.json) | 0.2452 / 0.188 | [passed](program_peer/metal__bf16__log__offset__r3_c129.json) | 0.269 / 0.1996 |
| `metal/bf16/log/offset/r16_c5` | correct | [passed](operator_peer/metal__bf16__log__offset__r16_c5.json) | 0.2508 / 0.1954 | [passed](program_peer/metal__bf16__log__offset__r16_c5.json) | 0.26 / 0.2013 |
| `metal/bf16/log/offset/r16_c32` | correct | [passed](operator_peer/metal__bf16__log__offset__r16_c32.json) | 0.2574 / 0.2015 | [passed](program_peer/metal__bf16__log__offset__r16_c32.json) | 0.2718 / 0.1948 |
| `metal/bf16/log/offset/r16_c65` | correct | [passed](operator_peer/metal__bf16__log__offset__r16_c65.json) | 0.2662 / 0.2084 | [passed](program_peer/metal__bf16__log__offset__r16_c65.json) | 0.3144 / 0.2067 |
| `metal/bf16/log/offset/r16_c129` | correct | [passed](operator_peer/metal__bf16__log__offset__r16_c129.json) | 0.2758 / 0.2149 | [passed](program_peer/metal__bf16__log__offset__r16_c129.json) | 0.3212 / 0.2218 |
| `metal/bf16/log/offset/r33_c5` | correct | [passed](operator_peer/metal__bf16__log__offset__r33_c5.json) | 0.2608 / 0.202 | [passed](program_peer/metal__bf16__log__offset__r33_c5.json) | 0.2658 / 0.1887 |
| `metal/bf16/log/offset/r33_c32` | correct | [passed](operator_peer/metal__bf16__log__offset__r33_c32.json) | 0.2754 / 0.2343 | [passed](program_peer/metal__bf16__log__offset__r33_c32.json) | 0.3222 / 0.211 |
| `metal/bf16/log/offset/r33_c65` | correct | [passed](operator_peer/metal__bf16__log__offset__r33_c65.json) | 0.2916 / 0.2209 | [passed](program_peer/metal__bf16__log__offset__r33_c65.json) | 0.3308 / 0.2231 |
| `metal/bf16/log/offset/r33_c129` | correct | [passed](operator_peer/metal__bf16__log__offset__r33_c129.json) | 0.3008 / 0.2381 | [passed](program_peer/metal__bf16__log__offset__r33_c129.json) | 0.3358 / 0.2449 |
| `metal/bf16/log/offset/r128_c5` | correct | [passed](operator_peer/metal__bf16__log__offset__r128_c5.json) | 0.2904 / 0.2022 | [passed](program_peer/metal__bf16__log__offset__r128_c5.json) | 0.3172 / 0.2105 |
| `metal/bf16/log/offset/r128_c32` | correct | [passed](operator_peer/metal__bf16__log__offset__r128_c32.json) | 0.34 / 0.2389 | [passed](program_peer/metal__bf16__log__offset__r128_c32.json) | 0.3766 / 0.2432 |
| `metal/bf16/log/offset/r128_c65` | correct | [passed](operator_peer/metal__bf16__log__offset__r128_c65.json) | 0.3852 / 0.307 | [passed](program_peer/metal__bf16__log__offset__r128_c65.json) | 0.3754 / 0.2973 |
| `metal/bf16/log/offset/r128_c129` | correct | [passed](operator_peer/metal__bf16__log__offset__r128_c129.json) | 0.4582 / 0.42 | [passed](program_peer/metal__bf16__log__offset__r128_c129.json) | 0.4758 / 0.3958 |
| `metal/bf16/sqrt/strided/r3_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r3_c5.json) | 0.244 / 0.2125 | [passed](program_peer/metal__bf16__sqrt__strided__r3_c5.json) | 0.2402 / 0.1773 |
| `metal/bf16/sqrt/strided/r3_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r3_c32.json) | 0.2486 / 0.2064 | [passed](program_peer/metal__bf16__sqrt__strided__r3_c32.json) | 0.2348 / 0.2041 |
| `metal/bf16/sqrt/strided/r3_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r3_c65.json) | 0.2476 / 0.2016 | [passed](program_peer/metal__bf16__sqrt__strided__r3_c65.json) | 0.2494 / 0.1853 |
| `metal/bf16/sqrt/strided/r3_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r3_c129.json) | 0.249 / 0.2063 | [passed](program_peer/metal__bf16__sqrt__strided__r3_c129.json) | 0.2624 / 0.2001 |
| `metal/bf16/sqrt/strided/r16_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r16_c5.json) | 0.2484 / 0.2023 | [passed](program_peer/metal__bf16__sqrt__strided__r16_c5.json) | 0.2664 / 0.2077 |
| `metal/bf16/sqrt/strided/r16_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r16_c32.json) | 0.258 / 0.1981 | [passed](program_peer/metal__bf16__sqrt__strided__r16_c32.json) | 0.2704 / 0.2014 |
| `metal/bf16/sqrt/strided/r16_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r16_c65.json) | 0.2622 / 0.199 | [passed](program_peer/metal__bf16__sqrt__strided__r16_c65.json) | 0.2768 / 0.2061 |
| `metal/bf16/sqrt/strided/r16_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r16_c129.json) | 0.277 / 0.2082 | [passed](program_peer/metal__bf16__sqrt__strided__r16_c129.json) | 0.3198 / 0.2185 |
| `metal/bf16/sqrt/strided/r33_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r33_c5.json) | 0.2538 / 0.1855 | [passed](program_peer/metal__bf16__sqrt__strided__r33_c5.json) | 0.2716 / 0.1964 |
| `metal/bf16/sqrt/strided/r33_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r33_c32.json) | 0.3148 / 0.2084 | [passed](program_peer/metal__bf16__sqrt__strided__r33_c32.json) | 0.3226 / 0.2069 |
| `metal/bf16/sqrt/strided/r33_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r33_c65.json) | 0.2866 / 0.2236 | [passed](program_peer/metal__bf16__sqrt__strided__r33_c65.json) | 0.297 / 0.2139 |
| `metal/bf16/sqrt/strided/r33_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r33_c129.json) | 0.3062 / 0.2409 | [passed](program_peer/metal__bf16__sqrt__strided__r33_c129.json) | 0.3448 / 0.2458 |
| `metal/bf16/sqrt/strided/r128_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r128_c5.json) | 0.2894 / 0.207 | [passed](program_peer/metal__bf16__sqrt__strided__r128_c5.json) | 0.31 / 0.2015 |
| `metal/bf16/sqrt/strided/r128_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r128_c32.json) | 0.3494 / 0.2384 | [passed](program_peer/metal__bf16__sqrt__strided__r128_c32.json) | 0.3778 / 0.2411 |
| `metal/bf16/sqrt/strided/r128_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r128_c65.json) | 0.3746 / 0.278 | [passed](program_peer/metal__bf16__sqrt__strided__r128_c65.json) | 0.4102 / 0.3013 |
| `metal/bf16/sqrt/strided/r128_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__strided__r128_c129.json) | 0.4594 / 0.4146 | [passed](program_peer/metal__bf16__sqrt__strided__r128_c129.json) | 0.4926 / 0.3951 |
| `metal/bf16/sqrt/offset/r3_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r3_c5.json) | 0.2472 / 0.186 | [passed](program_peer/metal__bf16__sqrt__offset__r3_c5.json) | 0.2434 / 0.1869 |
| `metal/bf16/sqrt/offset/r3_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r3_c32.json) | 0.247 / 0.196 | [passed](program_peer/metal__bf16__sqrt__offset__r3_c32.json) | 0.251 / 0.196 |
| `metal/bf16/sqrt/offset/r3_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r3_c65.json) | 0.2412 / 0.1894 | [passed](program_peer/metal__bf16__sqrt__offset__r3_c65.json) | 0.2446 / 0.1955 |
| `metal/bf16/sqrt/offset/r3_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r3_c129.json) | 0.2468 / 0.1926 | [passed](program_peer/metal__bf16__sqrt__offset__r3_c129.json) | 0.2524 / 0.1748 |
| `metal/bf16/sqrt/offset/r16_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r16_c5.json) | 0.2512 / 0.1981 | [passed](program_peer/metal__bf16__sqrt__offset__r16_c5.json) | 0.2542 / 0.2131 |
| `metal/bf16/sqrt/offset/r16_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r16_c32.json) | 0.2588 / 0.2057 | [passed](program_peer/metal__bf16__sqrt__offset__r16_c32.json) | 0.2718 / 0.2046 |
| `metal/bf16/sqrt/offset/r16_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r16_c65.json) | 0.2648 / 0.2028 | [passed](program_peer/metal__bf16__sqrt__offset__r16_c65.json) | 0.3058 / 0.2018 |
| `metal/bf16/sqrt/offset/r16_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r16_c129.json) | 0.276 / 0.218 | [passed](program_peer/metal__bf16__sqrt__offset__r16_c129.json) | 0.3226 / 0.2227 |
| `metal/bf16/sqrt/offset/r33_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r33_c5.json) | 0.2176 / 0.1996 | [passed](program_peer/metal__bf16__sqrt__offset__r33_c5.json) | 0.2704 / 0.2039 |
| `metal/bf16/sqrt/offset/r33_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r33_c32.json) | 0.2708 / 0.2318 | [passed](program_peer/metal__bf16__sqrt__offset__r33_c32.json) | 0.3186 / 0.2114 |
| `metal/bf16/sqrt/offset/r33_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r33_c65.json) | 0.2874 / 0.2159 | [passed](program_peer/metal__bf16__sqrt__offset__r33_c65.json) | 0.3284 / 0.2146 |
| `metal/bf16/sqrt/offset/r33_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r33_c129.json) | 0.2988 / 0.2388 | [passed](program_peer/metal__bf16__sqrt__offset__r33_c129.json) | 0.3298 / 0.241 |
| `metal/bf16/sqrt/offset/r128_c5` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r128_c5.json) | 0.2908 / 0.1986 | [passed](program_peer/metal__bf16__sqrt__offset__r128_c5.json) | 0.3096 / 0.2087 |
| `metal/bf16/sqrt/offset/r128_c32` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r128_c32.json) | 0.3412 / 0.2402 | [passed](program_peer/metal__bf16__sqrt__offset__r128_c32.json) | 0.3848 / 0.2424 |
| `metal/bf16/sqrt/offset/r128_c65` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r128_c65.json) | 0.3776 / 0.306 | [passed](program_peer/metal__bf16__sqrt__offset__r128_c65.json) | 0.419 / 0.2865 |
| `metal/bf16/sqrt/offset/r128_c129` | correct | [passed](operator_peer/metal__bf16__sqrt__offset__r128_c129.json) | 0.4538 / 0.421 | [passed](program_peer/metal__bf16__sqrt__offset__r128_c129.json) | 0.5324 / 0.4324 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
