# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2d5e1e0d9f71d4c661e668fad23af4b9feb10feebb9f7d5c6efb523c5f49de98`  
Binary SHA-256: `af1b758d59cd4f42a812cf3dcaae67d3b0f01abca19878e39bc7300e1ae67e35`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 128
- Numerical capability passed: 128
- Operator parity and timing passed: 128
- Program parity and timing passed: 128
- Probe failures or incorrect: 0
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/cast/strided/r1_c3` | correct | [passed](operator_peer/metal__f32__cast__strided__r1_c3.json) | 0.2732 / 0.1885 | [passed](program_peer/metal__f32__cast__strided__r1_c3.json) | 0.25 / 0.2126 |
| `metal/f32/cast/strided/r1_c7` | correct | [passed](operator_peer/metal__f32__cast__strided__r1_c7.json) | 0.1842 / 0.1772 | [passed](program_peer/metal__f32__cast__strided__r1_c7.json) | 0.257 / 0.1991 |
| `metal/f32/cast/strided/r1_c31` | correct | [passed](operator_peer/metal__f32__cast__strided__r1_c31.json) | 0.1984 / 0.1787 | [passed](program_peer/metal__f32__cast__strided__r1_c31.json) | 0.2558 / 0.206 |
| `metal/f32/cast/strided/r1_c32` | correct | [passed](operator_peer/metal__f32__cast__strided__r1_c32.json) | 0.2424 / 0.1885 | [passed](program_peer/metal__f32__cast__strided__r1_c32.json) | 0.253 / 0.199 |
| `metal/f32/cast/strided/r1_c33` | correct | [passed](operator_peer/metal__f32__cast__strided__r1_c33.json) | 0.2394 / 0.1958 | [passed](program_peer/metal__f32__cast__strided__r1_c33.json) | 0.2524 / 0.2065 |
| `metal/f32/cast/strided/r1_c63` | correct | [passed](operator_peer/metal__f32__cast__strided__r1_c63.json) | 0.2218 / 0.1914 | [passed](program_peer/metal__f32__cast__strided__r1_c63.json) | 0.2408 / 0.1945 |
| `metal/f32/cast/strided/r1_c64` | correct | [passed](operator_peer/metal__f32__cast__strided__r1_c64.json) | 0.2248 / 0.1931 | [passed](program_peer/metal__f32__cast__strided__r1_c64.json) | 0.2362 / 0.1842 |
| `metal/f32/cast/strided/r1_c65` | correct | [passed](operator_peer/metal__f32__cast__strided__r1_c65.json) | 0.2004 / 0.1917 | [passed](program_peer/metal__f32__cast__strided__r1_c65.json) | 0.2516 / 0.1925 |
| `metal/f32/cast/strided/r7_c3` | correct | [passed](operator_peer/metal__f32__cast__strided__r7_c3.json) | 0.2054 / 0.1905 | [passed](program_peer/metal__f32__cast__strided__r7_c3.json) | 0.247 / 0.1985 |
| `metal/f32/cast/strided/r7_c7` | correct | [passed](operator_peer/metal__f32__cast__strided__r7_c7.json) | 0.2222 / 0.1933 | [passed](program_peer/metal__f32__cast__strided__r7_c7.json) | 0.2672 / 0.1991 |
| `metal/f32/cast/strided/r7_c31` | correct | [passed](operator_peer/metal__f32__cast__strided__r7_c31.json) | 0.2476 / 0.1906 | [passed](program_peer/metal__f32__cast__strided__r7_c31.json) | 0.2758 / 0.1927 |
| `metal/f32/cast/strided/r7_c32` | correct | [passed](operator_peer/metal__f32__cast__strided__r7_c32.json) | 0.2058 / 0.192 | [passed](program_peer/metal__f32__cast__strided__r7_c32.json) | 0.2618 / 0.1933 |
| `metal/f32/cast/strided/r7_c33` | correct | [passed](operator_peer/metal__f32__cast__strided__r7_c33.json) | 0.1918 / 0.1995 | [passed](program_peer/metal__f32__cast__strided__r7_c33.json) | 0.2608 / 0.1887 |
| `metal/f32/cast/strided/r7_c63` | correct | [passed](operator_peer/metal__f32__cast__strided__r7_c63.json) | 0.2506 / 0.2024 | [passed](program_peer/metal__f32__cast__strided__r7_c63.json) | 0.245 / 0.1995 |
| `metal/f32/cast/strided/r7_c64` | correct | [passed](operator_peer/metal__f32__cast__strided__r7_c64.json) | 0.1936 / 0.1889 | [passed](program_peer/metal__f32__cast__strided__r7_c64.json) | 0.2628 / 0.2001 |
| `metal/f32/cast/strided/r7_c65` | correct | [passed](operator_peer/metal__f32__cast__strided__r7_c65.json) | 0.222 / 0.2023 | [passed](program_peer/metal__f32__cast__strided__r7_c65.json) | 0.2684 / 0.194 |
| `metal/f32/cast/strided/r32_c3` | correct | [passed](operator_peer/metal__f32__cast__strided__r32_c3.json) | 0.2058 / 0.1947 | [passed](program_peer/metal__f32__cast__strided__r32_c3.json) | 0.269 / 0.2004 |
| `metal/f32/cast/strided/r32_c7` | correct | [passed](operator_peer/metal__f32__cast__strided__r32_c7.json) | 0.2404 / 0.1946 | [passed](program_peer/metal__f32__cast__strided__r32_c7.json) | 0.2752 / 0.1946 |
| `metal/f32/cast/strided/r32_c31` | correct | [passed](operator_peer/metal__f32__cast__strided__r32_c31.json) | 0.2516 / 0.2039 | [passed](program_peer/metal__f32__cast__strided__r32_c31.json) | 0.2652 / 0.2131 |
| `metal/f32/cast/strided/r32_c32` | correct | [passed](operator_peer/metal__f32__cast__strided__r32_c32.json) | 0.2302 / 0.1946 | [passed](program_peer/metal__f32__cast__strided__r32_c32.json) | 0.2598 / 0.2152 |
| `metal/f32/cast/strided/r32_c33` | correct | [passed](operator_peer/metal__f32__cast__strided__r32_c33.json) | 0.2524 / 0.2111 | [passed](program_peer/metal__f32__cast__strided__r32_c33.json) | 0.3154 / 0.2093 |
| `metal/f32/cast/strided/r32_c63` | correct | [passed](operator_peer/metal__f32__cast__strided__r32_c63.json) | 0.2558 / 0.2221 | [passed](program_peer/metal__f32__cast__strided__r32_c63.json) | 0.3274 / 0.2195 |
| `metal/f32/cast/strided/r32_c64` | correct | [passed](operator_peer/metal__f32__cast__strided__r32_c64.json) | 0.2826 / 0.2114 | [passed](program_peer/metal__f32__cast__strided__r32_c64.json) | 0.3292 / 0.2194 |
| `metal/f32/cast/strided/r32_c65` | correct | [passed](operator_peer/metal__f32__cast__strided__r32_c65.json) | 0.2608 / 0.2251 | [passed](program_peer/metal__f32__cast__strided__r32_c65.json) | 0.3332 / 0.2227 |
| `metal/f32/cast/strided/r129_c3` | correct | [passed](operator_peer/metal__f32__cast__strided__r129_c3.json) | 0.2308 / 0.1982 | [passed](program_peer/metal__f32__cast__strided__r129_c3.json) | 0.2896 / 0.2019 |
| `metal/f32/cast/strided/r129_c7` | correct | [passed](operator_peer/metal__f32__cast__strided__r129_c7.json) | 0.2594 / 0.2082 | [passed](program_peer/metal__f32__cast__strided__r129_c7.json) | 0.343 / 0.2046 |
| `metal/f32/cast/strided/r129_c31` | correct | [passed](operator_peer/metal__f32__cast__strided__r129_c31.json) | 0.3208 / 0.2382 | [passed](program_peer/metal__f32__cast__strided__r129_c31.json) | 0.372 / 0.2397 |
| `metal/f32/cast/strided/r129_c32` | correct | [passed](operator_peer/metal__f32__cast__strided__r129_c32.json) | 0.3166 / 0.243 | [passed](program_peer/metal__f32__cast__strided__r129_c32.json) | 0.3866 / 0.2406 |
| `metal/f32/cast/strided/r129_c33` | correct | [passed](operator_peer/metal__f32__cast__strided__r129_c33.json) | 0.3028 / 0.2448 | [passed](program_peer/metal__f32__cast__strided__r129_c33.json) | 0.3404 / 0.2374 |
| `metal/f32/cast/strided/r129_c63` | correct | [passed](operator_peer/metal__f32__cast__strided__r129_c63.json) | 0.3408 / 0.276 | [passed](program_peer/metal__f32__cast__strided__r129_c63.json) | 0.4176 / 0.3106 |
| `metal/f32/cast/strided/r129_c64` | correct | [passed](operator_peer/metal__f32__cast__strided__r129_c64.json) | 0.3736 / 0.2998 | [passed](program_peer/metal__f32__cast__strided__r129_c64.json) | 0.4106 / 0.2779 |
| `metal/f32/cast/strided/r129_c65` | correct | [passed](operator_peer/metal__f32__cast__strided__r129_c65.json) | 0.375 / 0.2911 | [passed](program_peer/metal__f32__cast__strided__r129_c65.json) | 0.4232 / 0.3207 |
| `metal/f32/cast/offset/r1_c3` | correct | [passed](operator_peer/metal__f32__cast__offset__r1_c3.json) | 0.1922 / 0.193 | [passed](program_peer/metal__f32__cast__offset__r1_c3.json) | 0.2556 / 0.1888 |
| `metal/f32/cast/offset/r1_c7` | correct | [passed](operator_peer/metal__f32__cast__offset__r1_c7.json) | 0.1926 / 0.1937 | [passed](program_peer/metal__f32__cast__offset__r1_c7.json) | 0.255 / 0.1967 |
| `metal/f32/cast/offset/r1_c31` | correct | [passed](operator_peer/metal__f32__cast__offset__r1_c31.json) | 0.2328 / 0.1893 | [passed](program_peer/metal__f32__cast__offset__r1_c31.json) | 0.2582 / 0.1938 |
| `metal/f32/cast/offset/r1_c32` | correct | [passed](operator_peer/metal__f32__cast__offset__r1_c32.json) | 0.2008 / 0.1914 | [passed](program_peer/metal__f32__cast__offset__r1_c32.json) | 0.253 / 0.2004 |
| `metal/f32/cast/offset/r1_c33` | correct | [passed](operator_peer/metal__f32__cast__offset__r1_c33.json) | 0.2402 / 0.1922 | [passed](program_peer/metal__f32__cast__offset__r1_c33.json) | 0.2562 / 0.2109 |
| `metal/f32/cast/offset/r1_c63` | correct | [passed](operator_peer/metal__f32__cast__offset__r1_c63.json) | 0.1962 / 0.2005 | [passed](program_peer/metal__f32__cast__offset__r1_c63.json) | 0.257 / 0.1992 |
| `metal/f32/cast/offset/r1_c64` | correct | [passed](operator_peer/metal__f32__cast__offset__r1_c64.json) | 0.2006 / 0.2146 | [passed](program_peer/metal__f32__cast__offset__r1_c64.json) | 0.2538 / 0.1987 |
| `metal/f32/cast/offset/r1_c65` | correct | [passed](operator_peer/metal__f32__cast__offset__r1_c65.json) | 0.228 / 0.1975 | [passed](program_peer/metal__f32__cast__offset__r1_c65.json) | 0.2544 / 0.2006 |
| `metal/f32/cast/offset/r7_c3` | correct | [passed](operator_peer/metal__f32__cast__offset__r7_c3.json) | 0.1988 / 0.1857 | [passed](program_peer/metal__f32__cast__offset__r7_c3.json) | 0.272 / 0.1948 |
| `metal/f32/cast/offset/r7_c7` | correct | [passed](operator_peer/metal__f32__cast__offset__r7_c7.json) | 0.2532 / 0.1999 | [passed](program_peer/metal__f32__cast__offset__r7_c7.json) | 0.2598 / 0.2017 |
| `metal/f32/cast/offset/r7_c31` | correct | [passed](operator_peer/metal__f32__cast__offset__r7_c31.json) | 0.2148 / 0.2028 | [passed](program_peer/metal__f32__cast__offset__r7_c31.json) | 0.2662 / 0.1924 |
| `metal/f32/cast/offset/r7_c32` | correct | [passed](operator_peer/metal__f32__cast__offset__r7_c32.json) | 0.2266 / 0.2045 | [passed](program_peer/metal__f32__cast__offset__r7_c32.json) | 0.2626 / 0.1907 |
| `metal/f32/cast/offset/r7_c33` | correct | [passed](operator_peer/metal__f32__cast__offset__r7_c33.json) | 0.2436 / 0.2024 | [passed](program_peer/metal__f32__cast__offset__r7_c33.json) | 0.2514 / 0.2024 |
| `metal/f32/cast/offset/r7_c63` | correct | [passed](operator_peer/metal__f32__cast__offset__r7_c63.json) | 0.231 / 0.1957 | [passed](program_peer/metal__f32__cast__offset__r7_c63.json) | 0.2672 / 0.1883 |
| `metal/f32/cast/offset/r7_c64` | correct | [passed](operator_peer/metal__f32__cast__offset__r7_c64.json) | 0.2046 / 0.1901 | [passed](program_peer/metal__f32__cast__offset__r7_c64.json) | 0.2638 / 0.2073 |
| `metal/f32/cast/offset/r7_c65` | correct | [passed](operator_peer/metal__f32__cast__offset__r7_c65.json) | 0.2064 / 0.1934 | [passed](program_peer/metal__f32__cast__offset__r7_c65.json) | 0.2308 / 0.2044 |
| `metal/f32/cast/offset/r32_c3` | correct | [passed](operator_peer/metal__f32__cast__offset__r32_c3.json) | 0.2434 / 0.2099 | [passed](program_peer/metal__f32__cast__offset__r32_c3.json) | 0.2818 / 0.194 |
| `metal/f32/cast/offset/r32_c7` | correct | [passed](operator_peer/metal__f32__cast__offset__r32_c7.json) | 0.2016 / 0.1943 | [passed](program_peer/metal__f32__cast__offset__r32_c7.json) | 0.268 / 0.2049 |
| `metal/f32/cast/offset/r32_c31` | correct | [passed](operator_peer/metal__f32__cast__offset__r32_c31.json) | 0.2138 / 0.187 | [passed](program_peer/metal__f32__cast__offset__r32_c31.json) | 0.3016 / 0.2071 |
| `metal/f32/cast/offset/r32_c32` | correct | [passed](operator_peer/metal__f32__cast__offset__r32_c32.json) | 0.2372 / 0.1809 | [passed](program_peer/metal__f32__cast__offset__r32_c32.json) | 0.2528 / 0.1949 |
| `metal/f32/cast/offset/r32_c33` | correct | [passed](operator_peer/metal__f32__cast__offset__r32_c33.json) | 0.207 / 0.1654 | [passed](program_peer/metal__f32__cast__offset__r32_c33.json) | 0.3114 / 0.2059 |
| `metal/f32/cast/offset/r32_c63` | correct | [passed](operator_peer/metal__f32__cast__offset__r32_c63.json) | 0.249 / 0.2056 | [passed](program_peer/metal__f32__cast__offset__r32_c63.json) | 0.3302 / 0.2223 |
| `metal/f32/cast/offset/r32_c64` | correct | [passed](operator_peer/metal__f32__cast__offset__r32_c64.json) | 0.2396 / 0.2151 | [passed](program_peer/metal__f32__cast__offset__r32_c64.json) | 0.323 / 0.2305 |
| `metal/f32/cast/offset/r32_c65` | correct | [passed](operator_peer/metal__f32__cast__offset__r32_c65.json) | 0.2594 / 0.2213 | [passed](program_peer/metal__f32__cast__offset__r32_c65.json) | 0.33 / 0.218 |
| `metal/f32/cast/offset/r129_c3` | correct | [passed](operator_peer/metal__f32__cast__offset__r129_c3.json) | 0.2762 / 0.1994 | [passed](program_peer/metal__f32__cast__offset__r129_c3.json) | 0.2976 / 0.2088 |
| `metal/f32/cast/offset/r129_c7` | correct | [passed](operator_peer/metal__f32__cast__offset__r129_c7.json) | 0.3042 / 0.2134 | [passed](program_peer/metal__f32__cast__offset__r129_c7.json) | 0.3388 / 0.2096 |
| `metal/f32/cast/offset/r129_c31` | correct | [passed](operator_peer/metal__f32__cast__offset__r129_c31.json) | 0.3434 / 0.2412 | [passed](program_peer/metal__f32__cast__offset__r129_c31.json) | 0.3816 / 0.2428 |
| `metal/f32/cast/offset/r129_c32` | correct | [passed](operator_peer/metal__f32__cast__offset__r129_c32.json) | 0.345 / 0.2401 | [passed](program_peer/metal__f32__cast__offset__r129_c32.json) | 0.3732 / 0.2442 |
| `metal/f32/cast/offset/r129_c33` | correct | [passed](operator_peer/metal__f32__cast__offset__r129_c33.json) | 0.3374 / 0.2407 | [passed](program_peer/metal__f32__cast__offset__r129_c33.json) | 0.3368 / 0.2455 |
| `metal/f32/cast/offset/r129_c63` | correct | [passed](operator_peer/metal__f32__cast__offset__r129_c63.json) | 0.3764 / 0.2849 | [passed](program_peer/metal__f32__cast__offset__r129_c63.json) | 0.4088 / 0.302 |
| `metal/f32/cast/offset/r129_c64` | correct | [passed](operator_peer/metal__f32__cast__offset__r129_c64.json) | 0.3776 / 0.2858 | [passed](program_peer/metal__f32__cast__offset__r129_c64.json) | 0.3808 / 0.3119 |
| `metal/f32/cast/offset/r129_c65` | correct | [passed](operator_peer/metal__f32__cast__offset__r129_c65.json) | 0.381 / 0.2812 | [passed](program_peer/metal__f32__cast__offset__r129_c65.json) | 0.3988 / 0.2954 |
| `metal/bf16/cast/strided/r1_c3` | correct | [passed](operator_peer/metal__bf16__cast__strided__r1_c3.json) | 0.229 / 0.1894 | [passed](program_peer/metal__bf16__cast__strided__r1_c3.json) | 0.2504 / 0.2239 |
| `metal/bf16/cast/strided/r1_c7` | correct | [passed](operator_peer/metal__bf16__cast__strided__r1_c7.json) | 0.2436 / 0.2034 | [passed](program_peer/metal__bf16__cast__strided__r1_c7.json) | 0.2504 / 0.1919 |
| `metal/bf16/cast/strided/r1_c31` | correct | [passed](operator_peer/metal__bf16__cast__strided__r1_c31.json) | 0.1918 / 0.1899 | [passed](program_peer/metal__bf16__cast__strided__r1_c31.json) | 0.2506 / 0.1954 |
| `metal/bf16/cast/strided/r1_c32` | correct | [passed](operator_peer/metal__bf16__cast__strided__r1_c32.json) | 0.2438 / 0.1989 | [passed](program_peer/metal__bf16__cast__strided__r1_c32.json) | 0.2468 / 0.225 |
| `metal/bf16/cast/strided/r1_c33` | correct | [passed](operator_peer/metal__bf16__cast__strided__r1_c33.json) | 0.1928 / 0.1914 | [passed](program_peer/metal__bf16__cast__strided__r1_c33.json) | 0.2522 / 0.2172 |
| `metal/bf16/cast/strided/r1_c63` | correct | [passed](operator_peer/metal__bf16__cast__strided__r1_c63.json) | 0.2276 / 0.1904 | [passed](program_peer/metal__bf16__cast__strided__r1_c63.json) | 0.254 / 0.1981 |
| `metal/bf16/cast/strided/r1_c64` | correct | [passed](operator_peer/metal__bf16__cast__strided__r1_c64.json) | 0.2242 / 0.2013 | [passed](program_peer/metal__bf16__cast__strided__r1_c64.json) | 0.2524 / 0.1878 |
| `metal/bf16/cast/strided/r1_c65` | correct | [passed](operator_peer/metal__bf16__cast__strided__r1_c65.json) | 0.2006 / 0.206 | [passed](program_peer/metal__bf16__cast__strided__r1_c65.json) | 0.2248 / 0.1873 |
| `metal/bf16/cast/strided/r7_c3` | correct | [passed](operator_peer/metal__bf16__cast__strided__r7_c3.json) | 0.2464 / 0.1951 | [passed](program_peer/metal__bf16__cast__strided__r7_c3.json) | 0.2368 / 0.1988 |
| `metal/bf16/cast/strided/r7_c7` | correct | [passed](operator_peer/metal__bf16__cast__strided__r7_c7.json) | 0.249 / 0.2138 | [passed](program_peer/metal__bf16__cast__strided__r7_c7.json) | 0.2588 / 0.2078 |
| `metal/bf16/cast/strided/r7_c31` | correct | [passed](operator_peer/metal__bf16__cast__strided__r7_c31.json) | 0.202 / 0.1798 | [passed](program_peer/metal__bf16__cast__strided__r7_c31.json) | 0.2564 / 0.1961 |
| `metal/bf16/cast/strided/r7_c32` | correct | [passed](operator_peer/metal__bf16__cast__strided__r7_c32.json) | 0.2466 / 0.2004 | [passed](program_peer/metal__bf16__cast__strided__r7_c32.json) | 0.2554 / 0.195 |
| `metal/bf16/cast/strided/r7_c33` | correct | [passed](operator_peer/metal__bf16__cast__strided__r7_c33.json) | 0.246 / 0.2052 | [passed](program_peer/metal__bf16__cast__strided__r7_c33.json) | 0.2576 / 0.1998 |
| `metal/bf16/cast/strided/r7_c63` | correct | [passed](operator_peer/metal__bf16__cast__strided__r7_c63.json) | 0.2006 / 0.1954 | [passed](program_peer/metal__bf16__cast__strided__r7_c63.json) | 0.2562 / 0.1988 |
| `metal/bf16/cast/strided/r7_c64` | correct | [passed](operator_peer/metal__bf16__cast__strided__r7_c64.json) | 0.2524 / 0.2039 | [passed](program_peer/metal__bf16__cast__strided__r7_c64.json) | 0.2572 / 0.2089 |
| `metal/bf16/cast/strided/r7_c65` | correct | [passed](operator_peer/metal__bf16__cast__strided__r7_c65.json) | 0.2588 / 0.1923 | [passed](program_peer/metal__bf16__cast__strided__r7_c65.json) | 0.26 / 0.2074 |
| `metal/bf16/cast/strided/r32_c3` | correct | [passed](operator_peer/metal__bf16__cast__strided__r32_c3.json) | 0.2034 / 0.189 | [passed](program_peer/metal__bf16__cast__strided__r32_c3.json) | 0.2626 / 0.1987 |
| `metal/bf16/cast/strided/r32_c7` | correct | [passed](operator_peer/metal__bf16__cast__strided__r32_c7.json) | 0.2576 / 0.2035 | [passed](program_peer/metal__bf16__cast__strided__r32_c7.json) | 0.2632 / 0.2074 |
| `metal/bf16/cast/strided/r32_c31` | correct | [passed](operator_peer/metal__bf16__cast__strided__r32_c31.json) | 0.277 / 0.2119 | [passed](program_peer/metal__bf16__cast__strided__r32_c31.json) | 0.2764 / 0.2219 |
| `metal/bf16/cast/strided/r32_c32` | correct | [passed](operator_peer/metal__bf16__cast__strided__r32_c32.json) | 0.2726 / 0.2138 | [passed](program_peer/metal__bf16__cast__strided__r32_c32.json) | 0.2828 / 0.2085 |
| `metal/bf16/cast/strided/r32_c33` | correct | [passed](operator_peer/metal__bf16__cast__strided__r32_c33.json) | 0.2736 / 0.2125 | [passed](program_peer/metal__bf16__cast__strided__r32_c33.json) | 0.2832 / 0.2122 |
| `metal/bf16/cast/strided/r32_c63` | correct | [passed](operator_peer/metal__bf16__cast__strided__r32_c63.json) | 0.2854 / 0.2207 | [passed](program_peer/metal__bf16__cast__strided__r32_c63.json) | 0.2942 / 0.2206 |
| `metal/bf16/cast/strided/r32_c64` | correct | [passed](operator_peer/metal__bf16__cast__strided__r32_c64.json) | 0.277 / 0.2216 | [passed](program_peer/metal__bf16__cast__strided__r32_c64.json) | 0.2916 / 0.2199 |
| `metal/bf16/cast/strided/r32_c65` | correct | [passed](operator_peer/metal__bf16__cast__strided__r32_c65.json) | 0.291 / 0.2455 | [passed](program_peer/metal__bf16__cast__strided__r32_c65.json) | 0.291 / 0.2221 |
| `metal/bf16/cast/strided/r129_c3` | correct | [passed](operator_peer/metal__bf16__cast__strided__r129_c3.json) | 0.2784 / 0.2031 | [passed](program_peer/metal__bf16__cast__strided__r129_c3.json) | 0.2822 / 0.2004 |
| `metal/bf16/cast/strided/r129_c7` | correct | [passed](operator_peer/metal__bf16__cast__strided__r129_c7.json) | 0.3032 / 0.2091 | [passed](program_peer/metal__bf16__cast__strided__r129_c7.json) | 0.308 / 0.2107 |
| `metal/bf16/cast/strided/r129_c31` | correct | [passed](operator_peer/metal__bf16__cast__strided__r129_c31.json) | 0.3444 / 0.2377 | [passed](program_peer/metal__bf16__cast__strided__r129_c31.json) | 0.3452 / 0.2423 |
| `metal/bf16/cast/strided/r129_c32` | correct | [passed](operator_peer/metal__bf16__cast__strided__r129_c32.json) | 0.3394 / 0.2399 | [passed](program_peer/metal__bf16__cast__strided__r129_c32.json) | 0.3536 / 0.244 |
| `metal/bf16/cast/strided/r129_c33` | correct | [passed](operator_peer/metal__bf16__cast__strided__r129_c33.json) | 0.3474 / 0.24 | [passed](program_peer/metal__bf16__cast__strided__r129_c33.json) | 0.3492 / 0.2433 |
| `metal/bf16/cast/strided/r129_c63` | correct | [passed](operator_peer/metal__bf16__cast__strided__r129_c63.json) | 0.376 / 0.295 | [passed](program_peer/metal__bf16__cast__strided__r129_c63.json) | 0.382 / 0.2757 |
| `metal/bf16/cast/strided/r129_c64` | correct | [passed](operator_peer/metal__bf16__cast__strided__r129_c64.json) | 0.3762 / 0.2786 | [passed](program_peer/metal__bf16__cast__strided__r129_c64.json) | 0.3794 / 0.2925 |
| `metal/bf16/cast/strided/r129_c65` | correct | [passed](operator_peer/metal__bf16__cast__strided__r129_c65.json) | 0.3852 / 0.3081 | [passed](program_peer/metal__bf16__cast__strided__r129_c65.json) | 0.3914 / 0.2843 |
| `metal/bf16/cast/offset/r1_c3` | correct | [passed](operator_peer/metal__bf16__cast__offset__r1_c3.json) | 0.2486 / 0.1979 | [passed](program_peer/metal__bf16__cast__offset__r1_c3.json) | 0.2508 / 0.1981 |
| `metal/bf16/cast/offset/r1_c7` | correct | [passed](operator_peer/metal__bf16__cast__offset__r1_c7.json) | 0.2416 / 0.1866 | [passed](program_peer/metal__bf16__cast__offset__r1_c7.json) | 0.2514 / 0.2216 |
| `metal/bf16/cast/offset/r1_c31` | correct | [passed](operator_peer/metal__bf16__cast__offset__r1_c31.json) | 0.243 / 0.1959 | [passed](program_peer/metal__bf16__cast__offset__r1_c31.json) | 0.2426 / 0.1906 |
| `metal/bf16/cast/offset/r1_c32` | correct | [passed](operator_peer/metal__bf16__cast__offset__r1_c32.json) | 0.1902 / 0.1885 | [passed](program_peer/metal__bf16__cast__offset__r1_c32.json) | 0.2474 / 0.2046 |
| `metal/bf16/cast/offset/r1_c33` | correct | [passed](operator_peer/metal__bf16__cast__offset__r1_c33.json) | 0.2422 / 0.1953 | [passed](program_peer/metal__bf16__cast__offset__r1_c33.json) | 0.2526 / 0.1931 |
| `metal/bf16/cast/offset/r1_c63` | correct | [passed](operator_peer/metal__bf16__cast__offset__r1_c63.json) | 0.2448 / 0.1987 | [passed](program_peer/metal__bf16__cast__offset__r1_c63.json) | 0.2508 / 0.1929 |
| `metal/bf16/cast/offset/r1_c64` | correct | [passed](operator_peer/metal__bf16__cast__offset__r1_c64.json) | 0.2438 / 0.194 | [passed](program_peer/metal__bf16__cast__offset__r1_c64.json) | 0.2514 / 0.1936 |
| `metal/bf16/cast/offset/r1_c65` | correct | [passed](operator_peer/metal__bf16__cast__offset__r1_c65.json) | 0.1984 / 0.1925 | [passed](program_peer/metal__bf16__cast__offset__r1_c65.json) | 0.2478 / 0.1938 |
| `metal/bf16/cast/offset/r7_c3` | correct | [passed](operator_peer/metal__bf16__cast__offset__r7_c3.json) | 0.25 / 0.1926 | [passed](program_peer/metal__bf16__cast__offset__r7_c3.json) | 0.2538 / 0.1885 |
| `metal/bf16/cast/offset/r7_c7` | correct | [passed](operator_peer/metal__bf16__cast__offset__r7_c7.json) | 0.2466 / 0.222 | [passed](program_peer/metal__bf16__cast__offset__r7_c7.json) | 0.253 / 0.2036 |
| `metal/bf16/cast/offset/r7_c31` | correct | [passed](operator_peer/metal__bf16__cast__offset__r7_c31.json) | 0.2504 / 0.1977 | [passed](program_peer/metal__bf16__cast__offset__r7_c31.json) | 0.2506 / 0.2071 |
| `metal/bf16/cast/offset/r7_c32` | correct | [passed](operator_peer/metal__bf16__cast__offset__r7_c32.json) | 0.2476 / 0.2036 | [passed](program_peer/metal__bf16__cast__offset__r7_c32.json) | 0.254 / 0.2065 |
| `metal/bf16/cast/offset/r7_c33` | correct | [passed](operator_peer/metal__bf16__cast__offset__r7_c33.json) | 0.2506 / 0.2098 | [passed](program_peer/metal__bf16__cast__offset__r7_c33.json) | 0.258 / 0.2074 |
| `metal/bf16/cast/offset/r7_c63` | correct | [passed](operator_peer/metal__bf16__cast__offset__r7_c63.json) | 0.2562 / 0.2009 | [passed](program_peer/metal__bf16__cast__offset__r7_c63.json) | 0.2586 / 0.1929 |
| `metal/bf16/cast/offset/r7_c64` | correct | [passed](operator_peer/metal__bf16__cast__offset__r7_c64.json) | 0.2512 / 0.2002 | [passed](program_peer/metal__bf16__cast__offset__r7_c64.json) | 0.265 / 0.2076 |
| `metal/bf16/cast/offset/r7_c65` | correct | [passed](operator_peer/metal__bf16__cast__offset__r7_c65.json) | 0.2474 / 0.1881 | [passed](program_peer/metal__bf16__cast__offset__r7_c65.json) | 0.2576 / 0.2006 |
| `metal/bf16/cast/offset/r32_c3` | correct | [passed](operator_peer/metal__bf16__cast__offset__r32_c3.json) | 0.2566 / 0.1983 | [passed](program_peer/metal__bf16__cast__offset__r32_c3.json) | 0.2626 / 0.1941 |
| `metal/bf16/cast/offset/r32_c7` | correct | [passed](operator_peer/metal__bf16__cast__offset__r32_c7.json) | 0.2634 / 0.1961 | [passed](program_peer/metal__bf16__cast__offset__r32_c7.json) | 0.258 / 0.2062 |
| `metal/bf16/cast/offset/r32_c31` | correct | [passed](operator_peer/metal__bf16__cast__offset__r32_c31.json) | 0.2808 / 0.2175 | [passed](program_peer/metal__bf16__cast__offset__r32_c31.json) | 0.2848 / 0.2107 |
| `metal/bf16/cast/offset/r32_c32` | correct | [passed](operator_peer/metal__bf16__cast__offset__r32_c32.json) | 0.278 / 0.2013 | [passed](program_peer/metal__bf16__cast__offset__r32_c32.json) | 0.2828 / 0.2109 |
| `metal/bf16/cast/offset/r32_c33` | correct | [passed](operator_peer/metal__bf16__cast__offset__r32_c33.json) | 0.2754 / 0.2031 | [passed](program_peer/metal__bf16__cast__offset__r32_c33.json) | 0.2832 / 0.2106 |
| `metal/bf16/cast/offset/r32_c63` | correct | [passed](operator_peer/metal__bf16__cast__offset__r32_c63.json) | 0.287 / 0.2237 | [passed](program_peer/metal__bf16__cast__offset__r32_c63.json) | 0.292 / 0.2017 |
| `metal/bf16/cast/offset/r32_c64` | correct | [passed](operator_peer/metal__bf16__cast__offset__r32_c64.json) | 0.2846 / 0.247 | [passed](program_peer/metal__bf16__cast__offset__r32_c64.json) | 0.2886 / 0.2146 |
| `metal/bf16/cast/offset/r32_c65` | correct | [passed](operator_peer/metal__bf16__cast__offset__r32_c65.json) | 0.2886 / 0.2188 | [passed](program_peer/metal__bf16__cast__offset__r32_c65.json) | 0.2936 / 0.2122 |
| `metal/bf16/cast/offset/r129_c3` | correct | [passed](operator_peer/metal__bf16__cast__offset__r129_c3.json) | 0.2782 / 0.2016 | [passed](program_peer/metal__bf16__cast__offset__r129_c3.json) | 0.2768 / 0.2143 |
| `metal/bf16/cast/offset/r129_c7` | correct | [passed](operator_peer/metal__bf16__cast__offset__r129_c7.json) | 0.2996 / 0.2183 | [passed](program_peer/metal__bf16__cast__offset__r129_c7.json) | 0.282 / 0.2161 |
| `metal/bf16/cast/offset/r129_c31` | correct | [passed](operator_peer/metal__bf16__cast__offset__r129_c31.json) | 0.3424 / 0.2376 | [passed](program_peer/metal__bf16__cast__offset__r129_c31.json) | 0.3472 / 0.2399 |
| `metal/bf16/cast/offset/r129_c32` | correct | [passed](operator_peer/metal__bf16__cast__offset__r129_c32.json) | 0.342 / 0.2404 | [passed](program_peer/metal__bf16__cast__offset__r129_c32.json) | 0.3498 / 0.2415 |
| `metal/bf16/cast/offset/r129_c33` | correct | [passed](operator_peer/metal__bf16__cast__offset__r129_c33.json) | 0.345 / 0.2411 | [passed](program_peer/metal__bf16__cast__offset__r129_c33.json) | 0.3464 / 0.243 |
| `metal/bf16/cast/offset/r129_c63` | correct | [passed](operator_peer/metal__bf16__cast__offset__r129_c63.json) | 0.3798 / 0.2815 | [passed](program_peer/metal__bf16__cast__offset__r129_c63.json) | 0.3806 / 0.282 |
| `metal/bf16/cast/offset/r129_c64` | correct | [passed](operator_peer/metal__bf16__cast__offset__r129_c64.json) | 0.374 / 0.2807 | [passed](program_peer/metal__bf16__cast__offset__r129_c64.json) | 0.3778 / 0.2822 |
| `metal/bf16/cast/offset/r129_c65` | correct | [passed](operator_peer/metal__bf16__cast__offset__r129_c65.json) | 0.384 / 0.3011 | [passed](program_peer/metal__bf16__cast__offset__r129_c65.json) | 0.3856 / 0.2974 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
