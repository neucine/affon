# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `e04002043f72048452b5632ced82126000215260cd0134bd33dfb13e8a741572`  
Binary SHA-256: `8456905a89b6ac8d742cc5963122d969d09cc8f9cea4659b709e7c51f1dfcd93`  
Timing: 10 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 256
- Numerical capability passed: 192
- Operator parity and timing passed: 192
- Program parity and timing passed: 192
- Probe failures or incorrect: 64
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/add/strided/strided_r3_c5` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r3_c5.json) | 0.2503 / 0.1952 | [passed](program_peer/metal__f32__add__strided__strided_r3_c5.json) | 0.227 / 0.1842 |
| `metal/f32/add/strided/strided_r3_c32` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r3_c32.json) | 0.2321 / 0.1939 | [passed](program_peer/metal__f32__add__strided__strided_r3_c32.json) | 0.2454 / 0.1901 |
| `metal/f32/add/strided/strided_r3_c65` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r3_c65.json) | 0.2338 / 0.1951 | [passed](program_peer/metal__f32__add__strided__strided_r3_c65.json) | 0.2381 / 0.1875 |
| `metal/f32/add/strided/strided_r3_c129` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r3_c129.json) | 0.2362 / 0.1966 | [passed](program_peer/metal__f32__add__strided__strided_r3_c129.json) | 0.2517 / 0.1925 |
| `metal/f32/add/strided/strided_r16_c5` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r16_c5.json) | 0.2152 / 0.1843 | [passed](program_peer/metal__f32__add__strided__strided_r16_c5.json) | 0.251 / 0.197 |
| `metal/f32/add/strided/strided_r16_c32` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r16_c32.json) | 0.2355 / 0.1885 | [passed](program_peer/metal__f32__add__strided__strided_r16_c32.json) | 0.2606 / 0.1925 |
| `metal/f32/add/strided/strided_r16_c65` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r16_c65.json) | 0.2648 / 0.1995 | [passed](program_peer/metal__f32__add__strided__strided_r16_c65.json) | 0.2747 / 0.2038 |
| `metal/f32/add/strided/strided_r16_c129` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r16_c129.json) | 0.2814 / 0.2171 | [passed](program_peer/metal__f32__add__strided__strided_r16_c129.json) | 0.2825 / 0.216 |
| `metal/f32/add/strided/strided_r33_c5` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r33_c5.json) | 0.25 / 0.1989 | [passed](program_peer/metal__f32__add__strided__strided_r33_c5.json) | 0.2578 / 0.1918 |
| `metal/f32/add/strided/strided_r33_c32` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r33_c32.json) | 0.2757 / 0.2042 | [passed](program_peer/metal__f32__add__strided__strided_r33_c32.json) | 0.2816 / 0.202 |
| `metal/f32/add/strided/strided_r33_c65` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r33_c65.json) | 0.2907 / 0.2197 | [passed](program_peer/metal__f32__add__strided__strided_r33_c65.json) | 0.2967 / 0.2215 |
| `metal/f32/add/strided/strided_r33_c129` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r33_c129.json) | 0.3081 / 0.2364 | [passed](program_peer/metal__f32__add__strided__strided_r33_c129.json) | 0.3133 / 0.2397 |
| `metal/f32/add/strided/strided_r128_c5` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r128_c5.json) | 0.2911 / 0.2009 | [passed](program_peer/metal__f32__add__strided__strided_r128_c5.json) | 0.2956 / 0.2059 |
| `metal/f32/add/strided/strided_r128_c32` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r128_c32.json) | 0.339 / 0.24 | [passed](program_peer/metal__f32__add__strided__strided_r128_c32.json) | 0.3481 / 0.2396 |
| `metal/f32/add/strided/strided_r128_c65` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r128_c65.json) | 0.3818 / 0.2744 | [passed](program_peer/metal__f32__add__strided__strided_r128_c65.json) | 0.386 / 0.2973 |
| `metal/f32/add/strided/strided_r128_c129` | correct | [passed](operator_peer/metal__f32__add__strided__strided_r128_c129.json) | 0.4574 / 0.4116 | [passed](program_peer/metal__f32__add__strided__strided_r128_c129.json) | 0.4582 / 0.413 |
| `metal/f32/add/offset/offset_r3_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r3_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r3_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r3_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r16_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r16_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r16_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r16_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r33_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r33_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r33_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r33_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r128_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r128_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r128_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/add/offset/offset_r128_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/strided/strided_r3_c5` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r3_c5.json) | 0.2437 / 0.1848 | [passed](program_peer/metal__f32__sub__strided__strided_r3_c5.json) | 0.1959 / 0.1894 |
| `metal/f32/sub/strided/strided_r3_c32` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r3_c32.json) | 0.2428 / 0.1878 | [passed](program_peer/metal__f32__sub__strided__strided_r3_c32.json) | 0.2482 / 0.191 |
| `metal/f32/sub/strided/strided_r3_c65` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r3_c65.json) | 0.2444 / 0.188 | [passed](program_peer/metal__f32__sub__strided__strided_r3_c65.json) | 0.2488 / 0.1867 |
| `metal/f32/sub/strided/strided_r3_c129` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r3_c129.json) | 0.2431 / 0.189 | [passed](program_peer/metal__f32__sub__strided__strided_r3_c129.json) | 0.2514 / 0.1956 |
| `metal/f32/sub/strided/strided_r16_c5` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r16_c5.json) | 0.2412 / 0.195 | [passed](program_peer/metal__f32__sub__strided__strided_r16_c5.json) | 0.252 / 0.1924 |
| `metal/f32/sub/strided/strided_r16_c32` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r16_c32.json) | 0.2425 / 0.1999 | [passed](program_peer/metal__f32__sub__strided__strided_r16_c32.json) | 0.2641 / 0.1983 |
| `metal/f32/sub/strided/strided_r16_c65` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r16_c65.json) | 0.269 / 0.206 | [passed](program_peer/metal__f32__sub__strided__strided_r16_c65.json) | 0.2747 / 0.1985 |
| `metal/f32/sub/strided/strided_r16_c129` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r16_c129.json) | 0.2826 / 0.2125 | [passed](program_peer/metal__f32__sub__strided__strided_r16_c129.json) | 0.2794 / 0.2186 |
| `metal/f32/sub/strided/strided_r33_c5` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r33_c5.json) | 0.2571 / 0.1914 | [passed](program_peer/metal__f32__sub__strided__strided_r33_c5.json) | 0.2586 / 0.1884 |
| `metal/f32/sub/strided/strided_r33_c32` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r33_c32.json) | 0.2733 / 0.202 | [passed](program_peer/metal__f32__sub__strided__strided_r33_c32.json) | 0.2826 / 0.2061 |
| `metal/f32/sub/strided/strided_r33_c65` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r33_c65.json) | 0.2828 / 0.2184 | [passed](program_peer/metal__f32__sub__strided__strided_r33_c65.json) | 0.2955 / 0.2198 |
| `metal/f32/sub/strided/strided_r33_c129` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r33_c129.json) | 0.3057 / 0.2371 | [passed](program_peer/metal__f32__sub__strided__strided_r33_c129.json) | 0.3061 / 0.2389 |
| `metal/f32/sub/strided/strided_r128_c5` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r128_c5.json) | 0.2913 / 0.2092 | [passed](program_peer/metal__f32__sub__strided__strided_r128_c5.json) | 0.2931 / 0.2057 |
| `metal/f32/sub/strided/strided_r128_c32` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r128_c32.json) | 0.3388 / 0.2378 | [passed](program_peer/metal__f32__sub__strided__strided_r128_c32.json) | 0.2987 / 0.2385 |
| `metal/f32/sub/strided/strided_r128_c65` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r128_c65.json) | 0.3649 / 0.3057 | [passed](program_peer/metal__f32__sub__strided__strided_r128_c65.json) | 0.3946 / 0.282 |
| `metal/f32/sub/strided/strided_r128_c129` | correct | [passed](operator_peer/metal__f32__sub__strided__strided_r128_c129.json) | 0.4554 / 0.4087 | [passed](program_peer/metal__f32__sub__strided__strided_r128_c129.json) | 0.4584 / 0.4116 |
| `metal/f32/sub/offset/offset_r3_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r3_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r3_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r3_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r16_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r16_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r16_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r16_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r33_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r33_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r33_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r33_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r128_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r128_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r128_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/sub/offset/offset_r128_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/strided/strided_r3_c5` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r3_c5.json) | 0.2408 / 0.1895 | [passed](program_peer/metal__f32__mul__strided__strided_r3_c5.json) | 0.2491 / 0.1887 |
| `metal/f32/mul/strided/strided_r3_c32` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r3_c32.json) | 0.1931 / 0.1911 | [passed](program_peer/metal__f32__mul__strided__strided_r3_c32.json) | 0.195 / 0.1886 |
| `metal/f32/mul/strided/strided_r3_c65` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r3_c65.json) | 0.2076 / 0.1899 | [passed](program_peer/metal__f32__mul__strided__strided_r3_c65.json) | 0.2451 / 0.1863 |
| `metal/f32/mul/strided/strided_r3_c129` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r3_c129.json) | 0.2466 / 0.1973 | [passed](program_peer/metal__f32__mul__strided__strided_r3_c129.json) | 0.2485 / 0.191 |
| `metal/f32/mul/strided/strided_r16_c5` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r16_c5.json) | 0.242 / 0.1837 | [passed](program_peer/metal__f32__mul__strided__strided_r16_c5.json) | 0.2518 / 0.1977 |
| `metal/f32/mul/strided/strided_r16_c32` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r16_c32.json) | 0.2539 / 0.1922 | [passed](program_peer/metal__f32__mul__strided__strided_r16_c32.json) | 0.2609 / 0.1945 |
| `metal/f32/mul/strided/strided_r16_c65` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r16_c65.json) | 0.2673 / 0.2061 | [passed](program_peer/metal__f32__mul__strided__strided_r16_c65.json) | 0.2515 / 0.1983 |
| `metal/f32/mul/strided/strided_r16_c129` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r16_c129.json) | 0.2816 / 0.2147 | [passed](program_peer/metal__f32__mul__strided__strided_r16_c129.json) | 0.2606 / 0.2068 |
| `metal/f32/mul/strided/strided_r33_c5` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r33_c5.json) | 0.2528 / 0.1863 | [passed](program_peer/metal__f32__mul__strided__strided_r33_c5.json) | 0.2287 / 0.1896 |
| `metal/f32/mul/strided/strided_r33_c32` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r33_c32.json) | 0.2686 / 0.2065 | [passed](program_peer/metal__f32__mul__strided__strided_r33_c32.json) | 0.2861 / 0.2026 |
| `metal/f32/mul/strided/strided_r33_c65` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r33_c65.json) | 0.2909 / 0.2214 | [passed](program_peer/metal__f32__mul__strided__strided_r33_c65.json) | 0.2961 / 0.219 |
| `metal/f32/mul/strided/strided_r33_c129` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r33_c129.json) | 0.2797 / 0.2268 | [passed](program_peer/metal__f32__mul__strided__strided_r33_c129.json) | 0.3079 / 0.2378 |
| `metal/f32/mul/strided/strided_r128_c5` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r128_c5.json) | 0.2841 / 0.1968 | [passed](program_peer/metal__f32__mul__strided__strided_r128_c5.json) | 0.2976 / 0.2008 |
| `metal/f32/mul/strided/strided_r128_c32` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r128_c32.json) | 0.3297 / 0.2293 | [passed](program_peer/metal__f32__mul__strided__strided_r128_c32.json) | 0.3495 / 0.2366 |
| `metal/f32/mul/strided/strided_r128_c65` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r128_c65.json) | 0.3702 / 0.2893 | [passed](program_peer/metal__f32__mul__strided__strided_r128_c65.json) | 0.3871 / 0.3191 |
| `metal/f32/mul/strided/strided_r128_c129` | correct | [passed](operator_peer/metal__f32__mul__strided__strided_r128_c129.json) | 0.4404 / 0.4042 | [passed](program_peer/metal__f32__mul__strided__strided_r128_c129.json) | 0.4341 / 0.4152 |
| `metal/f32/mul/offset/offset_r3_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r3_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r3_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r3_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r16_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r16_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r16_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r16_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r33_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r33_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r33_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r33_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r128_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r128_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r128_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/mul/offset/offset_r128_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/strided/strided_r3_c5` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r3_c5.json) | 0.2419 / 0.1839 | [passed](program_peer/metal__f32__div__strided__strided_r3_c5.json) | 0.2529 / 0.1816 |
| `metal/f32/div/strided/strided_r3_c32` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r3_c32.json) | 0.2172 / 0.1966 | [passed](program_peer/metal__f32__div__strided__strided_r3_c32.json) | 0.2412 / 0.1883 |
| `metal/f32/div/strided/strided_r3_c65` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r3_c65.json) | 0.2115 / 0.1859 | [passed](program_peer/metal__f32__div__strided__strided_r3_c65.json) | 0.237 / 0.1946 |
| `metal/f32/div/strided/strided_r3_c129` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r3_c129.json) | 0.2317 / 0.1953 | [passed](program_peer/metal__f32__div__strided__strided_r3_c129.json) | 0.2495 / 0.1919 |
| `metal/f32/div/strided/strided_r16_c5` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r16_c5.json) | 0.2465 / 0.1941 | [passed](program_peer/metal__f32__div__strided__strided_r16_c5.json) | 0.2543 / 0.1891 |
| `metal/f32/div/strided/strided_r16_c32` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r16_c32.json) | 0.2262 / 0.1915 | [passed](program_peer/metal__f32__div__strided__strided_r16_c32.json) | 0.2053 / 0.1894 |
| `metal/f32/div/strided/strided_r16_c65` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r16_c65.json) | 0.2666 / 0.2017 | [passed](program_peer/metal__f32__div__strided__strided_r16_c65.json) | 0.2715 / 0.1916 |
| `metal/f32/div/strided/strided_r16_c129` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r16_c129.json) | 0.2605 / 0.215 | [passed](program_peer/metal__f32__div__strided__strided_r16_c129.json) | 0.2589 / 0.2138 |
| `metal/f32/div/strided/strided_r33_c5` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r33_c5.json) | 0.2498 / 0.1918 | [passed](program_peer/metal__f32__div__strided__strided_r33_c5.json) | 0.2468 / 0.1882 |
| `metal/f32/div/strided/strided_r33_c32` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r33_c32.json) | 0.2714 / 0.2063 | [passed](program_peer/metal__f32__div__strided__strided_r33_c32.json) | 0.2744 / 0.2066 |
| `metal/f32/div/strided/strided_r33_c65` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r33_c65.json) | 0.2916 / 0.2194 | [passed](program_peer/metal__f32__div__strided__strided_r33_c65.json) | 0.2967 / 0.2216 |
| `metal/f32/div/strided/strided_r33_c129` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r33_c129.json) | 0.3033 / 0.2371 | [passed](program_peer/metal__f32__div__strided__strided_r33_c129.json) | 0.2835 / 0.2243 |
| `metal/f32/div/strided/strided_r128_c5` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r128_c5.json) | 0.2922 / 0.1982 | [passed](program_peer/metal__f32__div__strided__strided_r128_c5.json) | 0.2815 / 0.1897 |
| `metal/f32/div/strided/strided_r128_c32` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r128_c32.json) | 0.334 / 0.234 | [passed](program_peer/metal__f32__div__strided__strided_r128_c32.json) | 0.3533 / 0.236 |
| `metal/f32/div/strided/strided_r128_c65` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r128_c65.json) | 0.368 / 0.2842 | [passed](program_peer/metal__f32__div__strided__strided_r128_c65.json) | 0.3849 / 0.2776 |
| `metal/f32/div/strided/strided_r128_c129` | correct | [passed](operator_peer/metal__f32__div__strided__strided_r128_c129.json) | 0.447 / 0.3869 | [passed](program_peer/metal__f32__div__strided__strided_r128_c129.json) | 0.459 / 0.4118 |
| `metal/f32/div/offset/offset_r3_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r3_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r3_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r3_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r16_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r16_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r16_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r16_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r33_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r33_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r33_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r33_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r128_c5` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r128_c32` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r128_c65` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/div/offset/offset_r128_c129` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/add/strided/strided_r3_c5` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r3_c5.json) | 0.2434 / 0.2007 | [passed](program_peer/metal__bf16__add__strided__strided_r3_c5.json) | 0.2526 / 0.1925 |
| `metal/bf16/add/strided/strided_r3_c32` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r3_c32.json) | 0.1987 / 0.1848 | [passed](program_peer/metal__bf16__add__strided__strided_r3_c32.json) | 0.2546 / 0.1892 |
| `metal/bf16/add/strided/strided_r3_c65` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r3_c65.json) | 0.2324 / 0.1998 | [passed](program_peer/metal__bf16__add__strided__strided_r3_c65.json) | 0.2546 / 0.1957 |
| `metal/bf16/add/strided/strided_r3_c129` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r3_c129.json) | 0.2468 / 0.2 | [passed](program_peer/metal__bf16__add__strided__strided_r3_c129.json) | 0.2645 / 0.1949 |
| `metal/bf16/add/strided/strided_r16_c5` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r16_c5.json) | 0.2465 / 0.1907 | [passed](program_peer/metal__bf16__add__strided__strided_r16_c5.json) | 0.2626 / 0.1903 |
| `metal/bf16/add/strided/strided_r16_c32` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r16_c32.json) | 0.2435 / 0.1973 | [passed](program_peer/metal__bf16__add__strided__strided_r16_c32.json) | 0.2772 / 0.1956 |
| `metal/bf16/add/strided/strided_r16_c65` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r16_c65.json) | 0.2679 / 0.1973 | [passed](program_peer/metal__bf16__add__strided__strided_r16_c65.json) | 0.2982 / 0.2065 |
| `metal/bf16/add/strided/strided_r16_c129` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r16_c129.json) | 0.2703 / 0.2219 | [passed](program_peer/metal__bf16__add__strided__strided_r16_c129.json) | 0.3214 / 0.2188 |
| `metal/bf16/add/strided/strided_r33_c5` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r33_c5.json) | 0.2559 / 0.191 | [passed](program_peer/metal__bf16__add__strided__strided_r33_c5.json) | 0.2529 / 0.1909 |
| `metal/bf16/add/strided/strided_r33_c32` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r33_c32.json) | 0.2751 / 0.2086 | [passed](program_peer/metal__bf16__add__strided__strided_r33_c32.json) | 0.2749 / 0.1954 |
| `metal/bf16/add/strided/strided_r33_c65` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r33_c65.json) | 0.285 / 0.2182 | [passed](program_peer/metal__bf16__add__strided__strided_r33_c65.json) | 0.3211 / 0.2128 |
| `metal/bf16/add/strided/strided_r33_c129` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r33_c129.json) | 0.3058 / 0.2346 | [passed](program_peer/metal__bf16__add__strided__strided_r33_c129.json) | 0.3397 / 0.2385 |
| `metal/bf16/add/strided/strided_r128_c5` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r128_c5.json) | 0.2901 / 0.199 | [passed](program_peer/metal__bf16__add__strided__strided_r128_c5.json) | 0.3114 / 0.2019 |
| `metal/bf16/add/strided/strided_r128_c32` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r128_c32.json) | 0.2758 / 0.2402 | [passed](program_peer/metal__bf16__add__strided__strided_r128_c32.json) | 0.3224 / 0.2319 |
| `metal/bf16/add/strided/strided_r128_c65` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r128_c65.json) | 0.385 / 0.2976 | [passed](program_peer/metal__bf16__add__strided__strided_r128_c65.json) | 0.4075 / 0.3269 |
| `metal/bf16/add/strided/strided_r128_c129` | correct | [passed](operator_peer/metal__bf16__add__strided__strided_r128_c129.json) | 0.4534 / 0.4098 | [passed](program_peer/metal__bf16__add__strided__strided_r128_c129.json) | 0.511 / 0.4122 |
| `metal/bf16/add/offset/offset_r3_c5` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r3_c5.json) | 0.2448 / 0.1937 | [passed](program_peer/metal__bf16__add__offset__offset_r3_c5.json) | 0.2558 / 0.1925 |
| `metal/bf16/add/offset/offset_r3_c32` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r3_c32.json) | 0.2435 / 0.1874 | [passed](program_peer/metal__bf16__add__offset__offset_r3_c32.json) | 0.255 / 0.1927 |
| `metal/bf16/add/offset/offset_r3_c65` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r3_c65.json) | 0.2427 / 0.1869 | [passed](program_peer/metal__bf16__add__offset__offset_r3_c65.json) | 0.2579 / 0.1879 |
| `metal/bf16/add/offset/offset_r3_c129` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r3_c129.json) | 0.245 / 0.1905 | [passed](program_peer/metal__bf16__add__offset__offset_r3_c129.json) | 0.2501 / 0.1905 |
| `metal/bf16/add/offset/offset_r16_c5` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r16_c5.json) | 0.2492 / 0.1909 | [passed](program_peer/metal__bf16__add__offset__offset_r16_c5.json) | 0.2611 / 0.1927 |
| `metal/bf16/add/offset/offset_r16_c32` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r16_c32.json) | 0.2596 / 0.2014 | [passed](program_peer/metal__bf16__add__offset__offset_r16_c32.json) | 0.273 / 0.1933 |
| `metal/bf16/add/offset/offset_r16_c65` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r16_c65.json) | 0.2694 / 0.2004 | [passed](program_peer/metal__bf16__add__offset__offset_r16_c65.json) | 0.2566 / 0.2061 |
| `metal/bf16/add/offset/offset_r16_c129` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r16_c129.json) | 0.2796 / 0.2137 | [passed](program_peer/metal__bf16__add__offset__offset_r16_c129.json) | 0.3187 / 0.2188 |
| `metal/bf16/add/offset/offset_r33_c5` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r33_c5.json) | 0.2568 / 0.1878 | [passed](program_peer/metal__bf16__add__offset__offset_r33_c5.json) | 0.2724 / 0.1923 |
| `metal/bf16/add/offset/offset_r33_c32` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r33_c32.json) | 0.2796 / 0.2074 | [passed](program_peer/metal__bf16__add__offset__offset_r33_c32.json) | 0.3162 / 0.2077 |
| `metal/bf16/add/offset/offset_r33_c65` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r33_c65.json) | 0.2798 / 0.2142 | [passed](program_peer/metal__bf16__add__offset__offset_r33_c65.json) | 0.329 / 0.2182 |
| `metal/bf16/add/offset/offset_r33_c129` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r33_c129.json) | 0.3024 / 0.2387 | [passed](program_peer/metal__bf16__add__offset__offset_r33_c129.json) | 0.2786 / 0.2372 |
| `metal/bf16/add/offset/offset_r128_c5` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r128_c5.json) | 0.2925 / 0.1993 | [passed](program_peer/metal__bf16__add__offset__offset_r128_c5.json) | 0.3217 / 0.2033 |
| `metal/bf16/add/offset/offset_r128_c32` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r128_c32.json) | 0.3171 / 0.2331 | [passed](program_peer/metal__bf16__add__offset__offset_r128_c32.json) | 0.3588 / 0.2375 |
| `metal/bf16/add/offset/offset_r128_c65` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r128_c65.json) | 0.3747 / 0.3291 | [passed](program_peer/metal__bf16__add__offset__offset_r128_c65.json) | 0.3795 / 0.3285 |
| `metal/bf16/add/offset/offset_r128_c129` | correct | [passed](operator_peer/metal__bf16__add__offset__offset_r128_c129.json) | 0.4595 / 0.4128 | [passed](program_peer/metal__bf16__add__offset__offset_r128_c129.json) | 0.4123 / 0.4141 |
| `metal/bf16/sub/strided/strided_r3_c5` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r3_c5.json) | 0.1971 / 0.1832 | [passed](program_peer/metal__bf16__sub__strided__strided_r3_c5.json) | 0.2601 / 0.1912 |
| `metal/bf16/sub/strided/strided_r3_c32` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r3_c32.json) | 0.2346 / 0.1904 | [passed](program_peer/metal__bf16__sub__strided__strided_r3_c32.json) | 0.2517 / 0.1859 |
| `metal/bf16/sub/strided/strided_r3_c65` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r3_c65.json) | 0.245 / 0.1865 | [passed](program_peer/metal__bf16__sub__strided__strided_r3_c65.json) | 0.2599 / 0.1962 |
| `metal/bf16/sub/strided/strided_r3_c129` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r3_c129.json) | 0.2492 / 0.1923 | [passed](program_peer/metal__bf16__sub__strided__strided_r3_c129.json) | 0.2239 / 0.1951 |
| `metal/bf16/sub/strided/strided_r16_c5` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r16_c5.json) | 0.2503 / 0.191 | [passed](program_peer/metal__bf16__sub__strided__strided_r16_c5.json) | 0.2487 / 0.1957 |
| `metal/bf16/sub/strided/strided_r16_c32` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r16_c32.json) | 0.2566 / 0.2002 | [passed](program_peer/metal__bf16__sub__strided__strided_r16_c32.json) | 0.2769 / 0.1962 |
| `metal/bf16/sub/strided/strided_r16_c65` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r16_c65.json) | 0.2715 / 0.2014 | [passed](program_peer/metal__bf16__sub__strided__strided_r16_c65.json) | 0.2602 / 0.1979 |
| `metal/bf16/sub/strided/strided_r16_c129` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r16_c129.json) | 0.279 / 0.2191 | [passed](program_peer/metal__bf16__sub__strided__strided_r16_c129.json) | 0.3129 / 0.212 |
| `metal/bf16/sub/strided/strided_r33_c5` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r33_c5.json) | 0.2557 / 0.1962 | [passed](program_peer/metal__bf16__sub__strided__strided_r33_c5.json) | 0.2696 / 0.1903 |
| `metal/bf16/sub/strided/strided_r33_c32` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r33_c32.json) | 0.2759 / 0.199 | [passed](program_peer/metal__bf16__sub__strided__strided_r33_c32.json) | 0.3001 / 0.1999 |
| `metal/bf16/sub/strided/strided_r33_c65` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r33_c65.json) | 0.2892 / 0.2172 | [passed](program_peer/metal__bf16__sub__strided__strided_r33_c65.json) | 0.3212 / 0.2198 |
| `metal/bf16/sub/strided/strided_r33_c129` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r33_c129.json) | 0.2893 / 0.236 | [passed](program_peer/metal__bf16__sub__strided__strided_r33_c129.json) | 0.3045 / 0.2392 |
| `metal/bf16/sub/strided/strided_r128_c5` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r128_c5.json) | 0.2925 / 0.2016 | [passed](program_peer/metal__bf16__sub__strided__strided_r128_c5.json) | 0.3228 / 0.2006 |
| `metal/bf16/sub/strided/strided_r128_c32` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r128_c32.json) | 0.3452 / 0.2371 | [passed](program_peer/metal__bf16__sub__strided__strided_r128_c32.json) | 0.346 / 0.2321 |
| `metal/bf16/sub/strided/strided_r128_c65` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r128_c65.json) | 0.3606 / 0.283 | [passed](program_peer/metal__bf16__sub__strided__strided_r128_c65.json) | 0.3825 / 0.2787 |
| `metal/bf16/sub/strided/strided_r128_c129` | correct | [passed](operator_peer/metal__bf16__sub__strided__strided_r128_c129.json) | 0.4363 / 0.4016 | [passed](program_peer/metal__bf16__sub__strided__strided_r128_c129.json) | 0.4276 / 0.4165 |
| `metal/bf16/sub/offset/offset_r3_c5` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r3_c5.json) | 0.2443 / 0.1913 | [passed](program_peer/metal__bf16__sub__offset__offset_r3_c5.json) | 0.2586 / 0.1983 |
| `metal/bf16/sub/offset/offset_r3_c32` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r3_c32.json) | 0.2389 / 0.1955 | [passed](program_peer/metal__bf16__sub__offset__offset_r3_c32.json) | 0.2553 / 0.189 |
| `metal/bf16/sub/offset/offset_r3_c65` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r3_c65.json) | 0.2442 / 0.1952 | [passed](program_peer/metal__bf16__sub__offset__offset_r3_c65.json) | 0.2473 / 0.194 |
| `metal/bf16/sub/offset/offset_r3_c129` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r3_c129.json) | 0.2333 / 0.1928 | [passed](program_peer/metal__bf16__sub__offset__offset_r3_c129.json) | 0.2631 / 0.1997 |
| `metal/bf16/sub/offset/offset_r16_c5` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r16_c5.json) | 0.2014 / 0.1904 | [passed](program_peer/metal__bf16__sub__offset__offset_r16_c5.json) | 0.2642 / 0.195 |
| `metal/bf16/sub/offset/offset_r16_c32` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r16_c32.json) | 0.2561 / 0.1943 | [passed](program_peer/metal__bf16__sub__offset__offset_r16_c32.json) | 0.2713 / 0.1982 |
| `metal/bf16/sub/offset/offset_r16_c65` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r16_c65.json) | 0.2496 / 0.1995 | [passed](program_peer/metal__bf16__sub__offset__offset_r16_c65.json) | 0.2491 / 0.1967 |
| `metal/bf16/sub/offset/offset_r16_c129` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r16_c129.json) | 0.2638 / 0.2059 | [passed](program_peer/metal__bf16__sub__offset__offset_r16_c129.json) | 0.3167 / 0.223 |
| `metal/bf16/sub/offset/offset_r33_c5` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r33_c5.json) | 0.2523 / 0.1943 | [passed](program_peer/metal__bf16__sub__offset__offset_r33_c5.json) | 0.2719 / 0.1898 |
| `metal/bf16/sub/offset/offset_r33_c32` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r33_c32.json) | 0.2759 / 0.2027 | [passed](program_peer/metal__bf16__sub__offset__offset_r33_c32.json) | 0.3165 / 0.2082 |
| `metal/bf16/sub/offset/offset_r33_c65` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r33_c65.json) | 0.2889 / 0.2203 | [passed](program_peer/metal__bf16__sub__offset__offset_r33_c65.json) | 0.3246 / 0.2218 |
| `metal/bf16/sub/offset/offset_r33_c129` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r33_c129.json) | 0.3034 / 0.2345 | [passed](program_peer/metal__bf16__sub__offset__offset_r33_c129.json) | 0.338 / 0.2386 |
| `metal/bf16/sub/offset/offset_r128_c5` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r128_c5.json) | 0.2916 / 0.2029 | [passed](program_peer/metal__bf16__sub__offset__offset_r128_c5.json) | 0.2599 / 0.2048 |
| `metal/bf16/sub/offset/offset_r128_c32` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r128_c32.json) | 0.3084 / 0.2367 | [passed](program_peer/metal__bf16__sub__offset__offset_r128_c32.json) | 0.3845 / 0.2347 |
| `metal/bf16/sub/offset/offset_r128_c65` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r128_c65.json) | 0.3819 / 0.2762 | [passed](program_peer/metal__bf16__sub__offset__offset_r128_c65.json) | 0.3693 / 0.2779 |
| `metal/bf16/sub/offset/offset_r128_c129` | correct | [passed](operator_peer/metal__bf16__sub__offset__offset_r128_c129.json) | 0.4558 / 0.4188 | [passed](program_peer/metal__bf16__sub__offset__offset_r128_c129.json) | 0.526 / 0.4207 |
| `metal/bf16/mul/strided/strided_r3_c5` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r3_c5.json) | 0.2414 / 0.1954 | [passed](program_peer/metal__bf16__mul__strided__strided_r3_c5.json) | 0.2552 / 0.1959 |
| `metal/bf16/mul/strided/strided_r3_c32` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r3_c32.json) | 0.2399 / 0.19 | [passed](program_peer/metal__bf16__mul__strided__strided_r3_c32.json) | 0.2575 / 0.1909 |
| `metal/bf16/mul/strided/strided_r3_c65` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r3_c65.json) | 0.1967 / 0.1861 | [passed](program_peer/metal__bf16__mul__strided__strided_r3_c65.json) | 0.2595 / 0.1911 |
| `metal/bf16/mul/strided/strided_r3_c129` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r3_c129.json) | 0.2035 / 0.1965 | [passed](program_peer/metal__bf16__mul__strided__strided_r3_c129.json) | 0.2634 / 0.1907 |
| `metal/bf16/mul/strided/strided_r16_c5` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r16_c5.json) | 0.2118 / 0.1822 | [passed](program_peer/metal__bf16__mul__strided__strided_r16_c5.json) | 0.2631 / 0.1879 |
| `metal/bf16/mul/strided/strided_r16_c32` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r16_c32.json) | 0.2572 / 0.1931 | [passed](program_peer/metal__bf16__mul__strided__strided_r16_c32.json) | 0.2746 / 0.1941 |
| `metal/bf16/mul/strided/strided_r16_c65` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r16_c65.json) | 0.2696 / 0.2014 | [passed](program_peer/metal__bf16__mul__strided__strided_r16_c65.json) | 0.3021 / 0.1981 |
| `metal/bf16/mul/strided/strided_r16_c129` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r16_c129.json) | 0.28 / 0.2198 | [passed](program_peer/metal__bf16__mul__strided__strided_r16_c129.json) | 0.2634 / 0.2165 |
| `metal/bf16/mul/strided/strided_r33_c5` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r33_c5.json) | 0.2582 / 0.1907 | [passed](program_peer/metal__bf16__mul__strided__strided_r33_c5.json) | 0.2598 / 0.1894 |
| `metal/bf16/mul/strided/strided_r33_c32` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r33_c32.json) | 0.2795 / 0.1995 | [passed](program_peer/metal__bf16__mul__strided__strided_r33_c32.json) | 0.3036 / 0.2023 |
| `metal/bf16/mul/strided/strided_r33_c65` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r33_c65.json) | 0.2889 / 0.2208 | [passed](program_peer/metal__bf16__mul__strided__strided_r33_c65.json) | 0.3148 / 0.2191 |
| `metal/bf16/mul/strided/strided_r33_c129` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r33_c129.json) | 0.2814 / 0.2358 | [passed](program_peer/metal__bf16__mul__strided__strided_r33_c129.json) | 0.2983 / 0.2388 |
| `metal/bf16/mul/strided/strided_r128_c5` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r128_c5.json) | 0.2919 / 0.1995 | [passed](program_peer/metal__bf16__mul__strided__strided_r128_c5.json) | 0.3119 / 0.1993 |
| `metal/bf16/mul/strided/strided_r128_c32` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r128_c32.json) | 0.3437 / 0.2332 | [passed](program_peer/metal__bf16__mul__strided__strided_r128_c32.json) | 0.377 / 0.2375 |
| `metal/bf16/mul/strided/strided_r128_c65` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r128_c65.json) | 0.3797 / 0.2884 | [passed](program_peer/metal__bf16__mul__strided__strided_r128_c65.json) | 0.4034 / 0.3142 |
| `metal/bf16/mul/strided/strided_r128_c129` | correct | [passed](operator_peer/metal__bf16__mul__strided__strided_r128_c129.json) | 0.4372 / 0.4034 | [passed](program_peer/metal__bf16__mul__strided__strided_r128_c129.json) | 0.506 / 0.4208 |
| `metal/bf16/mul/offset/offset_r3_c5` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r3_c5.json) | 0.2371 / 0.1833 | [passed](program_peer/metal__bf16__mul__offset__offset_r3_c5.json) | 0.2642 / 0.2255 |
| `metal/bf16/mul/offset/offset_r3_c32` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r3_c32.json) | 0.2359 / 0.1888 | [passed](program_peer/metal__bf16__mul__offset__offset_r3_c32.json) | 0.2539 / 0.1914 |
| `metal/bf16/mul/offset/offset_r3_c65` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r3_c65.json) | 0.2429 / 0.1927 | [passed](program_peer/metal__bf16__mul__offset__offset_r3_c65.json) | 0.2548 / 0.1859 |
| `metal/bf16/mul/offset/offset_r3_c129` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r3_c129.json) | 0.243 / 0.1895 | [passed](program_peer/metal__bf16__mul__offset__offset_r3_c129.json) | 0.2658 / 0.1943 |
| `metal/bf16/mul/offset/offset_r16_c5` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r16_c5.json) | 0.2489 / 0.1896 | [passed](program_peer/metal__bf16__mul__offset__offset_r16_c5.json) | 0.2609 / 0.19 |
| `metal/bf16/mul/offset/offset_r16_c32` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r16_c32.json) | 0.2576 / 0.1966 | [passed](program_peer/metal__bf16__mul__offset__offset_r16_c32.json) | 0.2804 / 0.2047 |
| `metal/bf16/mul/offset/offset_r16_c65` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r16_c65.json) | 0.259 / 0.1996 | [passed](program_peer/metal__bf16__mul__offset__offset_r16_c65.json) | 0.2906 / 0.2052 |
| `metal/bf16/mul/offset/offset_r16_c129` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r16_c129.json) | 0.2776 / 0.216 | [passed](program_peer/metal__bf16__mul__offset__offset_r16_c129.json) | 0.2835 / 0.2043 |
| `metal/bf16/mul/offset/offset_r33_c5` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r33_c5.json) | 0.2553 / 0.1941 | [passed](program_peer/metal__bf16__mul__offset__offset_r33_c5.json) | 0.2535 / 0.19 |
| `metal/bf16/mul/offset/offset_r33_c32` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r33_c32.json) | 0.2749 / 0.2039 | [passed](program_peer/metal__bf16__mul__offset__offset_r33_c32.json) | 0.2275 / 0.1991 |
| `metal/bf16/mul/offset/offset_r33_c65` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r33_c65.json) | 0.2882 / 0.2139 | [passed](program_peer/metal__bf16__mul__offset__offset_r33_c65.json) | 0.3268 / 0.2218 |
| `metal/bf16/mul/offset/offset_r33_c129` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r33_c129.json) | 0.3042 / 0.2367 | [passed](program_peer/metal__bf16__mul__offset__offset_r33_c129.json) | 0.3316 / 0.2371 |
| `metal/bf16/mul/offset/offset_r128_c5` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r128_c5.json) | 0.2912 / 0.1993 | [passed](program_peer/metal__bf16__mul__offset__offset_r128_c5.json) | 0.3221 / 0.2012 |
| `metal/bf16/mul/offset/offset_r128_c32` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r128_c32.json) | 0.3393 / 0.2388 | [passed](program_peer/metal__bf16__mul__offset__offset_r128_c32.json) | 0.3777 / 0.2396 |
| `metal/bf16/mul/offset/offset_r128_c65` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r128_c65.json) | 0.3737 / 0.3195 | [passed](program_peer/metal__bf16__mul__offset__offset_r128_c65.json) | 0.4166 / 0.3377 |
| `metal/bf16/mul/offset/offset_r128_c129` | correct | [passed](operator_peer/metal__bf16__mul__offset__offset_r128_c129.json) | 0.4138 / 0.4155 | [passed](program_peer/metal__bf16__mul__offset__offset_r128_c129.json) | 0.474 / 0.4236 |
| `metal/bf16/div/strided/strided_r3_c5` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r3_c5.json) | 0.2427 / 0.1859 | [passed](program_peer/metal__bf16__div__strided__strided_r3_c5.json) | 0.2573 / 0.1844 |
| `metal/bf16/div/strided/strided_r3_c32` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r3_c32.json) | 0.2441 / 0.1908 | [passed](program_peer/metal__bf16__div__strided__strided_r3_c32.json) | 0.2525 / 0.1902 |
| `metal/bf16/div/strided/strided_r3_c65` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r3_c65.json) | 0.2455 / 0.1972 | [passed](program_peer/metal__bf16__div__strided__strided_r3_c65.json) | 0.2562 / 0.1941 |
| `metal/bf16/div/strided/strided_r3_c129` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r3_c129.json) | 0.2462 / 0.1901 | [passed](program_peer/metal__bf16__div__strided__strided_r3_c129.json) | 0.263 / 0.1875 |
| `metal/bf16/div/strided/strided_r16_c5` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r16_c5.json) | 0.2495 / 0.184 | [passed](program_peer/metal__bf16__div__strided__strided_r16_c5.json) | 0.2282 / 0.1893 |
| `metal/bf16/div/strided/strided_r16_c32` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r16_c32.json) | 0.2601 / 0.193 | [passed](program_peer/metal__bf16__div__strided__strided_r16_c32.json) | 0.2797 / 0.1918 |
| `metal/bf16/div/strided/strided_r16_c65` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r16_c65.json) | 0.2678 / 0.1976 | [passed](program_peer/metal__bf16__div__strided__strided_r16_c65.json) | 0.287 / 0.2013 |
| `metal/bf16/div/strided/strided_r16_c129` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r16_c129.json) | 0.2768 / 0.2269 | [passed](program_peer/metal__bf16__div__strided__strided_r16_c129.json) | 0.3185 / 0.2186 |
| `metal/bf16/div/strided/strided_r33_c5` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r33_c5.json) | 0.258 / 0.1941 | [passed](program_peer/metal__bf16__div__strided__strided_r33_c5.json) | 0.2704 / 0.1861 |
| `metal/bf16/div/strided/strided_r33_c32` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r33_c32.json) | 0.2809 / 0.2071 | [passed](program_peer/metal__bf16__div__strided__strided_r33_c32.json) | 0.2577 / 0.205 |
| `metal/bf16/div/strided/strided_r33_c65` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r33_c65.json) | 0.2915 / 0.2183 | [passed](program_peer/metal__bf16__div__strided__strided_r33_c65.json) | 0.3246 / 0.2172 |
| `metal/bf16/div/strided/strided_r33_c129` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r33_c129.json) | 0.2957 / 0.2404 | [passed](program_peer/metal__bf16__div__strided__strided_r33_c129.json) | 0.3076 / 0.2398 |
| `metal/bf16/div/strided/strided_r128_c5` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r128_c5.json) | 0.292 / 0.1979 | [passed](program_peer/metal__bf16__div__strided__strided_r128_c5.json) | 0.3276 / 0.205 |
| `metal/bf16/div/strided/strided_r128_c32` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r128_c32.json) | 0.3413 / 0.2395 | [passed](program_peer/metal__bf16__div__strided__strided_r128_c32.json) | 0.3709 / 0.2304 |
| `metal/bf16/div/strided/strided_r128_c65` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r128_c65.json) | 0.3844 / 0.2949 | [passed](program_peer/metal__bf16__div__strided__strided_r128_c65.json) | 0.3892 / 0.3155 |
| `metal/bf16/div/strided/strided_r128_c129` | correct | [passed](operator_peer/metal__bf16__div__strided__strided_r128_c129.json) | 0.4431 / 0.3869 | [passed](program_peer/metal__bf16__div__strided__strided_r128_c129.json) | 0.5148 / 0.4166 |
| `metal/bf16/div/offset/offset_r3_c5` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r3_c5.json) | 0.2457 / 0.1926 | [passed](program_peer/metal__bf16__div__offset__offset_r3_c5.json) | 0.2582 / 0.1951 |
| `metal/bf16/div/offset/offset_r3_c32` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r3_c32.json) | 0.243 / 0.1941 | [passed](program_peer/metal__bf16__div__offset__offset_r3_c32.json) | 0.2551 / 0.1914 |
| `metal/bf16/div/offset/offset_r3_c65` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r3_c65.json) | 0.2439 / 0.1886 | [passed](program_peer/metal__bf16__div__offset__offset_r3_c65.json) | 0.2593 / 0.1876 |
| `metal/bf16/div/offset/offset_r3_c129` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r3_c129.json) | 0.1983 / 0.192 | [passed](program_peer/metal__bf16__div__offset__offset_r3_c129.json) | 0.2206 / 0.1912 |
| `metal/bf16/div/offset/offset_r16_c5` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r16_c5.json) | 0.2391 / 0.1868 | [passed](program_peer/metal__bf16__div__offset__offset_r16_c5.json) | 0.2373 / 0.1791 |
| `metal/bf16/div/offset/offset_r16_c32` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r16_c32.json) | 0.2559 / 0.1928 | [passed](program_peer/metal__bf16__div__offset__offset_r16_c32.json) | 0.2549 / 0.1875 |
| `metal/bf16/div/offset/offset_r16_c65` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r16_c65.json) | 0.2659 / 0.1999 | [passed](program_peer/metal__bf16__div__offset__offset_r16_c65.json) | 0.2647 / 0.2013 |
| `metal/bf16/div/offset/offset_r16_c129` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r16_c129.json) | 0.2809 / 0.2181 | [passed](program_peer/metal__bf16__div__offset__offset_r16_c129.json) | 0.3198 / 0.2214 |
| `metal/bf16/div/offset/offset_r33_c5` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r33_c5.json) | 0.2562 / 0.1946 | [passed](program_peer/metal__bf16__div__offset__offset_r33_c5.json) | 0.2695 / 0.191 |
| `metal/bf16/div/offset/offset_r33_c32` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r33_c32.json) | 0.2771 / 0.2019 | [passed](program_peer/metal__bf16__div__offset__offset_r33_c32.json) | 0.3097 / 0.2013 |
| `metal/bf16/div/offset/offset_r33_c65` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r33_c65.json) | 0.2779 / 0.2195 | [passed](program_peer/metal__bf16__div__offset__offset_r33_c65.json) | 0.2905 / 0.2046 |
| `metal/bf16/div/offset/offset_r33_c129` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r33_c129.json) | 0.3054 / 0.2375 | [passed](program_peer/metal__bf16__div__offset__offset_r33_c129.json) | 0.3294 / 0.2297 |
| `metal/bf16/div/offset/offset_r128_c5` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r128_c5.json) | 0.2488 / 0.1999 | [passed](program_peer/metal__bf16__div__offset__offset_r128_c5.json) | 0.2621 / 0.2022 |
| `metal/bf16/div/offset/offset_r128_c32` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r128_c32.json) | 0.3396 / 0.2351 | [passed](program_peer/metal__bf16__div__offset__offset_r128_c32.json) | 0.3554 / 0.2393 |
| `metal/bf16/div/offset/offset_r128_c65` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r128_c65.json) | 0.3733 / 0.3345 | [passed](program_peer/metal__bf16__div__offset__offset_r128_c65.json) | 0.4151 / 0.3447 |
| `metal/bf16/div/offset/offset_r128_c129` | correct | [passed](operator_peer/metal__bf16__div__offset__offset_r128_c129.json) | 0.4482 / 0.4099 | [passed](program_peer/metal__bf16__div__offset__offset_r128_c129.json) | 0.4695 / 0.42 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
