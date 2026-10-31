# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `490e99ba6ca453ee6555b8498cd656f4562e2d7d8ecb7fae1b7a8ec45c705f69`  
Binary SHA-256: `d994eecd1097509e3b0687a82f8722862269b510bfcbf89d818e56dbc16c8125`  
Timing: 10 iterations × 5 samples × 3 rounds

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
| `metal/f32/dot/strided/n1` | correct | [passed](operator_peer/metal__f32__dot__strided__n1.json) | 0.3214 / 0.2043 | [passed](program_peer/metal__f32__dot__strided__n1.json) | 0.3539 / 0.2076 |
| `metal/f32/dot/strided/n2` | correct | [passed](operator_peer/metal__f32__dot__strided__n2.json) | 0.3275 / 0.2082 | [passed](program_peer/metal__f32__dot__strided__n2.json) | 0.3561 / 0.2112 |
| `metal/f32/dot/strided/n3` | correct | [passed](operator_peer/metal__f32__dot__strided__n3.json) | 0.3229 / 0.2002 | [passed](program_peer/metal__f32__dot__strided__n3.json) | 0.3465 / 0.2139 |
| `metal/f32/dot/strided/n4` | correct | [passed](operator_peer/metal__f32__dot__strided__n4.json) | 0.327 / 0.2062 | [passed](program_peer/metal__f32__dot__strided__n4.json) | 0.3432 / 0.2057 |
| `metal/f32/dot/strided/n5` | correct | [passed](operator_peer/metal__f32__dot__strided__n5.json) | 0.3092 / 0.2039 | [passed](program_peer/metal__f32__dot__strided__n5.json) | 0.3257 / 0.2102 |
| `metal/f32/dot/strided/n6` | correct | [passed](operator_peer/metal__f32__dot__strided__n6.json) | 0.3254 / 0.2102 | [passed](program_peer/metal__f32__dot__strided__n6.json) | 0.343 / 0.2004 |
| `metal/f32/dot/strided/n7` | correct | [passed](operator_peer/metal__f32__dot__strided__n7.json) | 0.3309 / 0.2074 | [passed](program_peer/metal__f32__dot__strided__n7.json) | 0.3645 / 0.2069 |
| `metal/f32/dot/strided/n8` | correct | [passed](operator_peer/metal__f32__dot__strided__n8.json) | 0.338 / 0.208 | [passed](program_peer/metal__f32__dot__strided__n8.json) | 0.3452 / 0.2049 |
| `metal/f32/dot/strided/n9` | correct | [passed](operator_peer/metal__f32__dot__strided__n9.json) | 0.3552 / 0.2077 | [passed](program_peer/metal__f32__dot__strided__n9.json) | 0.3599 / 0.2082 |
| `metal/f32/dot/strided/n10` | correct | [passed](operator_peer/metal__f32__dot__strided__n10.json) | 0.349 / 0.2052 | [passed](program_peer/metal__f32__dot__strided__n10.json) | 0.3387 / 0.2007 |
| `metal/f32/dot/strided/n11` | correct | [passed](operator_peer/metal__f32__dot__strided__n11.json) | 0.3624 / 0.2027 | [passed](program_peer/metal__f32__dot__strided__n11.json) | 0.3594 / 0.2054 |
| `metal/f32/dot/strided/n12` | correct | [passed](operator_peer/metal__f32__dot__strided__n12.json) | 0.3549 / 0.2073 | [passed](program_peer/metal__f32__dot__strided__n12.json) | 0.3575 / 0.2107 |
| `metal/f32/dot/strided/n13` | correct | [passed](operator_peer/metal__f32__dot__strided__n13.json) | 0.3118 / 0.2058 | [passed](program_peer/metal__f32__dot__strided__n13.json) | 0.3598 / 0.2053 |
| `metal/f32/dot/strided/n14` | correct | [passed](operator_peer/metal__f32__dot__strided__n14.json) | 0.3173 / 0.2026 | [passed](program_peer/metal__f32__dot__strided__n14.json) | 0.3549 / 0.2043 |
| `metal/f32/dot/strided/n15` | correct | [passed](operator_peer/metal__f32__dot__strided__n15.json) | 0.3178 / 0.2008 | [passed](program_peer/metal__f32__dot__strided__n15.json) | 0.3614 / 0.2095 |
| `metal/f32/dot/strided` | correct | [passed](operator_peer/metal__f32__dot__strided.json) | 0.2983 / 0.2068 | [passed](program_peer/metal__f32__dot__strided.json) | 0.3066 / 0.2039 |
| `metal/f32/dot/strided/n17` | correct | [passed](operator_peer/metal__f32__dot__strided__n17.json) | 0.3545 / 0.2049 | [passed](program_peer/metal__f32__dot__strided__n17.json) | 0.362 / 0.2031 |
| `metal/f32/dot/strided/n18` | correct | [passed](operator_peer/metal__f32__dot__strided__n18.json) | 0.3557 / 0.2 | [passed](program_peer/metal__f32__dot__strided__n18.json) | 0.3628 / 0.2131 |
| `metal/f32/dot/strided/n19` | correct | [passed](operator_peer/metal__f32__dot__strided__n19.json) | 0.3146 / 0.2032 | [passed](program_peer/metal__f32__dot__strided__n19.json) | 0.36 / 0.2075 |
| `metal/f32/dot/strided/n20` | correct | [passed](operator_peer/metal__f32__dot__strided__n20.json) | 0.3179 / 0.2084 | [passed](program_peer/metal__f32__dot__strided__n20.json) | 0.3639 / 0.2017 |
| `metal/f32/dot/strided/n21` | correct | [passed](operator_peer/metal__f32__dot__strided__n21.json) | 0.3284 / 0.2082 | [passed](program_peer/metal__f32__dot__strided__n21.json) | 0.3126 / 0.2021 |
| `metal/f32/dot/strided/n22` | correct | [passed](operator_peer/metal__f32__dot__strided__n22.json) | 0.3352 / 0.1997 | [passed](program_peer/metal__f32__dot__strided__n22.json) | 0.3676 / 0.2041 |
| `metal/f32/dot/strided/n23` | correct | [passed](operator_peer/metal__f32__dot__strided__n23.json) | 0.323 / 0.2082 | [passed](program_peer/metal__f32__dot__strided__n23.json) | 0.3674 / 0.2019 |
| `metal/f32/dot/strided/n24` | correct | [passed](operator_peer/metal__f32__dot__strided__n24.json) | 0.3147 / 0.2084 | [passed](program_peer/metal__f32__dot__strided__n24.json) | 0.3506 / 0.204 |
| `metal/f32/dot/strided/n31` | correct | [passed](operator_peer/metal__f32__dot__strided__n31.json) | 0.3617 / 0.2027 | [passed](program_peer/metal__f32__dot__strided__n31.json) | 0.3681 / 0.2134 |
| `metal/f32/dot/strided/n32` | correct | [passed](operator_peer/metal__f32__dot__strided__n32.json) | 0.338 / 0.2064 | [passed](program_peer/metal__f32__dot__strided__n32.json) | 0.3618 / 0.1986 |
| `metal/f32/dot/strided/n33` | correct | [passed](operator_peer/metal__f32__dot__strided__n33.json) | 0.342 / 0.1986 | [passed](program_peer/metal__f32__dot__strided__n33.json) | 0.3555 / 0.2091 |
| `metal/f32/dot/strided/n63` | correct | [passed](operator_peer/metal__f32__dot__strided__n63.json) | 0.3344 / 0.2034 | [passed](program_peer/metal__f32__dot__strided__n63.json) | 0.3805 / 0.205 |
| `metal/f32/dot/strided/n64` | correct | [passed](operator_peer/metal__f32__dot__strided__n64.json) | 0.3734 / 0.1963 | [passed](program_peer/metal__f32__dot__strided__n64.json) | 0.354 / 0.1945 |
| `metal/f32/dot/strided/n65` | correct | [passed](operator_peer/metal__f32__dot__strided__n65.json) | 0.3641 / 0.2064 | [passed](program_peer/metal__f32__dot__strided__n65.json) | 0.3771 / 0.2102 |
| `metal/f32/dot/strided/n127` | correct | [passed](operator_peer/metal__f32__dot__strided__n127.json) | 0.4122 / 0.2055 | [passed](program_peer/metal__f32__dot__strided__n127.json) | 0.3801 / 0.2102 |
| `metal/f32/dot/strided/n128` | correct | [passed](operator_peer/metal__f32__dot__strided__n128.json) | 0.3709 / 0.2041 | [passed](program_peer/metal__f32__dot__strided__n128.json) | 0.3993 / 0.1989 |
| `metal/f32/dot/dot_or_vector/offset_n1` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n1.json) | 0.3138 / 0.1996 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n1.json) | 0.3455 / 0.1998 |
| `metal/f32/dot/dot_or_vector/offset_n2` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n2.json) | 0.315 / 0.2092 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n2.json) | 0.3556 / 0.2024 |
| `metal/f32/dot/dot_or_vector/offset_n3` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n3.json) | 0.3208 / 0.2006 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n3.json) | 0.3569 / 0.2064 |
| `metal/f32/dot/dot_or_vector/offset_n4` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n4.json) | 0.3321 / 0.1974 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n4.json) | 0.351 / 0.2006 |
| `metal/f32/dot/dot_or_vector/offset_n5` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n5.json) | 0.329 / 0.2019 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n5.json) | 0.3513 / 0.2013 |
| `metal/f32/dot/dot_or_vector/offset_n6` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n6.json) | 0.3287 / 0.2014 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n6.json) | 0.3506 / 0.1998 |
| `metal/f32/dot/dot_or_vector/offset_n7` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n7.json) | 0.3074 / 0.197 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n7.json) | 0.3622 / 0.2009 |
| `metal/f32/dot/dot_or_vector/offset_n8` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n8.json) | 0.3003 / 0.2021 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n8.json) | 0.3579 / 0.2004 |
| `metal/f32/dot/dot_or_vector/offset_n9` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n9.json) | 0.3213 / 0.1987 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n9.json) | 0.358 / 0.2086 |
| `metal/f32/dot/dot_or_vector/offset_n10` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n10.json) | 0.3365 / 0.2042 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n10.json) | 0.3485 / 0.2095 |
| `metal/f32/dot/dot_or_vector/offset_n11` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n11.json) | 0.3098 / 0.1986 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n11.json) | 0.3529 / 0.1991 |
| `metal/f32/dot/dot_or_vector/offset_n12` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n12.json) | 0.3291 / 0.1952 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n12.json) | 0.3454 / 0.2071 |
| `metal/f32/dot/dot_or_vector/offset_n13` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n13.json) | 0.3282 / 0.2015 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n13.json) | 0.3593 / 0.2112 |
| `metal/f32/dot/dot_or_vector/offset_n14` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n14.json) | 0.3443 / 0.1978 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n14.json) | 0.3544 / 0.1998 |
| `metal/f32/dot/dot_or_vector/offset_n15` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n15.json) | 0.312 / 0.2012 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n15.json) | 0.3604 / 0.2038 |
| `metal/f32/dot/dot_or_vector/offset_n16` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n16.json) | 0.3151 / 0.1974 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n16.json) | 0.36 / 0.2 |
| `metal/f32/dot/dot_or_vector/offset_n17` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n17.json) | 0.3244 / 0.1962 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n17.json) | 0.3192 / 0.1968 |
| `metal/f32/dot/dot_or_vector/offset_n18` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n18.json) | 0.3062 / 0.2001 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n18.json) | 0.3642 / 0.206 |
| `metal/f32/dot/dot_or_vector/offset_n19` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n19.json) | 0.3158 / 0.1979 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n19.json) | 0.3612 / 0.2014 |
| `metal/f32/dot/dot_or_vector/offset_n20` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n20.json) | 0.3041 / 0.1967 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n20.json) | 0.3622 / 0.2063 |
| `metal/f32/dot/dot_or_vector/offset_n21` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n21.json) | 0.3176 / 0.198 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n21.json) | 0.366 / 0.2071 |
| `metal/f32/dot/dot_or_vector/offset_n22` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n22.json) | 0.3436 / 0.1969 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n22.json) | 0.3651 / 0.2077 |
| `metal/f32/dot/dot_or_vector/offset_n23` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n23.json) | 0.3139 / 0.1993 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n23.json) | 0.3775 / 0.2054 |
| `metal/f32/dot/dot_or_vector/offset_n24` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n24.json) | 0.334 / 0.1986 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n24.json) | 0.3557 / 0.2054 |
| `metal/f32/dot/dot_or_vector/offset_n31` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n31.json) | 0.3461 / 0.1981 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n31.json) | 0.3712 / 0.2104 |
| `metal/f32/dot/dot_or_vector/offset_n32` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n32.json) | 0.321 / 0.2023 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n32.json) | 0.3622 / 0.2023 |
| `metal/f32/dot/dot_or_vector/offset_n33` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n33.json) | 0.3279 / 0.2033 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n33.json) | 0.3653 / 0.1991 |
| `metal/f32/dot/dot_or_vector/offset_n63` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n63.json) | 0.3401 / 0.204 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n63.json) | 0.3803 / 0.2086 |
| `metal/f32/dot/dot_or_vector/offset_n64` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n64.json) | 0.3384 / 0.1974 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n64.json) | 0.3532 / 0.2002 |
| `metal/f32/dot/dot_or_vector/offset_n65` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n65.json) | 0.3602 / 0.2014 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n65.json) | 0.3755 / 0.205 |
| `metal/f32/dot/dot_or_vector/offset_n127` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n127.json) | 0.3671 / 0.1965 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n127.json) | 0.4166 / 0.2026 |
| `metal/f32/dot/dot_or_vector/offset_n128` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n128.json) | 0.3795 / 0.1973 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n128.json) | 0.4116 / 0.2036 |
| `metal/bf16/dot/strided/n1` | correct | [passed](operator_peer/metal__bf16__dot__strided__n1.json) | 0.3053 / 0.1961 | [passed](program_peer/metal__bf16__dot__strided__n1.json) | 0.3631 / 0.21 |
| `metal/bf16/dot/strided/n2` | correct | [passed](operator_peer/metal__bf16__dot__strided__n2.json) | 0.3423 / 0.2024 | [passed](program_peer/metal__bf16__dot__strided__n2.json) | 0.3645 / 0.2082 |
| `metal/bf16/dot/strided/n3` | correct | [passed](operator_peer/metal__bf16__dot__strided__n3.json) | 0.3091 / 0.2072 | [passed](program_peer/metal__bf16__dot__strided__n3.json) | 0.3428 / 0.2043 |
| `metal/bf16/dot/strided/n4` | correct | [passed](operator_peer/metal__bf16__dot__strided__n4.json) | 0.3032 / 0.2023 | [passed](program_peer/metal__bf16__dot__strided__n4.json) | 0.3645 / 0.2046 |
| `metal/bf16/dot/strided/n5` | correct | [passed](operator_peer/metal__bf16__dot__strided__n5.json) | 0.3584 / 0.2015 | [passed](program_peer/metal__bf16__dot__strided__n5.json) | 0.367 / 0.2051 |
| `metal/bf16/dot/strided/n6` | correct | [passed](operator_peer/metal__bf16__dot__strided__n6.json) | 0.3131 / 0.2024 | [passed](program_peer/metal__bf16__dot__strided__n6.json) | 0.3619 / 0.2022 |
| `metal/bf16/dot/strided/n7` | correct | [passed](operator_peer/metal__bf16__dot__strided__n7.json) | 0.3124 / 0.2028 | [passed](program_peer/metal__bf16__dot__strided__n7.json) | 0.3142 / 0.2087 |
| `metal/bf16/dot/strided/n8` | correct | [passed](operator_peer/metal__bf16__dot__strided__n8.json) | 0.3178 / 0.2019 | [passed](program_peer/metal__bf16__dot__strided__n8.json) | 0.3663 / 0.2074 |
| `metal/bf16/dot/strided/n9` | correct | [passed](operator_peer/metal__bf16__dot__strided__n9.json) | 0.3124 / 0.2095 | [passed](program_peer/metal__bf16__dot__strided__n9.json) | 0.3639 / 0.2084 |
| `metal/bf16/dot/strided/n10` | correct | [passed](operator_peer/metal__bf16__dot__strided__n10.json) | 0.3308 / 0.2094 | [passed](program_peer/metal__bf16__dot__strided__n10.json) | 0.3332 / 0.2049 |
| `metal/bf16/dot/strided/n11` | correct | [passed](operator_peer/metal__bf16__dot__strided__n11.json) | 0.36 / 0.2031 | [passed](program_peer/metal__bf16__dot__strided__n11.json) | 0.3537 / 0.2045 |
| `metal/bf16/dot/strided/n12` | correct | [passed](operator_peer/metal__bf16__dot__strided__n12.json) | 0.3106 / 0.2041 | [passed](program_peer/metal__bf16__dot__strided__n12.json) | 0.3514 / 0.1965 |
| `metal/bf16/dot/strided/n13` | correct | [passed](operator_peer/metal__bf16__dot__strided__n13.json) | 0.3291 / 0.2034 | [passed](program_peer/metal__bf16__dot__strided__n13.json) | 0.3663 / 0.2061 |
| `metal/bf16/dot/strided/n14` | correct | [passed](operator_peer/metal__bf16__dot__strided__n14.json) | 0.3642 / 0.2039 | [passed](program_peer/metal__bf16__dot__strided__n14.json) | 0.3686 / 0.2098 |
| `metal/bf16/dot/strided/n15` | correct | [passed](operator_peer/metal__bf16__dot__strided__n15.json) | 0.322 / 0.2074 | [passed](program_peer/metal__bf16__dot__strided__n15.json) | 0.3666 / 0.2034 |
| `metal/bf16/dot/strided` | correct | [passed](operator_peer/metal__bf16__dot__strided.json) | 0.3222 / 0.2016 | [passed](program_peer/metal__bf16__dot__strided.json) | 0.3714 / 0.2045 |
| `metal/bf16/dot/strided/n17` | correct | [passed](operator_peer/metal__bf16__dot__strided__n17.json) | 0.3593 / 0.2074 | [passed](program_peer/metal__bf16__dot__strided__n17.json) | 0.3706 / 0.2049 |
| `metal/bf16/dot/strided/n18` | correct | [passed](operator_peer/metal__bf16__dot__strided__n18.json) | 0.3677 / 0.198 | [passed](program_peer/metal__bf16__dot__strided__n18.json) | 0.373 / 0.2059 |
| `metal/bf16/dot/strided/n19` | correct | [passed](operator_peer/metal__bf16__dot__strided__n19.json) | 0.3379 / 0.2025 | [passed](program_peer/metal__bf16__dot__strided__n19.json) | 0.3719 / 0.2063 |
| `metal/bf16/dot/strided/n20` | correct | [passed](operator_peer/metal__bf16__dot__strided__n20.json) | 0.3192 / 0.2018 | [passed](program_peer/metal__bf16__dot__strided__n20.json) | 0.3674 / 0.2014 |
| `metal/bf16/dot/strided/n21` | correct | [passed](operator_peer/metal__bf16__dot__strided__n21.json) | 0.3597 / 0.2091 | [passed](program_peer/metal__bf16__dot__strided__n21.json) | 0.3735 / 0.2057 |
| `metal/bf16/dot/strided/n22` | correct | [passed](operator_peer/metal__bf16__dot__strided__n22.json) | 0.3196 / 0.2069 | [passed](program_peer/metal__bf16__dot__strided__n22.json) | 0.3693 / 0.2033 |
| `metal/bf16/dot/strided/n23` | correct | [passed](operator_peer/metal__bf16__dot__strided__n23.json) | 0.3671 / 0.2014 | [passed](program_peer/metal__bf16__dot__strided__n23.json) | 0.3671 / 0.2051 |
| `metal/bf16/dot/strided/n24` | correct | [passed](operator_peer/metal__bf16__dot__strided__n24.json) | 0.3259 / 0.2056 | [passed](program_peer/metal__bf16__dot__strided__n24.json) | 0.3736 / 0.2025 |
| `metal/bf16/dot/strided/n31` | correct | [passed](operator_peer/metal__bf16__dot__strided__n31.json) | 0.3277 / 0.2066 | [passed](program_peer/metal__bf16__dot__strided__n31.json) | 0.3691 / 0.2024 |
| `metal/bf16/dot/strided/n32` | correct | [passed](operator_peer/metal__bf16__dot__strided__n32.json) | 0.3312 / 0.2062 | [passed](program_peer/metal__bf16__dot__strided__n32.json) | 0.3739 / 0.2058 |
| `metal/bf16/dot/strided/n33` | correct | [passed](operator_peer/metal__bf16__dot__strided__n33.json) | 0.3572 / 0.2011 | [passed](program_peer/metal__bf16__dot__strided__n33.json) | 0.3734 / 0.207 |
| `metal/bf16/dot/strided/n63` | correct | [passed](operator_peer/metal__bf16__dot__strided__n63.json) | 0.3537 / 0.1993 | [passed](program_peer/metal__bf16__dot__strided__n63.json) | 0.3684 / 0.201 |
| `metal/bf16/dot/strided/n64` | correct | [passed](operator_peer/metal__bf16__dot__strided__n64.json) | 0.3817 / 0.2044 | [passed](program_peer/metal__bf16__dot__strided__n64.json) | 0.3971 / 0.2112 |
| `metal/bf16/dot/strided/n65` | correct | [passed](operator_peer/metal__bf16__dot__strided__n65.json) | 0.336 / 0.1916 | [passed](program_peer/metal__bf16__dot__strided__n65.json) | 0.3439 / 0.2077 |
| `metal/bf16/dot/strided/n127` | correct | [passed](operator_peer/metal__bf16__dot__strided__n127.json) | 0.3741 / 0.2015 | [passed](program_peer/metal__bf16__dot__strided__n127.json) | 0.379 / 0.2086 |
| `metal/bf16/dot/strided/n128` | correct | [passed](operator_peer/metal__bf16__dot__strided__n128.json) | 0.3185 / 0.2013 | [passed](program_peer/metal__bf16__dot__strided__n128.json) | 0.3626 / 0.2051 |
| `metal/bf16/dot/dot_or_vector/offset_n1` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n1.json) | 0.3371 / 0.2041 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n1.json) | 0.3477 / 0.2004 |
| `metal/bf16/dot/dot_or_vector/offset_n2` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n2.json) | 0.3014 / 0.2036 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n2.json) | 0.3605 / 0.2034 |
| `metal/bf16/dot/dot_or_vector/offset_n3` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n3.json) | 0.362 / 0.2008 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n3.json) | 0.316 / 0.208 |
| `metal/bf16/dot/dot_or_vector/offset_n4` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n4.json) | 0.3085 / 0.2002 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n4.json) | 0.3458 / 0.2011 |
| `metal/bf16/dot/dot_or_vector/offset_n5` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n5.json) | 0.351 / 0.203 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n5.json) | 0.3468 / 0.2072 |
| `metal/bf16/dot/dot_or_vector/offset_n6` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n6.json) | 0.3498 / 0.204 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n6.json) | 0.3323 / 0.2081 |
| `metal/bf16/dot/dot_or_vector/offset_n7` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n7.json) | 0.3297 / 0.2047 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n7.json) | 0.3681 / 0.2059 |
| `metal/bf16/dot/dot_or_vector/offset_n8` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n8.json) | 0.3388 / 0.1999 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n8.json) | 0.3264 / 0.2041 |
| `metal/bf16/dot/dot_or_vector/offset_n9` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n9.json) | 0.3113 / 0.2017 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n9.json) | 0.3641 / 0.2 |
| `metal/bf16/dot/dot_or_vector/offset_n10` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n10.json) | 0.3614 / 0.1952 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n10.json) | 0.3663 / 0.1984 |
| `metal/bf16/dot/dot_or_vector/offset_n11` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n11.json) | 0.3326 / 0.2068 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n11.json) | 0.3684 / 0.203 |
| `metal/bf16/dot/dot_or_vector/offset_n12` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n12.json) | 0.3395 / 0.2014 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n12.json) | 0.3702 / 0.2028 |
| `metal/bf16/dot/dot_or_vector/offset_n13` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n13.json) | 0.3508 / 0.2028 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n13.json) | 0.3651 / 0.2035 |
| `metal/bf16/dot/dot_or_vector/offset_n14` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n14.json) | 0.3615 / 0.1981 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n14.json) | 0.371 / 0.2107 |
| `metal/bf16/dot/dot_or_vector/offset_n15` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n15.json) | 0.3589 / 0.2062 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n15.json) | 0.3724 / 0.2064 |
| `metal/bf16/dot/dot_or_vector/offset_n16` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n16.json) | 0.3637 / 0.207 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n16.json) | 0.371 / 0.2079 |
| `metal/bf16/dot/dot_or_vector/offset_n17` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n17.json) | 0.3154 / 0.2029 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n17.json) | 0.3689 / 0.2049 |
| `metal/bf16/dot/dot_or_vector/offset_n18` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n18.json) | 0.3559 / 0.2034 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n18.json) | 0.3681 / 0.2071 |
| `metal/bf16/dot/dot_or_vector/offset_n19` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n19.json) | 0.362 / 0.2062 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n19.json) | 0.3695 / 0.2043 |
| `metal/bf16/dot/dot_or_vector/offset_n20` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n20.json) | 0.3326 / 0.1968 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n20.json) | 0.3263 / 0.203 |
| `metal/bf16/dot/dot_or_vector/offset_n21` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n21.json) | 0.3631 / 0.2075 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n21.json) | 0.3737 / 0.2007 |
| `metal/bf16/dot/dot_or_vector/offset_n22` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n22.json) | 0.3527 / 0.2016 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n22.json) | 0.3731 / 0.2075 |
| `metal/bf16/dot/dot_or_vector/offset_n23` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n23.json) | 0.3624 / 0.2001 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n23.json) | 0.371 / 0.2001 |
| `metal/bf16/dot/dot_or_vector/offset_n24` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n24.json) | 0.3634 / 0.205 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n24.json) | 0.3705 / 0.196 |
| `metal/bf16/dot/dot_or_vector/offset_n31` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n31.json) | 0.3707 / 0.2037 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n31.json) | 0.3731 / 0.2034 |
| `metal/bf16/dot/dot_or_vector/offset_n32` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n32.json) | 0.3706 / 0.2051 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n32.json) | 0.3804 / 0.2052 |
| `metal/bf16/dot/dot_or_vector/offset_n33` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n33.json) | 0.3397 / 0.2028 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n33.json) | 0.3654 / 0.2053 |
| `metal/bf16/dot/dot_or_vector/offset_n63` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n63.json) | 0.3497 / 0.202 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n63.json) | 0.3836 / 0.196 |
| `metal/bf16/dot/dot_or_vector/offset_n64` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n64.json) | 0.3817 / 0.2058 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n64.json) | 0.3878 / 0.1998 |
| `metal/bf16/dot/dot_or_vector/offset_n65` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n65.json) | 0.375 / 0.1988 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n65.json) | 0.372 / 0.2077 |
| `metal/bf16/dot/dot_or_vector/offset_n127` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n127.json) | 0.4266 / 0.1994 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n127.json) | 0.3945 / 0.2087 |
| `metal/bf16/dot/dot_or_vector/offset_n128` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n128.json) | 0.3581 / 0.2018 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n128.json) | 0.3719 / 0.2096 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
