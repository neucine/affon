# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `039db535f2543b43eb3c444289f6e2001166598213a7a3762c6cc757a917b5f4`  
Binary SHA-256: `8cf45c727d1adf3054252a616442e3900674aa8612990d5ddcb3e0b39e128490`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 28
- Numerical check passed: 28
- Operator check and timing passed: 28
- Program check and timing passed: 28
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/std_all/outer_axis` | reduction | 64 × 128 | dense | correct | [passed](operator_peer/metal__f32__std_all__outer_axis.json) | 0.2956 / 0.2354 | [passed](program_peer/metal__f32__std_all__outer_axis.json) | 0.299 / 0.2303 |
| `metal/f32/variance_all/outer_axis` | reduction | 64 × 128 | dense | correct | [passed](operator_peer/metal__f32__variance_all__outer_axis.json) | 0.3027 / 0.2263 | [passed](program_peer/metal__f32__variance_all__outer_axis.json) | 0.2995 / 0.2331 |
| `metal/f32/max_all/strided/r64_c127` | reduction | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__max_all__strided__r64_c127.json) | 0.3779 / 0.208 | [passed](program_peer/metal__f32__max_all__strided__r64_c127.json) | 0.263 / 0.2108 |
| `metal/f32/sum_all/strided/r64_c127` | reduction | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__sum_all__strided__r64_c127.json) | 0.2581 / 0.2046 | [passed](program_peer/metal__f32__sum_all__strided__r64_c127.json) | 0.2877 / 0.2148 |
| `metal/f32/min_all/strided/r64_c127` | reduction | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__min_all__strided__r64_c127.json) | 0.2543 / 0.2044 | [passed](program_peer/metal__f32__min_all__strided__r64_c127.json) | 0.2659 / 0.2153 |
| `metal/f32/mean_all/strided/r64_c127` | reduction | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__mean_all__strided__r64_c127.json) | 0.2559 / 0.2099 | [passed](program_peer/metal__f32__mean_all__strided__r64_c127.json) | 0.2556 / 0.2192 |
| `metal/f32/mean_all/outer_axis` | reduction | 64 × 128 | dense | correct | [passed](operator_peer/metal__f32__mean_all__outer_axis.json) | 0.2756 / 0.2005 | [passed](program_peer/metal__f32__mean_all__outer_axis.json) | 0.2673 / 0.2032 |
| `metal/f32/sum_all/outer_axis` | reduction | 64 × 128 | dense | correct | [passed](operator_peer/metal__f32__sum_all__outer_axis.json) | 0.3272 / 0.2064 | [passed](program_peer/metal__f32__sum_all__outer_axis.json) | 0.2655 / 0.2156 |
| `metal/f32/max_all/outer_axis` | reduction | 64 × 128 | dense | correct | [passed](operator_peer/metal__f32__max_all__outer_axis.json) | 0.3357 / 0.2068 | [passed](program_peer/metal__f32__max_all__outer_axis.json) | 0.3376 / 0.2076 |
| `metal/f32/min_all/outer_axis` | reduction | 64 × 128 | dense | correct | [passed](operator_peer/metal__f32__min_all__outer_axis.json) | 0.2548 / 0.205 | [passed](program_peer/metal__f32__min_all__outer_axis.json) | 0.2906 / 0.2118 |
| `metal/f32/sum_all/ragged/r33_c129` | reduction | 33 × 129 | dense | correct | [passed](operator_peer/metal__f32__sum_all__ragged__r33_c129.json) | 0.3404 / 0.205 | [passed](program_peer/metal__f32__sum_all__ragged__r33_c129.json) | 0.2456 / 0.2106 |
| `metal/f32/max_all/ragged/r33_c129` | reduction | 33 × 129 | dense | correct | [passed](operator_peer/metal__f32__max_all__ragged__r33_c129.json) | 0.2378 / 0.2015 | [passed](program_peer/metal__f32__max_all__ragged__r33_c129.json) | 0.2349 / 0.2037 |
| `metal/f32/min_all/ragged/r33_c129` | reduction | 33 × 129 | dense | correct | [passed](operator_peer/metal__f32__min_all__ragged__r33_c129.json) | 0.3521 / 0.2045 | [passed](program_peer/metal__f32__min_all__ragged__r33_c129.json) | 0.3395 / 0.205 |
| `metal/f32/mean_all/ragged/r33_c129` | reduction | 33 × 129 | dense | correct | [passed](operator_peer/metal__f32__mean_all__ragged__r33_c129.json) | 0.2337 / 0.2064 | [passed](program_peer/metal__f32__mean_all__ragged__r33_c129.json) | 0.2955 / 0.2095 |
| `metal/f32/max_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/metal__f32__max_all__long_axis.json) | 0.3307 / 0.2055 | [passed](program_peer/metal__f32__max_all__long_axis.json) | 0.2515 / 0.1987 |
| `metal/f32/min_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/metal__f32__min_all__long_axis.json) | 0.2991 / 0.205 | [passed](program_peer/metal__f32__min_all__long_axis.json) | 0.2396 / 0.2041 |
| `metal/f32/sum_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/metal__f32__sum_all__long_axis.json) | 0.329 / 0.2011 | [passed](program_peer/metal__f32__sum_all__long_axis.json) | 0.3313 / 0.2009 |
| `metal/f32/mean_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/metal__f32__mean_all__long_axis.json) | 0.2273 / 0.2008 | [passed](program_peer/metal__f32__mean_all__long_axis.json) | 0.3277 / 0.2038 |
| `metal/f32/variance_all/ragged` | reduction | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__variance_all__ragged.json) | 0.2823 / 0.2067 | [passed](program_peer/metal__f32__variance_all__ragged.json) | 0.2917 / 0.211 |
| `metal/f32/std_all/ragged` | reduction | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__std_all__ragged.json) | 0.253 / 0.2063 | [passed](program_peer/metal__f32__std_all__ragged.json) | 0.3105 / 0.2113 |
| `metal/f32/max_all/ragged` | reduction | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__max_all__ragged.json) | 0.2912 / 0.1991 | [passed](program_peer/metal__f32__max_all__ragged.json) | 0.2866 / 0.1994 |
| `metal/f32/sum_all/ragged` | reduction | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__sum_all__ragged.json) | 0.2772 / 0.2008 | [passed](program_peer/metal__f32__sum_all__ragged.json) | 0.2923 / 0.2014 |
| `metal/f32/min_all/ragged` | reduction | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__min_all__ragged.json) | 0.2921 / 0.192 | [passed](program_peer/metal__f32__min_all__ragged.json) | 0.2922 / 0.1989 |
| `metal/f32/mean_all/ragged` | reduction | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__mean_all__ragged.json) | 0.2514 / 0.1988 | [passed](program_peer/metal__f32__mean_all__ragged.json) | 0.2489 / 0.2044 |
| `metal/f32/std_all/strided/r64_c127` | reduction | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__std_all__strided__r64_c127.json) | 0.3526 / 0.2397 | [passed](program_peer/metal__f32__std_all__strided__r64_c127.json) | 0.2992 / 0.2353 |
| `metal/f32/variance_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/metal__f32__variance_all__long_axis.json) | 0.289 / 0.208 | [passed](program_peer/metal__f32__variance_all__long_axis.json) | 0.2457 / 0.2042 |
| `metal/f32/variance_all/strided/r64_c127` | reduction | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__variance_all__strided__r64_c127.json) | 0.3019 / 0.2386 | [passed](program_peer/metal__f32__variance_all__strided__r64_c127.json) | 0.2978 / 0.2327 |
| `metal/f32/std_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/metal__f32__std_all__long_axis.json) | 0.3874 / 0.2109 | [passed](program_peer/metal__f32__std_all__long_axis.json) | 0.245 / 0.2084 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
