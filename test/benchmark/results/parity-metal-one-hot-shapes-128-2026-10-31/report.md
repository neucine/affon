# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `1ebe7ce95773f919143ad1776e4b5596ee5a697124c99e2f50864c4d59e6bec5`  
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
| `metal/f32/one_hot/small` | correct | [passed](operator_peer/metal__f32__one_hot__small.json) | 0.3343 / 0.2645 | [passed](program_peer/metal__f32__one_hot__small.json) | 0.3594 / 0.2632 |
| `metal/f32/one_hot/small/b1_n1_c3` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n1_c3.json) | 0.3373 / 0.2574 | [passed](program_peer/metal__f32__one_hot__small__b1_n1_c3.json) | 0.3604 / 0.2609 |
| `metal/f32/one_hot/small/b1_n1_c7` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n1_c7.json) | 0.3132 / 0.2528 | [passed](program_peer/metal__f32__one_hot__small__b1_n1_c7.json) | 0.3594 / 0.2637 |
| `metal/f32/one_hot/small/b1_n1_c16` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n1_c16.json) | 0.3304 / 0.2526 | [passed](program_peer/metal__f32__one_hot__small__b1_n1_c16.json) | 0.3586 / 0.261 |
| `metal/f32/one_hot/small/b1_n1_c17` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n1_c17.json) | 0.3394 / 0.2613 | [passed](program_peer/metal__f32__one_hot__small__b1_n1_c17.json) | 0.3435 / 0.2629 |
| `metal/f32/one_hot/small/b1_n1_c31` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n1_c31.json) | 0.3367 / 0.2543 | [passed](program_peer/metal__f32__one_hot__small__b1_n1_c31.json) | 0.3713 / 0.2636 |
| `metal/f32/one_hot/small/b1_n1_c32` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n1_c32.json) | 0.3286 / 0.251 | [passed](program_peer/metal__f32__one_hot__small__b1_n1_c32.json) | 0.3611 / 0.2606 |
| `metal/f32/one_hot/small/b1_n1_c33` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n1_c33.json) | 0.328 / 0.2545 | [passed](program_peer/metal__f32__one_hot__small__b1_n1_c33.json) | 0.3161 / 0.2591 |
| `metal/f32/one_hot/small/b1_n3_c2` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n3_c2.json) | 0.3409 / 0.2634 | [passed](program_peer/metal__f32__one_hot__small__b1_n3_c2.json) | 0.3608 / 0.2564 |
| `metal/f32/one_hot/small/b1_n3_c3` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n3_c3.json) | 0.3478 / 0.2594 | [passed](program_peer/metal__f32__one_hot__small__b1_n3_c3.json) | 0.3681 / 0.2604 |
| `metal/f32/one_hot/small/b1_n3_c7` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n3_c7.json) | 0.3558 / 0.2493 | [passed](program_peer/metal__f32__one_hot__small__b1_n3_c7.json) | 0.3535 / 0.2601 |
| `metal/f32/one_hot/small/b1_n3_c16` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n3_c16.json) | 0.335 / 0.2579 | [passed](program_peer/metal__f32__one_hot__small__b1_n3_c16.json) | 0.3681 / 0.2643 |
| `metal/f32/one_hot/small/b1_n3_c17` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n3_c17.json) | 0.3386 / 0.2566 | [passed](program_peer/metal__f32__one_hot__small__b1_n3_c17.json) | 0.3681 / 0.2596 |
| `metal/f32/one_hot/small/b1_n3_c31` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n3_c31.json) | 0.3476 / 0.2561 | [passed](program_peer/metal__f32__one_hot__small__b1_n3_c31.json) | 0.3706 / 0.2635 |
| `metal/f32/one_hot/small/b1_n3_c32` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n3_c32.json) | 0.3435 / 0.2521 | [passed](program_peer/metal__f32__one_hot__small__b1_n3_c32.json) | 0.3649 / 0.2643 |
| `metal/f32/one_hot/small/b1_n3_c33` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n3_c33.json) | 0.3467 / 0.2601 | [passed](program_peer/metal__f32__one_hot__small__b1_n3_c33.json) | 0.3689 / 0.2639 |
| `metal/f32/one_hot/small/b1_n8_c2` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n8_c2.json) | 0.3435 / 0.2626 | [passed](program_peer/metal__f32__one_hot__small__b1_n8_c2.json) | 0.3645 / 0.2629 |
| `metal/f32/one_hot/small/b1_n8_c3` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n8_c3.json) | 0.345 / 0.2625 | [passed](program_peer/metal__f32__one_hot__small__b1_n8_c3.json) | 0.324 / 0.2619 |
| `metal/f32/one_hot/small/b1_n8_c7` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n8_c7.json) | 0.3594 / 0.261 | [passed](program_peer/metal__f32__one_hot__small__b1_n8_c7.json) | 0.3789 / 0.2585 |
| `metal/f32/one_hot/small/b1_n8_c16` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n8_c16.json) | 0.3587 / 0.259 | [passed](program_peer/metal__f32__one_hot__small__b1_n8_c16.json) | 0.3834 / 0.264 |
| `metal/f32/one_hot/small/b1_n8_c17` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n8_c17.json) | 0.3531 / 0.2671 | [passed](program_peer/metal__f32__one_hot__small__b1_n8_c17.json) | 0.3719 / 0.2595 |
| `metal/f32/one_hot/small/b1_n8_c31` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n8_c31.json) | 0.3659 / 0.2553 | [passed](program_peer/metal__f32__one_hot__small__b1_n8_c31.json) | 0.3958 / 0.266 |
| `metal/f32/one_hot/small/b1_n8_c32` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n8_c32.json) | 0.354 / 0.2595 | [passed](program_peer/metal__f32__one_hot__small__b1_n8_c32.json) | 0.394 / 0.2531 |
| `metal/f32/one_hot/small/b1_n8_c33` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n8_c33.json) | 0.3688 / 0.262 | [passed](program_peer/metal__f32__one_hot__small__b1_n8_c33.json) | 0.3447 / 0.2642 |
| `metal/f32/one_hot/small/b1_n17_c2` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n17_c2.json) | 0.3634 / 0.2598 | [passed](program_peer/metal__f32__one_hot__small__b1_n17_c2.json) | 0.3739 / 0.2613 |
| `metal/f32/one_hot/small/b1_n17_c3` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n17_c3.json) | 0.363 / 0.2681 | [passed](program_peer/metal__f32__one_hot__small__b1_n17_c3.json) | 0.377 / 0.2641 |
| `metal/f32/one_hot/small/b1_n17_c7` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n17_c7.json) | 0.369 / 0.2605 | [passed](program_peer/metal__f32__one_hot__small__b1_n17_c7.json) | 0.3593 / 0.2568 |
| `metal/f32/one_hot/small/b1_n17_c16` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n17_c16.json) | 0.3798 / 0.2571 | [passed](program_peer/metal__f32__one_hot__small__b1_n17_c16.json) | 0.3997 / 0.2657 |
| `metal/f32/one_hot/small/b1_n17_c17` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n17_c17.json) | 0.368 / 0.2691 | [passed](program_peer/metal__f32__one_hot__small__b1_n17_c17.json) | 0.4045 / 0.2631 |
| `metal/f32/one_hot/small/b1_n17_c31` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n17_c31.json) | 0.4262 / 0.2599 | [passed](program_peer/metal__f32__one_hot__small__b1_n17_c31.json) | 0.4154 / 0.2647 |
| `metal/f32/one_hot/small/b1_n17_c32` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n17_c32.json) | 0.4229 / 0.2725 | [passed](program_peer/metal__f32__one_hot__small__b1_n17_c32.json) | 0.4305 / 0.2596 |
| `metal/f32/one_hot/small/b1_n17_c33` | correct | [passed](operator_peer/metal__f32__one_hot__small__b1_n17_c33.json) | 0.4211 / 0.2649 | [passed](program_peer/metal__f32__one_hot__small__b1_n17_c33.json) | 0.4315 / 0.2686 |
| `metal/f32/one_hot/small/b2_n1_c2` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n1_c2.json) | 0.3376 / 0.262 | [passed](program_peer/metal__f32__one_hot__small__b2_n1_c2.json) | 0.3632 / 0.2587 |
| `metal/f32/one_hot/small/b2_n1_c3` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n1_c3.json) | 0.3391 / 0.2592 | [passed](program_peer/metal__f32__one_hot__small__b2_n1_c3.json) | 0.3586 / 0.2591 |
| `metal/f32/one_hot/small/b2_n1_c7` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n1_c7.json) | 0.3464 / 0.2542 | [passed](program_peer/metal__f32__one_hot__small__b2_n1_c7.json) | 0.3694 / 0.2656 |
| `metal/f32/one_hot/small/b2_n1_c16` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n1_c16.json) | 0.3418 / 0.2591 | [passed](program_peer/metal__f32__one_hot__small__b2_n1_c16.json) | 0.3428 / 0.2577 |
| `metal/f32/one_hot/small/b2_n1_c17` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n1_c17.json) | 0.3415 / 0.2614 | [passed](program_peer/metal__f32__one_hot__small__b2_n1_c17.json) | 0.366 / 0.2654 |
| `metal/f32/one_hot/small/b2_n1_c31` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n1_c31.json) | 0.3391 / 0.2551 | [passed](program_peer/metal__f32__one_hot__small__b2_n1_c31.json) | 0.349 / 0.2578 |
| `metal/f32/one_hot/small/b2_n1_c32` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n1_c32.json) | 0.3459 / 0.2566 | [passed](program_peer/metal__f32__one_hot__small__b2_n1_c32.json) | 0.365 / 0.2599 |
| `metal/f32/one_hot/small/b2_n1_c33` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n1_c33.json) | 0.3237 / 0.2589 | [passed](program_peer/metal__f32__one_hot__small__b2_n1_c33.json) | 0.35 / 0.2563 |
| `metal/f32/one_hot/small/b2_n3_c2` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n3_c2.json) | 0.3628 / 0.2644 | [passed](program_peer/metal__f32__one_hot__small__b2_n3_c2.json) | 0.3664 / 0.2627 |
| `metal/f32/one_hot/small/b2_n3_c3` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n3_c3.json) | 0.329 / 0.253 | [passed](program_peer/metal__f32__one_hot__small__b2_n3_c3.json) | 0.3457 / 0.2612 |
| `metal/f32/one_hot/small/b2_n3_c7` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n3_c7.json) | 0.3538 / 0.2568 | [passed](program_peer/metal__f32__one_hot__small__b2_n3_c7.json) | 0.3743 / 0.2575 |
| `metal/f32/one_hot/small/b2_n3_c16` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n3_c16.json) | 0.3448 / 0.2578 | [passed](program_peer/metal__f32__one_hot__small__b2_n3_c16.json) | 0.3742 / 0.2668 |
| `metal/f32/one_hot/small/b2_n3_c17` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n3_c17.json) | 0.3202 / 0.2509 | [passed](program_peer/metal__f32__one_hot__small__b2_n3_c17.json) | 0.3649 / 0.2532 |
| `metal/f32/one_hot/small/b2_n3_c31` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n3_c31.json) | 0.3572 / 0.2632 | [passed](program_peer/metal__f32__one_hot__small__b2_n3_c31.json) | 0.39 / 0.2552 |
| `metal/f32/one_hot/small/b2_n3_c32` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n3_c32.json) | 0.3592 / 0.2587 | [passed](program_peer/metal__f32__one_hot__small__b2_n3_c32.json) | 0.3947 / 0.2655 |
| `metal/f32/one_hot/small/b2_n3_c33` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n3_c33.json) | 0.3562 / 0.263 | [passed](program_peer/metal__f32__one_hot__small__b2_n3_c33.json) | 0.3871 / 0.2637 |
| `metal/f32/one_hot/small/b2_n8_c2` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n8_c2.json) | 0.3541 / 0.278 | [passed](program_peer/metal__f32__one_hot__small__b2_n8_c2.json) | 0.331 / 0.2599 |
| `metal/f32/one_hot/small/b2_n8_c3` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n8_c3.json) | 0.316 / 0.2527 | [passed](program_peer/metal__f32__one_hot__small__b2_n8_c3.json) | 0.3221 / 0.2636 |
| `metal/f32/one_hot/small/b2_n8_c7` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n8_c7.json) | 0.3757 / 0.2587 | [passed](program_peer/metal__f32__one_hot__small__b2_n8_c7.json) | 0.3723 / 0.2553 |
| `metal/f32/one_hot/small/b2_n8_c16` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n8_c16.json) | 0.3711 / 0.2583 | [passed](program_peer/metal__f32__one_hot__small__b2_n8_c16.json) | 0.3582 / 0.2614 |
| `metal/f32/one_hot/small/b2_n8_c17` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n8_c17.json) | 0.3915 / 0.2586 | [passed](program_peer/metal__f32__one_hot__small__b2_n8_c17.json) | 0.4028 / 0.2654 |
| `metal/f32/one_hot/small/b2_n8_c31` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n8_c31.json) | 0.4222 / 0.2619 | [passed](program_peer/metal__f32__one_hot__small__b2_n8_c31.json) | 0.3987 / 0.2641 |
| `metal/f32/one_hot/small/b2_n8_c32` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n8_c32.json) | 0.4042 / 0.2662 | [passed](program_peer/metal__f32__one_hot__small__b2_n8_c32.json) | 0.4198 / 0.2649 |
| `metal/f32/one_hot/small/b2_n8_c33` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n8_c33.json) | 0.3894 / 0.2654 | [passed](program_peer/metal__f32__one_hot__small__b2_n8_c33.json) | 0.3832 / 0.283 |
| `metal/f32/one_hot/small/b2_n17_c2` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n17_c2.json) | 0.3596 / 0.2608 | [passed](program_peer/metal__f32__one_hot__small__b2_n17_c2.json) | 0.3791 / 0.2632 |
| `metal/f32/one_hot/small/b2_n17_c3` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n17_c3.json) | 0.3574 / 0.2542 | [passed](program_peer/metal__f32__one_hot__small__b2_n17_c3.json) | 0.3945 / 0.2559 |
| `metal/f32/one_hot/small/b2_n17_c7` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n17_c7.json) | 0.3898 / 0.261 | [passed](program_peer/metal__f32__one_hot__small__b2_n17_c7.json) | 0.4103 / 0.2663 |
| `metal/f32/one_hot/small/b2_n17_c16` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n17_c16.json) | 0.4417 / 0.2636 | [passed](program_peer/metal__f32__one_hot__small__b2_n17_c16.json) | 0.4376 / 0.2689 |
| `metal/f32/one_hot/small/b2_n17_c17` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n17_c17.json) | 0.4464 / 0.2657 | [passed](program_peer/metal__f32__one_hot__small__b2_n17_c17.json) | 0.4365 / 0.2758 |
| `metal/f32/one_hot/small/b2_n17_c31` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n17_c31.json) | 0.4804 / 0.2688 | [passed](program_peer/metal__f32__one_hot__small__b2_n17_c31.json) | 0.5047 / 0.2775 |
| `metal/f32/one_hot/small/b2_n17_c32` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n17_c32.json) | 0.507 / 0.2687 | [passed](program_peer/metal__f32__one_hot__small__b2_n17_c32.json) | 0.5034 / 0.273 |
| `metal/f32/one_hot/small/b2_n17_c33` | correct | [passed](operator_peer/metal__f32__one_hot__small__b2_n17_c33.json) | 0.5025 / 0.2717 | [passed](program_peer/metal__f32__one_hot__small__b2_n17_c33.json) | 0.506 / 0.2736 |
| `metal/f32/one_hot/large` | correct | [passed](operator_peer/metal__f32__one_hot__large.json) | 0.3316 / 0.2534 | [passed](program_peer/metal__f32__one_hot__large.json) | 0.3626 / 0.2585 |
| `metal/f32/one_hot/large/b4_n1_c3` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n1_c3.json) | 0.347 / 0.2708 | [passed](program_peer/metal__f32__one_hot__large__b4_n1_c3.json) | 0.357 / 0.2645 |
| `metal/f32/one_hot/large/b4_n1_c7` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n1_c7.json) | 0.3236 / 0.2558 | [passed](program_peer/metal__f32__one_hot__large__b4_n1_c7.json) | 0.3741 / 0.2597 |
| `metal/f32/one_hot/large/b4_n1_c16` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n1_c16.json) | 0.3467 / 0.2848 | [passed](program_peer/metal__f32__one_hot__large__b4_n1_c16.json) | 0.3512 / 0.2608 |
| `metal/f32/one_hot/large/b4_n1_c17` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n1_c17.json) | 0.3376 / 0.2595 | [passed](program_peer/metal__f32__one_hot__large__b4_n1_c17.json) | 0.3749 / 0.2608 |
| `metal/f32/one_hot/large/b4_n1_c31` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n1_c31.json) | 0.3507 / 0.2589 | [passed](program_peer/metal__f32__one_hot__large__b4_n1_c31.json) | 0.3646 / 0.2582 |
| `metal/f32/one_hot/large/b4_n1_c32` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n1_c32.json) | 0.3512 / 0.2573 | [passed](program_peer/metal__f32__one_hot__large__b4_n1_c32.json) | 0.3768 / 0.2634 |
| `metal/f32/one_hot/large/b4_n1_c33` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n1_c33.json) | 0.3519 / 0.2592 | [passed](program_peer/metal__f32__one_hot__large__b4_n1_c33.json) | 0.3617 / 0.2572 |
| `metal/f32/one_hot/large/b4_n3_c2` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n3_c2.json) | 0.3543 / 0.2587 | [passed](program_peer/metal__f32__one_hot__large__b4_n3_c2.json) | 0.3158 / 0.2597 |
| `metal/f32/one_hot/large/b4_n3_c3` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n3_c3.json) | 0.3465 / 0.2533 | [passed](program_peer/metal__f32__one_hot__large__b4_n3_c3.json) | 0.3746 / 0.2575 |
| `metal/f32/one_hot/large/b4_n3_c7` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n3_c7.json) | 0.3673 / 0.29 | [passed](program_peer/metal__f32__one_hot__large__b4_n3_c7.json) | 0.3661 / 0.2618 |
| `metal/f32/one_hot/large/b4_n3_c16` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n3_c16.json) | 0.3726 / 0.2576 | [passed](program_peer/metal__f32__one_hot__large__b4_n3_c16.json) | 0.3674 / 0.259 |
| `metal/f32/one_hot/large/b4_n3_c17` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n3_c17.json) | 0.3435 / 0.2653 | [passed](program_peer/metal__f32__one_hot__large__b4_n3_c17.json) | 0.3713 / 0.2675 |
| `metal/f32/one_hot/large/b4_n3_c31` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n3_c31.json) | 0.3743 / 0.2697 | [passed](program_peer/metal__f32__one_hot__large__b4_n3_c31.json) | 0.4077 / 0.2698 |
| `metal/f32/one_hot/large/b4_n3_c32` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n3_c32.json) | 0.3833 / 0.2528 | [passed](program_peer/metal__f32__one_hot__large__b4_n3_c32.json) | 0.4102 / 0.2723 |
| `metal/f32/one_hot/large/b4_n3_c33` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n3_c33.json) | 0.3933 / 0.2685 | [passed](program_peer/metal__f32__one_hot__large__b4_n3_c33.json) | 0.4147 / 0.2677 |
| `metal/f32/one_hot/large/b4_n8_c2` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n8_c2.json) | 0.3678 / 0.2793 | [passed](program_peer/metal__f32__one_hot__large__b4_n8_c2.json) | 0.3376 / 0.265 |
| `metal/f32/one_hot/large/b4_n8_c3` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n8_c3.json) | 0.3901 / 0.2606 | [passed](program_peer/metal__f32__one_hot__large__b4_n8_c3.json) | 0.3682 / 0.2521 |
| `metal/f32/one_hot/large/b4_n8_c7` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n8_c7.json) | 0.3597 / 0.2598 | [passed](program_peer/metal__f32__one_hot__large__b4_n8_c7.json) | 0.4086 / 0.2621 |
| `metal/f32/one_hot/large/b4_n8_c16` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n8_c16.json) | 0.4336 / 0.2697 | [passed](program_peer/metal__f32__one_hot__large__b4_n8_c16.json) | 0.4376 / 0.2677 |
| `metal/f32/one_hot/large/b4_n8_c17` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n8_c17.json) | 0.4366 / 0.2676 | [passed](program_peer/metal__f32__one_hot__large__b4_n8_c17.json) | 0.4496 / 0.2702 |
| `metal/f32/one_hot/large/b4_n8_c31` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n8_c31.json) | 0.4998 / 0.2789 | [passed](program_peer/metal__f32__one_hot__large__b4_n8_c31.json) | 0.4639 / 0.2755 |
| `metal/f32/one_hot/large/b4_n8_c32` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n8_c32.json) | 0.5018 / 0.2763 | [passed](program_peer/metal__f32__one_hot__large__b4_n8_c32.json) | 0.5011 / 0.2782 |
| `metal/f32/one_hot/large/b4_n8_c33` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n8_c33.json) | 0.4922 / 0.2744 | [passed](program_peer/metal__f32__one_hot__large__b4_n8_c33.json) | 0.4909 / 0.2787 |
| `metal/f32/one_hot/large/b4_n17_c2` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n17_c2.json) | 0.4014 / 0.261 | [passed](program_peer/metal__f32__one_hot__large__b4_n17_c2.json) | 0.4088 / 0.26 |
| `metal/f32/one_hot/large/b4_n17_c3` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n17_c3.json) | 0.4043 / 0.2728 | [passed](program_peer/metal__f32__one_hot__large__b4_n17_c3.json) | 0.4262 / 0.263 |
| `metal/f32/one_hot/large/b4_n17_c7` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n17_c7.json) | 0.4628 / 0.269 | [passed](program_peer/metal__f32__one_hot__large__b4_n17_c7.json) | 0.4724 / 0.2691 |
| `metal/f32/one_hot/large/b4_n17_c16` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n17_c16.json) | 0.5262 / 0.2716 | [passed](program_peer/metal__f32__one_hot__large__b4_n17_c16.json) | 0.5008 / 0.3117 |
| `metal/f32/one_hot/large/b4_n17_c17` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n17_c17.json) | 0.5415 / 0.2872 | [passed](program_peer/metal__f32__one_hot__large__b4_n17_c17.json) | 0.541 / 0.2774 |
| `metal/f32/one_hot/large/b4_n17_c31` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n17_c31.json) | 0.6375 / 0.2993 | [passed](program_peer/metal__f32__one_hot__large__b4_n17_c31.json) | 0.6396 / 0.3205 |
| `metal/f32/one_hot/large/b4_n17_c32` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n17_c32.json) | 0.6387 / 0.3206 | [passed](program_peer/metal__f32__one_hot__large__b4_n17_c32.json) | 0.6505 / 0.3215 |
| `metal/f32/one_hot/large/b4_n17_c33` | correct | [passed](operator_peer/metal__f32__one_hot__large__b4_n17_c33.json) | 0.6417 / 0.3228 | [passed](program_peer/metal__f32__one_hot__large__b4_n17_c33.json) | 0.6471 / 0.3191 |
| `metal/f32/one_hot/large/b8_n1_c2` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n1_c2.json) | 0.3582 / 0.2612 | [passed](program_peer/metal__f32__one_hot__large__b8_n1_c2.json) | 0.3705 / 0.2594 |
| `metal/f32/one_hot/large/b8_n1_c3` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n1_c3.json) | 0.3638 / 0.2669 | [passed](program_peer/metal__f32__one_hot__large__b8_n1_c3.json) | 0.3733 / 0.2595 |
| `metal/f32/one_hot/large/b8_n1_c7` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n1_c7.json) | 0.3208 / 0.2532 | [passed](program_peer/metal__f32__one_hot__large__b8_n1_c7.json) | 0.3659 / 0.2595 |
| `metal/f32/one_hot/large/b8_n1_c16` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n1_c16.json) | 0.3498 / 0.2468 | [passed](program_peer/metal__f32__one_hot__large__b8_n1_c16.json) | 0.3808 / 0.2544 |
| `metal/f32/one_hot/large/b8_n1_c17` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n1_c17.json) | 0.3585 / 0.2559 | [passed](program_peer/metal__f32__one_hot__large__b8_n1_c17.json) | 0.3839 / 0.2619 |
| `metal/f32/one_hot/large/b8_n1_c31` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n1_c31.json) | 0.3745 / 0.2659 | [passed](program_peer/metal__f32__one_hot__large__b8_n1_c31.json) | 0.3776 / 0.2652 |
| `metal/f32/one_hot/large/b8_n1_c32` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n1_c32.json) | 0.3689 / 0.2589 | [passed](program_peer/metal__f32__one_hot__large__b8_n1_c32.json) | 0.3949 / 0.272 |
| `metal/f32/one_hot/large/b8_n1_c33` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n1_c33.json) | 0.3709 / 0.2603 | [passed](program_peer/metal__f32__one_hot__large__b8_n1_c33.json) | 0.3975 / 0.2625 |
| `metal/f32/one_hot/large/b8_n3_c2` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n3_c2.json) | 0.3646 / 0.2597 | [passed](program_peer/metal__f32__one_hot__large__b8_n3_c2.json) | 0.3362 / 0.2603 |
| `metal/f32/one_hot/large/b8_n3_c3` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n3_c3.json) | 0.3755 / 0.2601 | [passed](program_peer/metal__f32__one_hot__large__b8_n3_c3.json) | 0.3565 / 0.2656 |
| `metal/f32/one_hot/large/b8_n3_c7` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n3_c7.json) | 0.3519 / 0.2481 | [passed](program_peer/metal__f32__one_hot__large__b8_n3_c7.json) | 0.3966 / 0.2581 |
| `metal/f32/one_hot/large/b8_n3_c16` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n3_c16.json) | 0.4033 / 0.2653 | [passed](program_peer/metal__f32__one_hot__large__b8_n3_c16.json) | 0.3958 / 0.2619 |
| `metal/f32/one_hot/large/b8_n3_c17` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n3_c17.json) | 0.4078 / 0.2614 | [passed](program_peer/metal__f32__one_hot__large__b8_n3_c17.json) | 0.412 / 0.2679 |
| `metal/f32/one_hot/large/b8_n3_c31` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n3_c31.json) | 0.4535 / 0.2785 | [passed](program_peer/metal__f32__one_hot__large__b8_n3_c31.json) | 0.4608 / 0.2772 |
| `metal/f32/one_hot/large/b8_n3_c32` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n3_c32.json) | 0.4542 / 0.2779 | [passed](program_peer/metal__f32__one_hot__large__b8_n3_c32.json) | 0.4658 / 0.3106 |
| `metal/f32/one_hot/large/b8_n3_c33` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n3_c33.json) | 0.4621 / 0.2653 | [passed](program_peer/metal__f32__one_hot__large__b8_n3_c33.json) | 0.4183 / 0.2642 |
| `metal/f32/one_hot/large/b8_n8_c2` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n8_c2.json) | 0.3965 / 0.2742 | [passed](program_peer/metal__f32__one_hot__large__b8_n8_c2.json) | 0.415 / 0.2633 |
| `metal/f32/one_hot/large/b8_n8_c3` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n8_c3.json) | 0.4126 / 0.2644 | [passed](program_peer/metal__f32__one_hot__large__b8_n8_c3.json) | 0.425 / 0.2826 |
| `metal/f32/one_hot/large/b8_n8_c7` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n8_c7.json) | 0.4596 / 0.2673 | [passed](program_peer/metal__f32__one_hot__large__b8_n8_c7.json) | 0.4487 / 0.2608 |
| `metal/f32/one_hot/large/b8_n8_c16` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n8_c16.json) | 0.5192 / 0.2899 | [passed](program_peer/metal__f32__one_hot__large__b8_n8_c16.json) | 0.4979 / 0.3096 |
| `metal/f32/one_hot/large/b8_n8_c17` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n8_c17.json) | 0.5364 / 0.2719 | [passed](program_peer/metal__f32__one_hot__large__b8_n8_c17.json) | 0.5483 / 0.3088 |
| `metal/f32/one_hot/large/b8_n8_c31` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n8_c31.json) | 0.6247 / 0.3191 | [passed](program_peer/metal__f32__one_hot__large__b8_n8_c31.json) | 0.6361 / 0.2964 |
| `metal/f32/one_hot/large/b8_n8_c32` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n8_c32.json) | 0.6269 / 0.2828 | [passed](program_peer/metal__f32__one_hot__large__b8_n8_c32.json) | 0.6422 / 0.3166 |
| `metal/f32/one_hot/large/b8_n8_c33` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n8_c33.json) | 0.6332 / 0.2918 | [passed](program_peer/metal__f32__one_hot__large__b8_n8_c33.json) | 0.6447 / 0.3214 |
| `metal/f32/one_hot/large/b8_n17_c2` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n17_c2.json) | 0.4513 / 0.2599 | [passed](program_peer/metal__f32__one_hot__large__b8_n17_c2.json) | 0.4475 / 0.2672 |
| `metal/f32/one_hot/large/b8_n17_c3` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n17_c3.json) | 0.486 / 0.269 | [passed](program_peer/metal__f32__one_hot__large__b8_n17_c3.json) | 0.4862 / 0.2779 |
| `metal/f32/one_hot/large/b8_n17_c7` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n17_c7.json) | 0.578 / 0.3028 | [passed](program_peer/metal__f32__one_hot__large__b8_n17_c7.json) | 0.5748 / 0.2749 |
| `metal/f32/one_hot/large/b8_n17_c16` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n17_c16.json) | 0.7015 / 0.3128 | [passed](program_peer/metal__f32__one_hot__large__b8_n17_c16.json) | 0.7105 / 0.293 |
| `metal/f32/one_hot/large/b8_n17_c17` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n17_c17.json) | 0.7216 / 0.2971 | [passed](program_peer/metal__f32__one_hot__large__b8_n17_c17.json) | 0.7347 / 0.3235 |
| `metal/f32/one_hot/large/b8_n17_c31` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n17_c31.json) | 0.9243 / 0.3415 | [passed](program_peer/metal__f32__one_hot__large__b8_n17_c31.json) | 0.9378 / 0.3503 |
| `metal/f32/one_hot/large/b8_n17_c32` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n17_c32.json) | 0.9336 / 0.3364 | [passed](program_peer/metal__f32__one_hot__large__b8_n17_c32.json) | 0.9336 / 0.3568 |
| `metal/f32/one_hot/large/b8_n17_c33` | correct | [passed](operator_peer/metal__f32__one_hot__large__b8_n17_c33.json) | 0.9537 / 0.3532 | [passed](program_peer/metal__f32__one_hot__large__b8_n17_c33.json) | 0.9776 / 0.3554 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
