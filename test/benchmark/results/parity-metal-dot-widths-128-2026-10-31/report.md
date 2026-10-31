# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `ab9dbf8b6ed82e5d549d78eb5ed1bdeb0922220aac1fbfd7249bb91ae60aba0b`  
Binary SHA-256: `d994eecd1097509e3b0687a82f8722862269b510bfcbf89d818e56dbc16c8125`  
Timing: 10 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 128
- Numerical capability passed: 128
- Operator parity and timing passed: 125
- Program parity and timing passed: 125
- Probe failures or incorrect: 0
- Operator incorrect or error: 3
- Program incorrect or error: 3

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/dot/dot_or_vector/n1` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n1.json) | 0.3317 / 0.2075 | [passed](program_peer/metal__f32__dot__dot_or_vector__n1.json) | 0.3123 / 0.2161 |
| `metal/f32/dot/dot_or_vector/n2` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n2.json) | 0.326 / 0.2156 | [passed](program_peer/metal__f32__dot__dot_or_vector__n2.json) | 0.359 / 0.2056 |
| `metal/f32/dot/dot_or_vector/n3` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n3.json) | 0.3359 / 0.2002 | [passed](program_peer/metal__f32__dot__dot_or_vector__n3.json) | 0.3562 / 0.1948 |
| `metal/f32/dot/dot_or_vector/n4` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n4.json) | 0.3303 / 0.2011 | [passed](program_peer/metal__f32__dot__dot_or_vector__n4.json) | 0.3063 / 0.2034 |
| `metal/f32/dot/dot_or_vector/n5` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n5.json) | 0.3341 / 0.1976 | [passed](program_peer/metal__f32__dot__dot_or_vector__n5.json) | 0.3537 / 0.2013 |
| `metal/f32/dot/dot_or_vector/n6` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n6.json) | 0.3302 / 0.2046 | [passed](program_peer/metal__f32__dot__dot_or_vector__n6.json) | 0.355 / 0.1985 |
| `metal/f32/dot/dot_or_vector/n7` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n7.json) | 0.3327 / 0.1995 | [passed](program_peer/metal__f32__dot__dot_or_vector__n7.json) | 0.3439 / 0.2019 |
| `metal/f32/dot/dot_or_vector/n8` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n8.json) | 0.3364 / 0.1994 | [passed](program_peer/metal__f32__dot__dot_or_vector__n8.json) | 0.3167 / 0.1912 |
| `metal/f32/dot/dot_or_vector/n9` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n9.json) | 0.3329 / 0.2046 | [passed](program_peer/metal__f32__dot__dot_or_vector__n9.json) | 0.3626 / 0.1914 |
| `metal/f32/dot/dot_or_vector/n10` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n10.json) | 0.344 / 0.1913 | [passed](program_peer/metal__f32__dot__dot_or_vector__n10.json) | 0.3578 / 0.2019 |
| `metal/f32/dot/dot_or_vector/n11` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n11.json) | 0.3466 / 0.206 | [passed](program_peer/metal__f32__dot__dot_or_vector__n11.json) | 0.3546 / 0.2037 |
| `metal/f32/dot/dot_or_vector/n12` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n12.json) | 0.3549 / 0.2044 | [passed](program_peer/metal__f32__dot__dot_or_vector__n12.json) | 0.3358 / 0.1881 |
| `metal/f32/dot/dot_or_vector/n13` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n13.json) | 0.3414 / 0.1977 | [passed](program_peer/metal__f32__dot__dot_or_vector__n13.json) | 0.3602 / 0.1894 |
| `metal/f32/dot/dot_or_vector/n14` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n14.json) | 0.3519 / 0.2022 | [passed](program_peer/metal__f32__dot__dot_or_vector__n14.json) | 0.3613 / 0.1996 |
| `metal/f32/dot/dot_or_vector/n15` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n15.json) | 0.3484 / 0.2036 | [passed](program_peer/metal__f32__dot__dot_or_vector__n15.json) | 0.3137 / 0.2029 |
| `metal/f32/dot/dot_or_vector` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector.json) | 0.3552 / 0.1983 | [passed](program_peer/metal__f32__dot__dot_or_vector.json) | 0.3599 / 0.1991 |
| `metal/f32/dot/dot_or_vector/n17` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n17.json) | 0.3568 / 0.1966 | [passed](program_peer/metal__f32__dot__dot_or_vector__n17.json) | 0.3629 / 0.1988 |
| `metal/f32/dot/dot_or_vector/n18` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n18.json) | 0.3525 / 0.2044 | [passed](program_peer/metal__f32__dot__dot_or_vector__n18.json) | 0.3615 / 0.2072 |
| `metal/f32/dot/dot_or_vector/n19` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n19.json) | 0.3001 / 0.2025 | [passed](program_peer/metal__f32__dot__dot_or_vector__n19.json) | 0.3603 / 0.2087 |
| `metal/f32/dot/dot_or_vector/n20` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n20.json) | 0.3447 / 0.2011 | [passed](program_peer/metal__f32__dot__dot_or_vector__n20.json) | 0.3418 / 0.1922 |
| `metal/f32/dot/dot_or_vector/n21` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n21.json) | 0.3507 / 0.1991 | [passed](program_peer/metal__f32__dot__dot_or_vector__n21.json) | 0.365 / 0.2047 |
| `metal/f32/dot/dot_or_vector/n22` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n22.json) | 0.3365 / 0.1972 | [passed](program_peer/metal__f32__dot__dot_or_vector__n22.json) | 0.3624 / 0.2044 |
| `metal/f32/dot/dot_or_vector/n23` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n23.json) | 0.3082 / 0.2006 | [passed](program_peer/metal__f32__dot__dot_or_vector__n23.json) | 0.3463 / 0.2051 |
| `metal/f32/dot/dot_or_vector/n24` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n24.json) | 0.3308 / 0.1981 | [passed](program_peer/metal__f32__dot__dot_or_vector__n24.json) | 0.3674 / 0.1992 |
| `metal/f32/dot/dot_or_vector/n25` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n25.json) | 0.33 / 0.19 | [passed](program_peer/metal__f32__dot__dot_or_vector__n25.json) | 0.3624 / 0.206 |
| `metal/f32/dot/dot_or_vector/n26` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n26.json) | 0.3562 / 0.2043 | [passed](program_peer/metal__f32__dot__dot_or_vector__n26.json) | 0.3678 / 0.1959 |
| `metal/f32/dot/dot_or_vector/n27` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n27.json) | 0.3263 / 0.1956 | [passed](program_peer/metal__f32__dot__dot_or_vector__n27.json) | 0.3663 / 0.2013 |
| `metal/f32/dot/dot_or_vector/n28` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n28.json) | 0.3565 / 0.2025 | [passed](program_peer/metal__f32__dot__dot_or_vector__n28.json) | 0.362 / 0.2027 |
| `metal/f32/dot/dot_or_vector/n29` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n29.json) | 0.3166 / 0.1923 | [passed](program_peer/metal__f32__dot__dot_or_vector__n29.json) | 0.3549 / 0.2064 |
| `metal/f32/dot/dot_or_vector/n30` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n30.json) | 0.3601 / 0.2008 | [passed](program_peer/metal__f32__dot__dot_or_vector__n30.json) | 0.3362 / 0.2061 |
| `metal/f32/dot/dot_or_vector/n31` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n31.json) | 0.3615 / 0.1994 | [passed](program_peer/metal__f32__dot__dot_or_vector__n31.json) | 0.3658 / 0.1977 |
| `metal/f32/dot/dot_or_vector/n32` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n32.json) | 0.3595 / 0.195 | [passed](program_peer/metal__f32__dot__dot_or_vector__n32.json) | 0.3666 / 0.1999 |
| `metal/f32/dot/dot_or_vector/n33` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n33.json) | 0.3634 / 0.2008 | [passed](program_peer/metal__f32__dot__dot_or_vector__n33.json) | 0.377 / 0.2027 |
| `metal/f32/dot/dot_or_vector/n34` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n34.json) | 0.3609 / 0.2 | [passed](program_peer/metal__f32__dot__dot_or_vector__n34.json) | 0.3608 / 0.1936 |
| `metal/f32/dot/dot_or_vector/n35` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n35.json) | 0.36 / 0.2041 | [passed](program_peer/metal__f32__dot__dot_or_vector__n35.json) | 0.3741 / 0.2032 |
| `metal/f32/dot/dot_or_vector/n36` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n36.json) | 0.3604 / 0.1978 | [passed](program_peer/metal__f32__dot__dot_or_vector__n36.json) | 0.3677 / 0.2063 |
| `metal/f32/dot/dot_or_vector/n37` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n37.json) | 0.3636 / 0.1965 | [passed](program_peer/metal__f32__dot__dot_or_vector__n37.json) | 0.3719 / 0.1948 |
| `metal/f32/dot/dot_or_vector/n38` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n38.json) | 0.3706 / 0.2028 | [passed](program_peer/metal__f32__dot__dot_or_vector__n38.json) | 0.3611 / 0.1968 |
| `metal/f32/dot/dot_or_vector/n39` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n39.json) | 0.366 / 0.2042 | [passed](program_peer/metal__f32__dot__dot_or_vector__n39.json) | 0.3646 / 0.203 |
| `metal/f32/dot/dot_or_vector/n40` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n40.json) | 0.3613 / 0.2013 | [passed](program_peer/metal__f32__dot__dot_or_vector__n40.json) | 0.3712 / 0.2028 |
| `metal/f32/dot/dot_or_vector/n41` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n41.json) | 0.3639 / 0.1977 | [passed](program_peer/metal__f32__dot__dot_or_vector__n41.json) | 0.3415 / 0.2008 |
| `metal/f32/dot/dot_or_vector/n42` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n42.json) | 0.3649 / 0.2055 | [passed](program_peer/metal__f32__dot__dot_or_vector__n42.json) | 0.3657 / 0.1957 |
| `metal/f32/dot/dot_or_vector/n43` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n43.json) | 0.3376 / 0.2016 | [passed](program_peer/metal__f32__dot__dot_or_vector__n43.json) | 0.3764 / 0.2051 |
| `metal/f32/dot/dot_or_vector/n44` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n44.json) | 0.3701 / 0.2037 | [passed](program_peer/metal__f32__dot__dot_or_vector__n44.json) | 0.368 / 0.2046 |
| `metal/f32/dot/dot_or_vector/n45` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n45.json) | 0.3648 / 0.1952 | [passed](program_peer/metal__f32__dot__dot_or_vector__n45.json) | 0.3748 / 0.1964 |
| `metal/f32/dot/dot_or_vector/n46` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n46.json) | 0.3654 / 0.1961 | [passed](program_peer/metal__f32__dot__dot_or_vector__n46.json) | 0.371 / 0.2039 |
| `metal/f32/dot/dot_or_vector/n47` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n47.json) | 0.3545 / 0.2062 | [passed](program_peer/metal__f32__dot__dot_or_vector__n47.json) | 0.3618 / 0.2071 |
| `metal/f32/dot/dot_or_vector/n48` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n48.json) | 0.333 / 0.2011 | [passed](program_peer/metal__f32__dot__dot_or_vector__n48.json) | 0.351 / 0.1932 |
| `metal/f32/dot/dot_or_vector/n63` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n63.json) | 0.3777 / 0.2022 | [passed](program_peer/metal__f32__dot__dot_or_vector__n63.json) | 0.3765 / 0.1986 |
| `metal/f32/dot/dot_or_vector/n64` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n64.json) | 0.3754 / 0.1961 | [passed](program_peer/metal__f32__dot__dot_or_vector__n64.json) | 0.3551 / 0.2082 |
| `metal/f32/dot/dot_or_vector/n65` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n65.json) | 0.3762 / 0.2009 | [passed](program_peer/metal__f32__dot__dot_or_vector__n65.json) | 0.3745 / 0.1978 |
| `metal/f32/dot/dot_or_vector/n127` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n127.json) | 0.3783 / 0.2006 | [passed](program_peer/metal__f32__dot__dot_or_vector__n127.json) | 0.3667 / 0.2002 |
| `metal/f32/dot/dot_or_vector/n128` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n128.json) | 0.3597 / 0.1934 | [passed](program_peer/metal__f32__dot__dot_or_vector__n128.json) | 0.4049 / 0.1951 |
| `metal/f32/dot/dot_or_vector/n129` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n129.json) | 0.3722 / 0.1957 | [passed](program_peer/metal__f32__dot__dot_or_vector__n129.json) | 0.4069 / 0.1974 |
| `metal/f32/dot/dot_or_vector/n255` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n255.json) | 0.3634 / 0.2048 | [passed](program_peer/metal__f32__dot__dot_or_vector__n255.json) | 0.3708 / 0.1988 |
| `metal/f32/dot/dot_or_vector/n256` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n256.json) | 0.3964 / 0.196 | [passed](program_peer/metal__f32__dot__dot_or_vector__n256.json) | 0.3673 / 0.2036 |
| `metal/f32/dot/dot_or_vector/n257` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n257.json) | 0.3928 / 0.1932 | [passed](program_peer/metal__f32__dot__dot_or_vector__n257.json) | 0.4782 / 0.204 |
| `metal/f32/dot/dot_or_vector/n511` | correct | [incorrect](operator_peer_failures/metal__f32__dot__dot_or_vector__n511.json) | 0.4126 / 0.1968 | [incorrect](program_peer_failures/metal__f32__dot__dot_or_vector__n511.json) | 0.4003 / 0.2012 |
| `metal/f32/dot/dot_or_vector/n512` | correct | [incorrect](operator_peer_failures/metal__f32__dot__dot_or_vector__n512.json) | 0.381 / 0.203 | [incorrect](program_peer_failures/metal__f32__dot__dot_or_vector__n512.json) | 0.3831 / 0.2 |
| `metal/f32/dot/dot_or_vector/n513` | correct | [incorrect](operator_peer_failures/metal__f32__dot__dot_or_vector__n513.json) | 0.5225 / 0.2032 | [incorrect](program_peer_failures/metal__f32__dot__dot_or_vector__n513.json) | 0.3891 / 0.1987 |
| `metal/f32/dot/dot_or_vector/n1023` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n1023.json) | 0.4292 / 0.2026 | [passed](program_peer/metal__f32__dot__dot_or_vector__n1023.json) | 0.4329 / 0.1999 |
| `metal/f32/dot/dot_or_vector/n1024` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n1024.json) | 0.4295 / 0.2007 | [passed](program_peer/metal__f32__dot__dot_or_vector__n1024.json) | 0.4314 / 0.1976 |
| `metal/f32/dot/dot_or_vector/n1025` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n1025.json) | 0.4226 / 0.2022 | [passed](program_peer/metal__f32__dot__dot_or_vector__n1025.json) | 0.4071 / 0.198 |
| `metal/f32/dot/dot_or_vector/n2048` | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n2048.json) | 0.525 / 0.1981 | [passed](program_peer/metal__f32__dot__dot_or_vector__n2048.json) | 0.535 / 0.2018 |
| `metal/bf16/dot/dot_or_vector/n1` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n1.json) | 0.3572 / 0.199 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n1.json) | 0.3623 / 0.2031 |
| `metal/bf16/dot/dot_or_vector/n2` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n2.json) | 0.3555 / 0.1956 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n2.json) | 0.3671 / 0.2021 |
| `metal/bf16/dot/dot_or_vector/n3` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n3.json) | 0.3548 / 0.2032 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n3.json) | 0.355 / 0.1965 |
| `metal/bf16/dot/dot_or_vector/n4` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n4.json) | 0.3565 / 0.2512 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n4.json) | 0.3652 / 0.2036 |
| `metal/bf16/dot/dot_or_vector/n5` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n5.json) | 0.3417 / 0.2009 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n5.json) | 0.3557 / 0.2017 |
| `metal/bf16/dot/dot_or_vector/n6` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n6.json) | 0.3423 / 0.1987 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n6.json) | 0.3694 / 0.1955 |
| `metal/bf16/dot/dot_or_vector/n7` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n7.json) | 0.3598 / 0.1999 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n7.json) | 0.3551 / 0.2016 |
| `metal/bf16/dot/dot_or_vector/n8` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n8.json) | 0.3558 / 0.1936 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n8.json) | 0.3587 / 0.2017 |
| `metal/bf16/dot/dot_or_vector/n9` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n9.json) | 0.36 / 0.2039 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n9.json) | 0.3683 / 0.2059 |
| `metal/bf16/dot/dot_or_vector/n10` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n10.json) | 0.3376 / 0.1936 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n10.json) | 0.361 / 0.1977 |
| `metal/bf16/dot/dot_or_vector/n11` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n11.json) | 0.3575 / 0.1952 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n11.json) | 0.3652 / 0.202 |
| `metal/bf16/dot/dot_or_vector/n12` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n12.json) | 0.3577 / 0.2004 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n12.json) | 0.3291 / 0.2013 |
| `metal/bf16/dot/dot_or_vector/n13` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n13.json) | 0.3602 / 0.2059 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n13.json) | 0.3647 / 0.1956 |
| `metal/bf16/dot/dot_or_vector/n14` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n14.json) | 0.3597 / 0.1958 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n14.json) | 0.3389 / 0.2024 |
| `metal/bf16/dot/dot_or_vector/n15` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n15.json) | 0.3619 / 0.2024 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n15.json) | 0.3687 / 0.1983 |
| `metal/bf16/dot/dot_or_vector` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector.json) | 0.3611 / 0.2048 | [passed](program_peer/metal__bf16__dot__dot_or_vector.json) | 0.3671 / 0.1984 |
| `metal/bf16/dot/dot_or_vector/n17` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n17.json) | 0.3347 / 0.1958 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n17.json) | 0.3652 / 0.2004 |
| `metal/bf16/dot/dot_or_vector/n18` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n18.json) | 0.3539 / 0.2021 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n18.json) | 0.3668 / 0.2026 |
| `metal/bf16/dot/dot_or_vector/n19` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n19.json) | 0.3721 / 0.1919 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n19.json) | 0.3686 / 0.2029 |
| `metal/bf16/dot/dot_or_vector/n20` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n20.json) | 0.3533 / 0.1999 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n20.json) | 0.3487 / 0.2035 |
| `metal/bf16/dot/dot_or_vector/n21` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n21.json) | 0.3681 / 0.1994 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n21.json) | 0.3683 / 0.1999 |
| `metal/bf16/dot/dot_or_vector/n22` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n22.json) | 0.358 / 0.2045 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n22.json) | 0.3254 / 0.202 |
| `metal/bf16/dot/dot_or_vector/n23` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n23.json) | 0.3585 / 0.2055 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n23.json) | 0.3727 / 0.2056 |
| `metal/bf16/dot/dot_or_vector/n24` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n24.json) | 0.3229 / 0.1999 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n24.json) | 0.3562 / 0.2021 |
| `metal/bf16/dot/dot_or_vector/n25` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n25.json) | 0.3519 / 0.198 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n25.json) | 0.317 / 0.1935 |
| `metal/bf16/dot/dot_or_vector/n26` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n26.json) | 0.3553 / 0.2031 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n26.json) | 0.3771 / 0.2019 |
| `metal/bf16/dot/dot_or_vector/n27` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n27.json) | 0.3686 / 0.2027 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n27.json) | 0.3755 / 0.2014 |
| `metal/bf16/dot/dot_or_vector/n28` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n28.json) | 0.3584 / 0.1996 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n28.json) | 0.3737 / 0.1935 |
| `metal/bf16/dot/dot_or_vector/n29` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n29.json) | 0.3691 / 0.1961 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n29.json) | 0.3724 / 0.2037 |
| `metal/bf16/dot/dot_or_vector/n30` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n30.json) | 0.3544 / 0.2013 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n30.json) | 0.3792 / 0.206 |
| `metal/bf16/dot/dot_or_vector/n31` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n31.json) | 0.3716 / 0.2038 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n31.json) | 0.3478 / 0.1935 |
| `metal/bf16/dot/dot_or_vector/n32` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n32.json) | 0.3446 / 0.2015 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n32.json) | 0.3317 / 0.1897 |
| `metal/bf16/dot/dot_or_vector/n33` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n33.json) | 0.3722 / 0.195 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n33.json) | 0.3798 / 0.2021 |
| `metal/bf16/dot/dot_or_vector/n34` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n34.json) | 0.3696 / 0.2007 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n34.json) | 0.3761 / 0.2037 |
| `metal/bf16/dot/dot_or_vector/n35` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n35.json) | 0.3687 / 0.2024 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n35.json) | 0.3459 / 0.2085 |
| `metal/bf16/dot/dot_or_vector/n36` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n36.json) | 0.3766 / 0.1952 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n36.json) | 0.3761 / 0.1961 |
| `metal/bf16/dot/dot_or_vector/n37` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n37.json) | 0.3727 / 0.1939 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n37.json) | 0.3407 / 0.1995 |
| `metal/bf16/dot/dot_or_vector/n38` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n38.json) | 0.3765 / 0.199 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n38.json) | 0.3725 / 0.2043 |
| `metal/bf16/dot/dot_or_vector/n39` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n39.json) | 0.344 / 0.1991 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n39.json) | 0.3768 / 0.2019 |
| `metal/bf16/dot/dot_or_vector/n40` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n40.json) | 0.3756 / 0.201 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n40.json) | 0.3797 / 0.2063 |
| `metal/bf16/dot/dot_or_vector/n41` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n41.json) | 0.3695 / 0.1995 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n41.json) | 0.3738 / 0.1951 |
| `metal/bf16/dot/dot_or_vector/n42` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n42.json) | 0.3749 / 0.1972 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n42.json) | 0.3696 / 0.2019 |
| `metal/bf16/dot/dot_or_vector/n43` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n43.json) | 0.3351 / 0.2017 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n43.json) | 0.3878 / 0.204 |
| `metal/bf16/dot/dot_or_vector/n44` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n44.json) | 0.3765 / 0.1999 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n44.json) | 0.3875 / 0.1925 |
| `metal/bf16/dot/dot_or_vector/n45` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n45.json) | 0.3829 / 0.2048 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n45.json) | 0.3354 / 0.2025 |
| `metal/bf16/dot/dot_or_vector/n46` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n46.json) | 0.375 / 0.2021 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n46.json) | 0.362 / 0.2015 |
| `metal/bf16/dot/dot_or_vector/n47` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n47.json) | 0.3737 / 0.2012 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n47.json) | 0.3838 / 0.2019 |
| `metal/bf16/dot/dot_or_vector/n48` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n48.json) | 0.3754 / 0.1991 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n48.json) | 0.3838 / 0.2025 |
| `metal/bf16/dot/dot_or_vector/n63` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n63.json) | 0.3894 / 0.1991 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n63.json) | 0.386 / 0.1914 |
| `metal/bf16/dot/dot_or_vector/n64` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n64.json) | 0.3851 / 0.1893 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n64.json) | 0.3932 / 0.1968 |
| `metal/bf16/dot/dot_or_vector/n65` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n65.json) | 0.3853 / 0.2008 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n65.json) | 0.3909 / 0.2006 |
| `metal/bf16/dot/dot_or_vector/n127` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n127.json) | 0.4297 / 0.1947 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n127.json) | 0.3644 / 0.203 |
| `metal/bf16/dot/dot_or_vector/n128` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n128.json) | 0.3591 / 0.2026 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n128.json) | 0.3394 / 0.2032 |
| `metal/bf16/dot/dot_or_vector/n129` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n129.json) | 0.302 / 0.1958 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n129.json) | 0.3439 / 0.1917 |
| `metal/bf16/dot/dot_or_vector/n255` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n255.json) | 0.3597 / 0.1994 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n255.json) | 0.3725 / 0.2025 |
| `metal/bf16/dot/dot_or_vector/n256` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n256.json) | 0.336 / 0.1951 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n256.json) | 0.3564 / 0.1989 |
| `metal/bf16/dot/dot_or_vector/n257` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n257.json) | 0.3681 / 0.1951 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n257.json) | 0.3705 / 0.2024 |
| `metal/bf16/dot/dot_or_vector/n511` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n511.json) | 0.3634 / 0.2014 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n511.json) | 0.3729 / 0.2004 |
| `metal/bf16/dot/dot_or_vector/n512` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n512.json) | 0.3583 / 0.2021 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n512.json) | 0.375 / 0.1981 |
| `metal/bf16/dot/dot_or_vector/n513` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n513.json) | 0.362 / 0.1986 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n513.json) | 0.3636 / 0.2004 |
| `metal/bf16/dot/dot_or_vector/n1023` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n1023.json) | 0.3611 / 0.2027 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n1023.json) | 0.3601 / 0.2016 |
| `metal/bf16/dot/dot_or_vector/n1024` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n1024.json) | 0.3719 / 0.2057 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n1024.json) | 0.3718 / 0.2 |
| `metal/bf16/dot/dot_or_vector/n1025` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n1025.json) | 0.329 / 0.2032 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n1025.json) | 0.3848 / 0.191 |
| `metal/bf16/dot/dot_or_vector/n2048` | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n2048.json) | 0.3396 / 0.2016 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n2048.json) | 0.3777 / 0.1959 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
