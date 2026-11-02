# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `624da9250559f7a715d283da7c3c3921a115d0b0973b2c1044d6043bf78ff3db`  
Binary SHA-256: `f25caea82a00bafd045213e187e5aa8233a760c352d2f4df4d3079e60d2aaadd`  
Timing: 30 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 64
- Numerical check passed: 64
- Operator check and timing passed: 64
- Program check and timing passed: 64
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/dot/dot_or_vector/n1` | matrix | 1 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n1.json) | 0.1765 / 0.2078 | [passed](program_peer/metal__f32__dot__dot_or_vector__n1.json) | 0.1731 / 0.21 |
| `metal/f32/dot/dot_or_vector/n2` | matrix | 2 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n2.json) | 0.1658 / 0.2055 | [passed](program_peer/metal__f32__dot__dot_or_vector__n2.json) | 0.1927 / 0.2103 |
| `metal/f32/dot/dot_or_vector/n3` | matrix | 3 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n3.json) | 0.1904 / 0.2078 | [passed](program_peer/metal__f32__dot__dot_or_vector__n3.json) | 0.1877 / 0.2089 |
| `metal/f32/dot/dot_or_vector/n4` | matrix | 4 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n4.json) | 0.1799 / 0.2065 | [passed](program_peer/metal__f32__dot__dot_or_vector__n4.json) | 0.1735 / 0.2057 |
| `metal/f32/dot/dot_or_vector/n5` | matrix | 5 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n5.json) | 0.1716 / 0.2084 | [passed](program_peer/metal__f32__dot__dot_or_vector__n5.json) | 0.1752 / 0.2083 |
| `metal/f32/dot/dot_or_vector/n6` | matrix | 6 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n6.json) | 0.1787 / 0.2049 | [passed](program_peer/metal__f32__dot__dot_or_vector__n6.json) | 0.1706 / 0.2068 |
| `metal/f32/dot/dot_or_vector/n7` | matrix | 7 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n7.json) | 0.1727 / 0.2053 | [passed](program_peer/metal__f32__dot__dot_or_vector__n7.json) | 0.188 / 0.2069 |
| `metal/f32/dot/dot_or_vector/n8` | matrix | 8 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n8.json) | 0.1746 / 0.2051 | [passed](program_peer/metal__f32__dot__dot_or_vector__n8.json) | 0.1985 / 0.2092 |
| `metal/f32/dot/dot_or_vector/n9` | matrix | 9 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n9.json) | 0.1823 / 0.2049 | [passed](program_peer/metal__f32__dot__dot_or_vector__n9.json) | 0.1784 / 0.2087 |
| `metal/f32/dot/dot_or_vector/n10` | matrix | 10 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n10.json) | 0.1737 / 0.2059 | [passed](program_peer/metal__f32__dot__dot_or_vector__n10.json) | 0.1788 / 0.2042 |
| `metal/f32/dot/dot_or_vector/n11` | matrix | 11 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n11.json) | 0.1739 / 0.2046 | [passed](program_peer/metal__f32__dot__dot_or_vector__n11.json) | 0.1981 / 0.207 |
| `metal/f32/dot/dot_or_vector/n12` | matrix | 12 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n12.json) | 0.1891 / 0.2086 | [passed](program_peer/metal__f32__dot__dot_or_vector__n12.json) | 0.1731 / 0.2062 |
| `metal/f32/dot/dot_or_vector/n13` | matrix | 13 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n13.json) | 0.179 / 0.1993 | [passed](program_peer/metal__f32__dot__dot_or_vector__n13.json) | 0.1753 / 0.204 |
| `metal/f32/dot/dot_or_vector/n14` | matrix | 14 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n14.json) | 0.19 / 0.2045 | [passed](program_peer/metal__f32__dot__dot_or_vector__n14.json) | 0.1734 / 0.2072 |
| `metal/f32/dot/dot_or_vector/n15` | matrix | 15 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n15.json) | 0.1748 / 0.207 | [passed](program_peer/metal__f32__dot__dot_or_vector__n15.json) | 0.1838 / 0.216 |
| `metal/f32/dot/dot_or_vector` | matrix | 16 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector.json) | 0.1732 / 0.2045 | [passed](program_peer/metal__f32__dot__dot_or_vector.json) | 0.1909 / 0.2043 |
| `metal/f32/dot/dot_or_vector/n17` | matrix | 17 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n17.json) | 0.1816 / 0.1956 | [passed](program_peer/metal__f32__dot__dot_or_vector__n17.json) | 0.2044 / 0.2047 |
| `metal/f32/dot/dot_or_vector/n18` | matrix | 18 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n18.json) | 0.148 / 0.1795 | [passed](program_peer/metal__f32__dot__dot_or_vector__n18.json) | 0.1816 / 0.2045 |
| `metal/f32/dot/dot_or_vector/n19` | matrix | 19 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n19.json) | 0.1798 / 0.2104 | [passed](program_peer/metal__f32__dot__dot_or_vector__n19.json) | 0.1948 / 0.1983 |
| `metal/f32/dot/dot_or_vector/n20` | matrix | 20 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n20.json) | 0.1822 / 0.2059 | [passed](program_peer/metal__f32__dot__dot_or_vector__n20.json) | 0.1878 / 0.2063 |
| `metal/f32/dot/dot_or_vector/n21` | matrix | 21 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n21.json) | 0.1891 / 0.2077 | [passed](program_peer/metal__f32__dot__dot_or_vector__n21.json) | 0.1874 / 0.2084 |
| `metal/f32/dot/dot_or_vector/n22` | matrix | 22 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n22.json) | 0.1937 / 0.2069 | [passed](program_peer/metal__f32__dot__dot_or_vector__n22.json) | 0.1809 / 0.2048 |
| `metal/f32/dot/dot_or_vector/n23` | matrix | 23 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n23.json) | 0.1826 / 0.2072 | [passed](program_peer/metal__f32__dot__dot_or_vector__n23.json) | 0.1831 / 0.2074 |
| `metal/f32/dot/dot_or_vector/n24` | matrix | 24 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n24.json) | 0.175 / 0.2051 | [passed](program_peer/metal__f32__dot__dot_or_vector__n24.json) | 0.1966 / 0.2009 |
| `metal/f32/dot/dot_or_vector/n25` | matrix | 25 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n25.json) | 0.164 / 0.205 | [passed](program_peer/metal__f32__dot__dot_or_vector__n25.json) | 0.1845 / 0.2056 |
| `metal/f32/dot/dot_or_vector/n26` | matrix | 26 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n26.json) | 0.1913 / 0.2045 | [passed](program_peer/metal__f32__dot__dot_or_vector__n26.json) | 0.1836 / 0.2064 |
| `metal/f32/dot/dot_or_vector/n27` | matrix | 27 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n27.json) | 0.1765 / 0.1932 | [passed](program_peer/metal__f32__dot__dot_or_vector__n27.json) | 0.1867 / 0.2079 |
| `metal/f32/dot/dot_or_vector/n28` | matrix | 28 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n28.json) | 0.183 / 0.2069 | [passed](program_peer/metal__f32__dot__dot_or_vector__n28.json) | 0.1873 / 0.201 |
| `metal/f32/dot/dot_or_vector/n29` | matrix | 29 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n29.json) | 0.187 / 0.2014 | [passed](program_peer/metal__f32__dot__dot_or_vector__n29.json) | 0.206 / 0.2087 |
| `metal/f32/dot/dot_or_vector/n30` | matrix | 30 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n30.json) | 0.1973 / 0.2075 | [passed](program_peer/metal__f32__dot__dot_or_vector__n30.json) | 0.1961 / 0.2079 |
| `metal/f32/dot/dot_or_vector/n31` | matrix | 31 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n31.json) | 0.2042 / 0.208 | [passed](program_peer/metal__f32__dot__dot_or_vector__n31.json) | 0.196 / 0.2088 |
| `metal/f32/dot/dot_or_vector/n32` | matrix | 32 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n32.json) | 0.1997 / 0.2016 | [passed](program_peer/metal__f32__dot__dot_or_vector__n32.json) | 0.2054 / 0.2084 |
| `metal/f32/dot/dot_or_vector/n33` | matrix | 33 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n33.json) | 0.1792 / 0.2035 | [passed](program_peer/metal__f32__dot__dot_or_vector__n33.json) | 0.1864 / 0.209 |
| `metal/f32/dot/dot_or_vector/n34` | matrix | 34 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n34.json) | 0.1802 / 0.2052 | [passed](program_peer/metal__f32__dot__dot_or_vector__n34.json) | 0.1838 / 0.2086 |
| `metal/f32/dot/dot_or_vector/n35` | matrix | 35 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n35.json) | 0.1846 / 0.2038 | [passed](program_peer/metal__f32__dot__dot_or_vector__n35.json) | 0.2124 / 0.2079 |
| `metal/f32/dot/dot_or_vector/n36` | matrix | 36 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n36.json) | 0.2068 / 0.2037 | [passed](program_peer/metal__f32__dot__dot_or_vector__n36.json) | 0.1801 / 0.2101 |
| `metal/f32/dot/dot_or_vector/n37` | matrix | 37 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n37.json) | 0.1828 / 0.2066 | [passed](program_peer/metal__f32__dot__dot_or_vector__n37.json) | 0.189 / 0.2083 |
| `metal/f32/dot/dot_or_vector/n38` | matrix | 38 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n38.json) | 0.1936 / 0.2035 | [passed](program_peer/metal__f32__dot__dot_or_vector__n38.json) | 0.1904 / 0.2077 |
| `metal/f32/dot/dot_or_vector/n39` | matrix | 39 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n39.json) | 0.1967 / 0.2056 | [passed](program_peer/metal__f32__dot__dot_or_vector__n39.json) | 0.2002 / 0.2101 |
| `metal/f32/dot/dot_or_vector/n40` | matrix | 40 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n40.json) | 0.1848 / 0.203 | [passed](program_peer/metal__f32__dot__dot_or_vector__n40.json) | 0.1901 / 0.2079 |
| `metal/f32/dot/dot_or_vector/n41` | matrix | 41 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n41.json) | 0.1976 / 0.2061 | [passed](program_peer/metal__f32__dot__dot_or_vector__n41.json) | 0.224 / 0.2064 |
| `metal/f32/dot/dot_or_vector/n42` | matrix | 42 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n42.json) | 0.1949 / 0.2041 | [passed](program_peer/metal__f32__dot__dot_or_vector__n42.json) | 0.2094 / 0.2086 |
| `metal/f32/dot/dot_or_vector/n43` | matrix | 43 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n43.json) | 0.1822 / 0.2056 | [passed](program_peer/metal__f32__dot__dot_or_vector__n43.json) | 0.1825 / 0.202 |
| `metal/f32/dot/dot_or_vector/n44` | matrix | 44 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n44.json) | 0.1973 / 0.2062 | [passed](program_peer/metal__f32__dot__dot_or_vector__n44.json) | 0.1869 / 0.2048 |
| `metal/f32/dot/dot_or_vector/n45` | matrix | 45 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n45.json) | 0.1894 / 0.2063 | [passed](program_peer/metal__f32__dot__dot_or_vector__n45.json) | 0.1933 / 0.2088 |
| `metal/f32/dot/dot_or_vector/n46` | matrix | 46 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n46.json) | 0.1828 / 0.2047 | [passed](program_peer/metal__f32__dot__dot_or_vector__n46.json) | 0.1919 / 0.1983 |
| `metal/f32/dot/dot_or_vector/n47` | matrix | 47 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n47.json) | 0.1936 / 0.2039 | [passed](program_peer/metal__f32__dot__dot_or_vector__n47.json) | 0.2078 / 0.2084 |
| `metal/f32/dot/dot_or_vector/n48` | matrix | 48 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n48.json) | 0.1881 / 0.2029 | [passed](program_peer/metal__f32__dot__dot_or_vector__n48.json) | 0.1843 / 0.206 |
| `metal/f32/dot/dot_or_vector/n63` | matrix | 63 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n63.json) | 0.1998 / 0.2004 | [passed](program_peer/metal__f32__dot__dot_or_vector__n63.json) | 0.2054 / 0.2089 |
| `metal/f32/dot/dot_or_vector/n64` | matrix | 64 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n64.json) | 0.1965 / 0.205 | [passed](program_peer/metal__f32__dot__dot_or_vector__n64.json) | 0.2177 / 0.2072 |
| `metal/f32/dot/dot_or_vector/n65` | matrix | 65 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n65.json) | 0.1895 / 0.2065 | [passed](program_peer/metal__f32__dot__dot_or_vector__n65.json) | 0.2165 / 0.2048 |
| `metal/f32/dot/dot_or_vector/n127` | matrix | 127 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n127.json) | 0.2192 / 0.2045 | [passed](program_peer/metal__f32__dot__dot_or_vector__n127.json) | 0.222 / 0.205 |
| `metal/f32/dot/dot_or_vector/n128` | matrix | 128 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n128.json) | 0.1787 / 0.2048 | [passed](program_peer/metal__f32__dot__dot_or_vector__n128.json) | 0.1765 / 0.2068 |
| `metal/f32/dot/dot_or_vector/n129` | matrix | 129 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n129.json) | 0.1792 / 0.2062 | [passed](program_peer/metal__f32__dot__dot_or_vector__n129.json) | 0.1701 / 0.2098 |
| `metal/f32/dot/dot_or_vector/n255` | matrix | 255 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n255.json) | 0.2049 / 0.2043 | [passed](program_peer/metal__f32__dot__dot_or_vector__n255.json) | 0.1883 / 0.2111 |
| `metal/f32/dot/dot_or_vector/n256` | matrix | 256 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n256.json) | 0.1787 / 0.2074 | [passed](program_peer/metal__f32__dot__dot_or_vector__n256.json) | 0.2059 / 0.2057 |
| `metal/f32/dot/dot_or_vector/n257` | matrix | 257 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n257.json) | 0.1816 / 0.2011 | [passed](program_peer/metal__f32__dot__dot_or_vector__n257.json) | 0.1738 / 0.2128 |
| `metal/f32/dot/dot_or_vector/n511` | matrix | 511 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n511.json) | 0.1755 / 0.2069 | [passed](program_peer/metal__f32__dot__dot_or_vector__n511.json) | 0.1721 / 0.2096 |
| `metal/f32/dot/dot_or_vector/n512` | matrix | 512 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n512.json) | 0.1819 / 0.2079 | [passed](program_peer/metal__f32__dot__dot_or_vector__n512.json) | 0.1843 / 0.2087 |
| `metal/f32/dot/dot_or_vector/n513` | matrix | 513 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n513.json) | 0.1818 / 0.2035 | [passed](program_peer/metal__f32__dot__dot_or_vector__n513.json) | 0.195 / 0.2112 |
| `metal/f32/dot/dot_or_vector/n1023` | matrix | 1023 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n1023.json) | 0.1889 / 0.1968 | [passed](program_peer/metal__f32__dot__dot_or_vector__n1023.json) | 0.2173 / 0.2075 |
| `metal/f32/dot/dot_or_vector/n1024` | matrix | 1024 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n1024.json) | 0.1841 / 0.2069 | [passed](program_peer/metal__f32__dot__dot_or_vector__n1024.json) | 0.1838 / 0.2062 |
| `metal/f32/dot/dot_or_vector/n1025` | matrix | 1025 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n1025.json) | 0.1763 / 0.2055 | [passed](program_peer/metal__f32__dot__dot_or_vector__n1025.json) | 0.1912 / 0.2084 |
| `metal/f32/dot/dot_or_vector/n2048` | matrix | 2048 | dense | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__n2048.json) | 0.1803 / 0.2057 | [passed](program_peer/metal__f32__dot__dot_or_vector__n2048.json) | 0.1951 / 0.2102 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
