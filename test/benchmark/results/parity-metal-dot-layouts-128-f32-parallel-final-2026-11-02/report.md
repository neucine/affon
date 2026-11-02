# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `18d072058fee144f7dcfdd03ca0394ad1d7f8d31974859a99a5e4440a88d0d6c`  
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
| `metal/f32/dot/strided/n1` | matrix | 1 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n1.json) | 0.189 / 0.2038 | [passed](program_peer/metal__f32__dot__strided__n1.json) | 0.1755 / 0.2123 |
| `metal/f32/dot/strided/n2` | matrix | 2 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n2.json) | 0.1766 / 0.2105 | [passed](program_peer/metal__f32__dot__strided__n2.json) | 0.1777 / 0.2091 |
| `metal/f32/dot/strided/n3` | matrix | 3 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n3.json) | 0.1796 / 0.2155 | [passed](program_peer/metal__f32__dot__strided__n3.json) | 0.2147 / 0.2135 |
| `metal/f32/dot/strided/n4` | matrix | 4 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n4.json) | 0.1799 / 0.2122 | [passed](program_peer/metal__f32__dot__strided__n4.json) | 0.1773 / 0.2143 |
| `metal/f32/dot/strided/n5` | matrix | 5 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n5.json) | 0.1737 / 0.2101 | [passed](program_peer/metal__f32__dot__strided__n5.json) | 0.172 / 0.2113 |
| `metal/f32/dot/strided/n6` | matrix | 6 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n6.json) | 0.1797 / 0.2126 | [passed](program_peer/metal__f32__dot__strided__n6.json) | 0.1753 / 0.2087 |
| `metal/f32/dot/strided/n7` | matrix | 7 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n7.json) | 0.1857 / 0.2129 | [passed](program_peer/metal__f32__dot__strided__n7.json) | 0.1745 / 0.207 |
| `metal/f32/dot/strided/n8` | matrix | 8 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n8.json) | 0.1784 / 0.2126 | [passed](program_peer/metal__f32__dot__strided__n8.json) | 0.1895 / 0.2141 |
| `metal/f32/dot/strided/n9` | matrix | 9 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n9.json) | 0.1747 / 0.1984 | [passed](program_peer/metal__f32__dot__strided__n9.json) | 0.195 / 0.2102 |
| `metal/f32/dot/strided/n10` | matrix | 10 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n10.json) | 0.1796 / 0.2123 | [passed](program_peer/metal__f32__dot__strided__n10.json) | 0.179 / 0.2158 |
| `metal/f32/dot/strided/n11` | matrix | 11 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n11.json) | 0.1724 / 0.2117 | [passed](program_peer/metal__f32__dot__strided__n11.json) | 0.1881 / 0.2125 |
| `metal/f32/dot/strided/n12` | matrix | 12 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n12.json) | 0.171 / 0.2061 | [passed](program_peer/metal__f32__dot__strided__n12.json) | 0.1746 / 0.2139 |
| `metal/f32/dot/strided/n13` | matrix | 13 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n13.json) | 0.1731 / 0.2106 | [passed](program_peer/metal__f32__dot__strided__n13.json) | 0.1977 / 0.2123 |
| `metal/f32/dot/strided/n14` | matrix | 14 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n14.json) | 0.1786 / 0.2105 | [passed](program_peer/metal__f32__dot__strided__n14.json) | 0.1966 / 0.211 |
| `metal/f32/dot/strided/n15` | matrix | 15 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n15.json) | 0.1966 / 0.2107 | [passed](program_peer/metal__f32__dot__strided__n15.json) | 0.1731 / 0.1948 |
| `metal/f32/dot/strided` | matrix | 16 | strided | correct | [passed](operator_peer/metal__f32__dot__strided.json) | 0.196 / 0.2112 | [passed](program_peer/metal__f32__dot__strided.json) | 0.1826 / 0.2125 |
| `metal/f32/dot/strided/n17` | matrix | 17 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n17.json) | 0.1859 / 0.2092 | [passed](program_peer/metal__f32__dot__strided__n17.json) | 0.1977 / 0.2148 |
| `metal/f32/dot/strided/n18` | matrix | 18 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n18.json) | 0.1726 / 0.1926 | [passed](program_peer/metal__f32__dot__strided__n18.json) | 0.1817 / 0.2041 |
| `metal/f32/dot/strided/n19` | matrix | 19 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n19.json) | 0.1736 / 0.207 | [passed](program_peer/metal__f32__dot__strided__n19.json) | 0.1753 / 0.2092 |
| `metal/f32/dot/strided/n20` | matrix | 20 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n20.json) | 0.2046 / 0.2098 | [passed](program_peer/metal__f32__dot__strided__n20.json) | 0.1951 / 0.2105 |
| `metal/f32/dot/strided/n21` | matrix | 21 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n21.json) | 0.1927 / 0.2094 | [passed](program_peer/metal__f32__dot__strided__n21.json) | 0.1886 / 0.2222 |
| `metal/f32/dot/strided/n22` | matrix | 22 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n22.json) | 0.1768 / 0.2105 | [passed](program_peer/metal__f32__dot__strided__n22.json) | 0.1865 / 0.2121 |
| `metal/f32/dot/strided/n23` | matrix | 23 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n23.json) | 0.1761 / 0.2091 | [passed](program_peer/metal__f32__dot__strided__n23.json) | 0.1746 / 0.2119 |
| `metal/f32/dot/strided/n24` | matrix | 24 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n24.json) | 0.1803 / 0.2102 | [passed](program_peer/metal__f32__dot__strided__n24.json) | 0.198 / 0.1865 |
| `metal/f32/dot/strided/n31` | matrix | 31 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n31.json) | 0.181 / 0.203 | [passed](program_peer/metal__f32__dot__strided__n31.json) | 0.1874 / 0.2102 |
| `metal/f32/dot/strided/n32` | matrix | 32 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n32.json) | 0.1761 / 0.2049 | [passed](program_peer/metal__f32__dot__strided__n32.json) | 0.1997 / 0.2134 |
| `metal/f32/dot/strided/n33` | matrix | 33 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n33.json) | 0.1846 / 0.2087 | [passed](program_peer/metal__f32__dot__strided__n33.json) | 0.1845 / 0.21 |
| `metal/f32/dot/strided/n63` | matrix | 63 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n63.json) | 0.2001 / 0.2107 | [passed](program_peer/metal__f32__dot__strided__n63.json) | 0.2098 / 0.1993 |
| `metal/f32/dot/strided/n64` | matrix | 64 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n64.json) | 0.2263 / 0.2076 | [passed](program_peer/metal__f32__dot__strided__n64.json) | 0.1919 / 0.1961 |
| `metal/f32/dot/strided/n65` | matrix | 65 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n65.json) | 0.1995 / 0.208 | [passed](program_peer/metal__f32__dot__strided__n65.json) | 0.1908 / 0.212 |
| `metal/f32/dot/strided/n127` | matrix | 127 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n127.json) | 0.2243 / 0.2035 | [passed](program_peer/metal__f32__dot__strided__n127.json) | 0.2363 / 0.2137 |
| `metal/f32/dot/strided/n128` | matrix | 128 | strided | correct | [passed](operator_peer/metal__f32__dot__strided__n128.json) | 0.1721 / 0.2076 | [passed](program_peer/metal__f32__dot__strided__n128.json) | 0.1779 / 0.2139 |
| `metal/f32/dot/dot_or_vector/offset_n1` | matrix | 1 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n1.json) | 0.1763 / 0.2057 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n1.json) | 0.1721 / 0.1987 |
| `metal/f32/dot/dot_or_vector/offset_n2` | matrix | 2 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n2.json) | 0.1743 / 0.209 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n2.json) | 0.1727 / 0.212 |
| `metal/f32/dot/dot_or_vector/offset_n3` | matrix | 3 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n3.json) | 0.1697 / 0.2073 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n3.json) | 0.2063 / 0.2141 |
| `metal/f32/dot/dot_or_vector/offset_n4` | matrix | 4 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n4.json) | 0.1763 / 0.2115 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n4.json) | 0.1824 / 0.2101 |
| `metal/f32/dot/dot_or_vector/offset_n5` | matrix | 5 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n5.json) | 0.1806 / 0.2072 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n5.json) | 0.1801 / 0.2123 |
| `metal/f32/dot/dot_or_vector/offset_n6` | matrix | 6 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n6.json) | 0.1806 / 0.2117 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n6.json) | 0.1759 / 0.2047 |
| `metal/f32/dot/dot_or_vector/offset_n7` | matrix | 7 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n7.json) | 0.1741 / 0.2077 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n7.json) | 0.1744 / 0.2095 |
| `metal/f32/dot/dot_or_vector/offset_n8` | matrix | 8 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n8.json) | 0.1768 / 0.2089 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n8.json) | 0.1834 / 0.2107 |
| `metal/f32/dot/dot_or_vector/offset_n9` | matrix | 9 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n9.json) | 0.1809 / 0.2095 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n9.json) | 0.192 / 0.2105 |
| `metal/f32/dot/dot_or_vector/offset_n10` | matrix | 10 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n10.json) | 0.1827 / 0.2096 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n10.json) | 0.1761 / 0.2103 |
| `metal/f32/dot/dot_or_vector/offset_n11` | matrix | 11 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n11.json) | 0.1853 / 0.2099 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n11.json) | 0.1739 / 0.2116 |
| `metal/f32/dot/dot_or_vector/offset_n12` | matrix | 12 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n12.json) | 0.186 / 0.2085 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n12.json) | 0.1926 / 0.2078 |
| `metal/f32/dot/dot_or_vector/offset_n13` | matrix | 13 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n13.json) | 0.1769 / 0.2093 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n13.json) | 0.1822 / 0.2165 |
| `metal/f32/dot/dot_or_vector/offset_n14` | matrix | 14 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n14.json) | 0.1744 / 0.2081 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n14.json) | 0.1741 / 0.2113 |
| `metal/f32/dot/dot_or_vector/offset_n15` | matrix | 15 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n15.json) | 0.2122 / 0.2115 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n15.json) | 0.1837 / 0.2094 |
| `metal/f32/dot/dot_or_vector/offset_n16` | matrix | 16 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n16.json) | 0.1837 / 0.21 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n16.json) | 0.1844 / 0.2074 |
| `metal/f32/dot/dot_or_vector/offset_n17` | matrix | 17 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n17.json) | 0.164 / 0.2075 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n17.json) | 0.1846 / 0.2097 |
| `metal/f32/dot/dot_or_vector/offset_n18` | matrix | 18 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n18.json) | 0.1838 / 0.2091 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n18.json) | 0.1774 / 0.2098 |
| `metal/f32/dot/dot_or_vector/offset_n19` | matrix | 19 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n19.json) | 0.1746 / 0.2078 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n19.json) | 0.1833 / 0.2092 |
| `metal/f32/dot/dot_or_vector/offset_n20` | matrix | 20 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n20.json) | 0.1746 / 0.2024 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n20.json) | 0.1852 / 0.2106 |
| `metal/f32/dot/dot_or_vector/offset_n21` | matrix | 21 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n21.json) | 0.1816 / 0.2069 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n21.json) | 0.1904 / 0.2113 |
| `metal/f32/dot/dot_or_vector/offset_n22` | matrix | 22 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n22.json) | 0.1829 / 0.2097 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n22.json) | 0.2112 / 0.2087 |
| `metal/f32/dot/dot_or_vector/offset_n23` | matrix | 23 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n23.json) | 0.1806 / 0.2074 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n23.json) | 0.1801 / 0.2122 |
| `metal/f32/dot/dot_or_vector/offset_n24` | matrix | 24 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n24.json) | 0.1904 / 0.2103 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n24.json) | 0.1867 / 0.2138 |
| `metal/f32/dot/dot_or_vector/offset_n31` | matrix | 31 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n31.json) | 0.1936 / 0.2117 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n31.json) | 0.1839 / 0.2125 |
| `metal/f32/dot/dot_or_vector/offset_n32` | matrix | 32 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n32.json) | 0.1924 / 0.2078 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n32.json) | 0.1921 / 0.2095 |
| `metal/f32/dot/dot_or_vector/offset_n33` | matrix | 33 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n33.json) | 0.1833 / 0.2093 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n33.json) | 0.2145 / 0.2126 |
| `metal/f32/dot/dot_or_vector/offset_n63` | matrix | 63 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n63.json) | 0.2178 / 0.2071 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n63.json) | 0.2113 / 0.2113 |
| `metal/f32/dot/dot_or_vector/offset_n64` | matrix | 64 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n64.json) | 0.1845 / 0.2086 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n64.json) | 0.1978 / 0.2116 |
| `metal/f32/dot/dot_or_vector/offset_n65` | matrix | 65 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n65.json) | 0.1993 / 0.1991 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n65.json) | 0.1961 / 0.215 |
| `metal/f32/dot/dot_or_vector/offset_n127` | matrix | 127 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n127.json) | 0.2214 / 0.2088 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n127.json) | 0.2288 / 0.2109 |
| `metal/f32/dot/dot_or_vector/offset_n128` | matrix | 128 | offset | correct | [passed](operator_peer/metal__f32__dot__dot_or_vector__offset_n128.json) | 0.1899 / 0.2105 | [passed](program_peer/metal__f32__dot__dot_or_vector__offset_n128.json) | 0.2042 / 0.2108 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
