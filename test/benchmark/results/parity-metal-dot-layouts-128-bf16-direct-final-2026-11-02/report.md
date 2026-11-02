# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `e11d4b2ce659a1282f468684dc67785fa7e6d8eadc09ccb9385101d11f61f8a0`  
Binary SHA-256: `9b093ba558b2da5073efbbb5746f369bc12fc55578b54a47186c448e4b325d70`  
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
| `metal/bf16/dot/strided/n1` | matrix | 1 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n1.json) | 0.1964 / 0.2084 | [passed](program_peer/metal__bf16__dot__strided__n1.json) | 0.1614 / 0.1962 |
| `metal/bf16/dot/strided/n2` | matrix | 2 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n2.json) | 0.1714 / 0.2076 | [passed](program_peer/metal__bf16__dot__strided__n2.json) | 0.2015 / 0.1973 |
| `metal/bf16/dot/strided/n3` | matrix | 3 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n3.json) | 0.1702 / 0.2055 | [passed](program_peer/metal__bf16__dot__strided__n3.json) | 0.1611 / 0.1941 |
| `metal/bf16/dot/strided/n4` | matrix | 4 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n4.json) | 0.1673 / 0.1856 | [passed](program_peer/metal__bf16__dot__strided__n4.json) | 0.1622 / 0.1948 |
| `metal/bf16/dot/strided/n5` | matrix | 5 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n5.json) | 0.1561 / 0.1919 | [passed](program_peer/metal__bf16__dot__strided__n5.json) | 0.1641 / 0.197 |
| `metal/bf16/dot/strided/n6` | matrix | 6 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n6.json) | 0.1733 / 0.2108 | [passed](program_peer/metal__bf16__dot__strided__n6.json) | 0.1604 / 0.1932 |
| `metal/bf16/dot/strided/n7` | matrix | 7 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n7.json) | 0.172 / 0.212 | [passed](program_peer/metal__bf16__dot__strided__n7.json) | 0.1668 / 0.1949 |
| `metal/bf16/dot/strided/n8` | matrix | 8 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n8.json) | 0.1566 / 0.1946 | [passed](program_peer/metal__bf16__dot__strided__n8.json) | 0.1992 / 0.1967 |
| `metal/bf16/dot/strided/n9` | matrix | 9 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n9.json) | 0.1726 / 0.1871 | [passed](program_peer/metal__bf16__dot__strided__n9.json) | 0.1608 / 0.1953 |
| `metal/bf16/dot/strided/n10` | matrix | 10 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n10.json) | 0.1808 / 0.2119 | [passed](program_peer/metal__bf16__dot__strided__n10.json) | 0.1546 / 0.1974 |
| `metal/bf16/dot/strided/n11` | matrix | 11 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n11.json) | 0.1825 / 0.2112 | [passed](program_peer/metal__bf16__dot__strided__n11.json) | 0.1645 / 0.1995 |
| `metal/bf16/dot/strided/n12` | matrix | 12 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n12.json) | 0.1776 / 0.1949 | [passed](program_peer/metal__bf16__dot__strided__n12.json) | 0.1658 / 0.1987 |
| `metal/bf16/dot/strided/n13` | matrix | 13 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n13.json) | 0.1675 / 0.2125 | [passed](program_peer/metal__bf16__dot__strided__n13.json) | 0.201 / 0.1979 |
| `metal/bf16/dot/strided/n14` | matrix | 14 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n14.json) | 0.1821 / 0.2104 | [passed](program_peer/metal__bf16__dot__strided__n14.json) | 0.1648 / 0.1974 |
| `metal/bf16/dot/strided/n15` | matrix | 15 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n15.json) | 0.1751 / 0.2094 | [passed](program_peer/metal__bf16__dot__strided__n15.json) | 0.1653 / 0.1945 |
| `metal/bf16/dot/strided` | matrix | 16 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided.json) | 0.177 / 0.2108 | [passed](program_peer/metal__bf16__dot__strided.json) | 0.1733 / 0.198 |
| `metal/bf16/dot/strided/n17` | matrix | 17 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n17.json) | 0.1769 / 0.2099 | [passed](program_peer/metal__bf16__dot__strided__n17.json) | 0.1734 / 0.1981 |
| `metal/bf16/dot/strided/n18` | matrix | 18 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n18.json) | 0.1854 / 0.2105 | [passed](program_peer/metal__bf16__dot__strided__n18.json) | 0.1707 / 0.1973 |
| `metal/bf16/dot/strided/n19` | matrix | 19 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n19.json) | 0.1758 / 0.2118 | [passed](program_peer/metal__bf16__dot__strided__n19.json) | 0.1837 / 0.1956 |
| `metal/bf16/dot/strided/n20` | matrix | 20 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n20.json) | 0.1731 / 0.2111 | [passed](program_peer/metal__bf16__dot__strided__n20.json) | 0.1738 / 0.1976 |
| `metal/bf16/dot/strided/n21` | matrix | 21 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n21.json) | 0.1817 / 0.2127 | [passed](program_peer/metal__bf16__dot__strided__n21.json) | 0.1724 / 0.1981 |
| `metal/bf16/dot/strided/n22` | matrix | 22 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n22.json) | 0.1714 / 0.2086 | [passed](program_peer/metal__bf16__dot__strided__n22.json) | 0.1663 / 0.1981 |
| `metal/bf16/dot/strided/n23` | matrix | 23 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n23.json) | 0.1572 / 0.1935 | [passed](program_peer/metal__bf16__dot__strided__n23.json) | 0.1662 / 0.1969 |
| `metal/bf16/dot/strided/n24` | matrix | 24 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n24.json) | 0.1674 / 0.1946 | [passed](program_peer/metal__bf16__dot__strided__n24.json) | 0.2018 / 0.1979 |
| `metal/bf16/dot/strided/n31` | matrix | 31 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n31.json) | 0.1629 / 0.1945 | [passed](program_peer/metal__bf16__dot__strided__n31.json) | 0.1672 / 0.1987 |
| `metal/bf16/dot/strided/n32` | matrix | 32 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n32.json) | 0.156 / 0.195 | [passed](program_peer/metal__bf16__dot__strided__n32.json) | 0.1682 / 0.1987 |
| `metal/bf16/dot/strided/n33` | matrix | 33 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n33.json) | 0.1634 / 0.1951 | [passed](program_peer/metal__bf16__dot__strided__n33.json) | 0.1729 / 0.198 |
| `metal/bf16/dot/strided/n63` | matrix | 63 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n63.json) | 0.2 / 0.1931 | [passed](program_peer/metal__bf16__dot__strided__n63.json) | 0.1816 / 0.1878 |
| `metal/bf16/dot/strided/n64` | matrix | 64 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n64.json) | 0.1921 / 0.1965 | [passed](program_peer/metal__bf16__dot__strided__n64.json) | 0.1714 / 0.1973 |
| `metal/bf16/dot/strided/n65` | matrix | 65 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n65.json) | 0.1847 / 0.1967 | [passed](program_peer/metal__bf16__dot__strided__n65.json) | 0.1796 / 0.1987 |
| `metal/bf16/dot/strided/n127` | matrix | 127 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n127.json) | 0.1937 / 0.1966 | [passed](program_peer/metal__bf16__dot__strided__n127.json) | 0.2099 / 0.1968 |
| `metal/bf16/dot/strided/n128` | matrix | 128 | strided | correct | [passed](operator_peer/metal__bf16__dot__strided__n128.json) | 0.1659 / 0.1938 | [passed](program_peer/metal__bf16__dot__strided__n128.json) | 0.1951 / 0.1967 |
| `metal/bf16/dot/dot_or_vector/offset_n1` | matrix | 1 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n1.json) | 0.1611 / 0.1946 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n1.json) | 0.1592 / 0.1985 |
| `metal/bf16/dot/dot_or_vector/offset_n2` | matrix | 2 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n2.json) | 0.1644 / 0.1976 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n2.json) | 0.1698 / 0.1971 |
| `metal/bf16/dot/dot_or_vector/offset_n3` | matrix | 3 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n3.json) | 0.1601 / 0.1967 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n3.json) | 0.1809 / 0.2131 |
| `metal/bf16/dot/dot_or_vector/offset_n4` | matrix | 4 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n4.json) | 0.1765 / 0.196 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n4.json) | 0.1906 / 0.2093 |
| `metal/bf16/dot/dot_or_vector/offset_n5` | matrix | 5 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n5.json) | 0.1575 / 0.1967 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n5.json) | 0.1736 / 0.2115 |
| `metal/bf16/dot/dot_or_vector/offset_n6` | matrix | 6 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n6.json) | 0.1818 / 0.1914 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n6.json) | 0.2065 / 0.2058 |
| `metal/bf16/dot/dot_or_vector/offset_n7` | matrix | 7 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n7.json) | 0.1515 / 0.1894 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n7.json) | 0.1733 / 0.2101 |
| `metal/bf16/dot/dot_or_vector/offset_n8` | matrix | 8 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n8.json) | 0.1707 / 0.1952 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n8.json) | 0.1928 / 0.2082 |
| `metal/bf16/dot/dot_or_vector/offset_n9` | matrix | 9 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n9.json) | 0.1717 / 0.2081 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n9.json) | 0.1896 / 0.2089 |
| `metal/bf16/dot/dot_or_vector/offset_n10` | matrix | 10 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n10.json) | 0.1835 / 0.2081 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n10.json) | 0.1739 / 0.2111 |
| `metal/bf16/dot/dot_or_vector/offset_n11` | matrix | 11 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n11.json) | 0.1691 / 0.2067 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n11.json) | 0.1803 / 0.2104 |
| `metal/bf16/dot/dot_or_vector/offset_n12` | matrix | 12 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n12.json) | 0.1635 / 0.1919 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n12.json) | 0.1866 / 0.2091 |
| `metal/bf16/dot/dot_or_vector/offset_n13` | matrix | 13 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n13.json) | 0.1595 / 0.1932 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n13.json) | 0.2094 / 0.2299 |
| `metal/bf16/dot/dot_or_vector/offset_n14` | matrix | 14 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n14.json) | 0.1729 / 0.1931 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n14.json) | 0.1811 / 0.2122 |
| `metal/bf16/dot/dot_or_vector/offset_n15` | matrix | 15 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n15.json) | 0.1525 / 0.1919 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n15.json) | 0.1882 / 0.2077 |
| `metal/bf16/dot/dot_or_vector/offset_n16` | matrix | 16 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n16.json) | 0.1675 / 0.193 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n16.json) | 0.1869 / 0.2094 |
| `metal/bf16/dot/dot_or_vector/offset_n17` | matrix | 17 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n17.json) | 0.1615 / 0.1944 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n17.json) | 0.1803 / 0.2138 |
| `metal/bf16/dot/dot_or_vector/offset_n18` | matrix | 18 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n18.json) | 0.1875 / 0.1954 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n18.json) | 0.1868 / 0.2122 |
| `metal/bf16/dot/dot_or_vector/offset_n19` | matrix | 19 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n19.json) | 0.1596 / 0.1931 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n19.json) | 0.1784 / 0.2134 |
| `metal/bf16/dot/dot_or_vector/offset_n20` | matrix | 20 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n20.json) | 0.163 / 0.1952 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n20.json) | 0.2126 / 0.2102 |
| `metal/bf16/dot/dot_or_vector/offset_n21` | matrix | 21 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n21.json) | 0.1649 / 0.1962 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n21.json) | 0.197 / 0.2154 |
| `metal/bf16/dot/dot_or_vector/offset_n22` | matrix | 22 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n22.json) | 0.1611 / 0.1962 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n22.json) | 0.1837 / 0.1994 |
| `metal/bf16/dot/dot_or_vector/offset_n23` | matrix | 23 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n23.json) | 0.1568 / 0.1978 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n23.json) | 0.1933 / 0.2097 |
| `metal/bf16/dot/dot_or_vector/offset_n24` | matrix | 24 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n24.json) | 0.1659 / 0.1954 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n24.json) | 0.1937 / 0.2106 |
| `metal/bf16/dot/dot_or_vector/offset_n31` | matrix | 31 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n31.json) | 0.172 / 0.1922 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n31.json) | 0.1973 / 0.1966 |
| `metal/bf16/dot/dot_or_vector/offset_n32` | matrix | 32 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n32.json) | 0.1718 / 0.1946 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n32.json) | 0.1871 / 0.2097 |
| `metal/bf16/dot/dot_or_vector/offset_n33` | matrix | 33 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n33.json) | 0.1682 / 0.1936 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n33.json) | 0.1883 / 0.2113 |
| `metal/bf16/dot/dot_or_vector/offset_n63` | matrix | 63 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n63.json) | 0.1706 / 0.195 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n63.json) | 0.1888 / 0.2144 |
| `metal/bf16/dot/dot_or_vector/offset_n64` | matrix | 64 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n64.json) | 0.2033 / 0.1959 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n64.json) | 0.2161 / 0.2111 |
| `metal/bf16/dot/dot_or_vector/offset_n65` | matrix | 65 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n65.json) | 0.1678 / 0.1976 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n65.json) | 0.2075 / 0.208 |
| `metal/bf16/dot/dot_or_vector/offset_n127` | matrix | 127 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n127.json) | 0.2043 / 0.1928 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n127.json) | 0.2134 / 0.211 |
| `metal/bf16/dot/dot_or_vector/offset_n128` | matrix | 128 | offset | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__offset_n128.json) | 0.1639 / 0.1931 | [passed](program_peer/metal__bf16__dot__dot_or_vector__offset_n128.json) | 0.1875 / 0.2104 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
