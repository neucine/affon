# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `9204bbf2a64a9ac51ffedba2198e76caa798379395d0ce30600f37d323df8e60`  
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
| `metal/bf16/dot/dot_or_vector/n1` | matrix | 1 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n1.json) | 0.18 / 0.2032 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n1.json) | 0.1515 / 0.1817 |
| `metal/bf16/dot/dot_or_vector/n2` | matrix | 2 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n2.json) | 0.1717 / 0.2059 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n2.json) | 0.1598 / 0.1873 |
| `metal/bf16/dot/dot_or_vector/n3` | matrix | 3 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n3.json) | 0.1675 / 0.206 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n3.json) | 0.1784 / 0.1871 |
| `metal/bf16/dot/dot_or_vector/n4` | matrix | 4 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n4.json) | 0.1782 / 0.2065 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n4.json) | 0.1608 / 0.1883 |
| `metal/bf16/dot/dot_or_vector/n5` | matrix | 5 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n5.json) | 0.1832 / 0.2044 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n5.json) | 0.1641 / 0.1867 |
| `metal/bf16/dot/dot_or_vector/n6` | matrix | 6 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n6.json) | 0.1725 / 0.2016 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n6.json) | 0.1698 / 0.1925 |
| `metal/bf16/dot/dot_or_vector/n7` | matrix | 7 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n7.json) | 0.1712 / 0.2051 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n7.json) | 0.1702 / 0.1908 |
| `metal/bf16/dot/dot_or_vector/n8` | matrix | 8 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n8.json) | 0.1796 / 0.2068 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n8.json) | 0.1616 / 0.1934 |
| `metal/bf16/dot/dot_or_vector/n9` | matrix | 9 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n9.json) | 0.1728 / 0.2036 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n9.json) | 0.1848 / 0.1888 |
| `metal/bf16/dot/dot_or_vector/n10` | matrix | 10 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n10.json) | 0.1721 / 0.2014 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n10.json) | 0.1602 / 0.1881 |
| `metal/bf16/dot/dot_or_vector/n11` | matrix | 11 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n11.json) | 0.1803 / 0.2028 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n11.json) | 0.1681 / 0.1919 |
| `metal/bf16/dot/dot_or_vector/n12` | matrix | 12 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n12.json) | 0.1698 / 0.2051 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n12.json) | 0.1881 / 0.1917 |
| `metal/bf16/dot/dot_or_vector/n13` | matrix | 13 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n13.json) | 0.194 / 0.2071 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n13.json) | 0.1676 / 0.1918 |
| `metal/bf16/dot/dot_or_vector/n14` | matrix | 14 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n14.json) | 0.1766 / 0.2036 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n14.json) | 0.177 / 0.1928 |
| `metal/bf16/dot/dot_or_vector/n15` | matrix | 15 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n15.json) | 0.1816 / 0.207 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n15.json) | 0.1612 / 0.191 |
| `metal/bf16/dot/dot_or_vector` | matrix | 16 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector.json) | 0.1783 / 0.2068 | [passed](program_peer/metal__bf16__dot__dot_or_vector.json) | 0.1646 / 0.1906 |
| `metal/bf16/dot/dot_or_vector/n17` | matrix | 17 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n17.json) | 0.1724 / 0.204 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n17.json) | 0.1656 / 0.1907 |
| `metal/bf16/dot/dot_or_vector/n18` | matrix | 18 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n18.json) | 0.1775 / 0.2052 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n18.json) | 0.1645 / 0.1924 |
| `metal/bf16/dot/dot_or_vector/n19` | matrix | 19 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n19.json) | 0.182 / 0.2043 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n19.json) | 0.161 / 0.1894 |
| `metal/bf16/dot/dot_or_vector/n20` | matrix | 20 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n20.json) | 0.1748 / 0.2042 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n20.json) | 0.1618 / 0.1919 |
| `metal/bf16/dot/dot_or_vector/n21` | matrix | 21 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n21.json) | 0.1829 / 0.2033 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n21.json) | 0.1743 / 0.1912 |
| `metal/bf16/dot/dot_or_vector/n22` | matrix | 22 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n22.json) | 0.1802 / 0.2045 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n22.json) | 0.1677 / 0.1922 |
| `metal/bf16/dot/dot_or_vector/n23` | matrix | 23 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n23.json) | 0.1798 / 0.202 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n23.json) | 0.1616 / 0.1929 |
| `metal/bf16/dot/dot_or_vector/n24` | matrix | 24 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n24.json) | 0.1738 / 0.2067 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n24.json) | 0.1667 / 0.1945 |
| `metal/bf16/dot/dot_or_vector/n25` | matrix | 25 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n25.json) | 0.1825 / 0.2071 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n25.json) | 0.1746 / 0.1956 |
| `metal/bf16/dot/dot_or_vector/n26` | matrix | 26 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n26.json) | 0.1747 / 0.1879 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n26.json) | 0.1848 / 0.1935 |
| `metal/bf16/dot/dot_or_vector/n27` | matrix | 27 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n27.json) | 0.1778 / 0.2061 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n27.json) | 0.1672 / 0.1941 |
| `metal/bf16/dot/dot_or_vector/n28` | matrix | 28 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n28.json) | 0.1788 / 0.2068 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n28.json) | 0.1651 / 0.1914 |
| `metal/bf16/dot/dot_or_vector/n29` | matrix | 29 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n29.json) | 0.1857 / 0.1802 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n29.json) | 0.1634 / 0.1896 |
| `metal/bf16/dot/dot_or_vector/n30` | matrix | 30 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n30.json) | 0.1749 / 0.2044 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n30.json) | 0.1641 / 0.1777 |
| `metal/bf16/dot/dot_or_vector/n31` | matrix | 31 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n31.json) | 0.1784 / 0.2068 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n31.json) | 0.1776 / 0.1894 |
| `metal/bf16/dot/dot_or_vector/n32` | matrix | 32 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n32.json) | 0.1997 / 0.2054 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n32.json) | 0.1807 / 0.1907 |
| `metal/bf16/dot/dot_or_vector/n33` | matrix | 33 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n33.json) | 0.1933 / 0.2055 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n33.json) | 0.175 / 0.1909 |
| `metal/bf16/dot/dot_or_vector/n34` | matrix | 34 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n34.json) | 0.1791 / 0.2047 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n34.json) | 0.2 / 0.1931 |
| `metal/bf16/dot/dot_or_vector/n35` | matrix | 35 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n35.json) | 0.1867 / 0.2027 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n35.json) | 0.1674 / 0.1906 |
| `metal/bf16/dot/dot_or_vector/n36` | matrix | 36 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n36.json) | 0.1838 / 0.2018 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n36.json) | 0.185 / 0.1895 |
| `metal/bf16/dot/dot_or_vector/n37` | matrix | 37 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n37.json) | 0.1886 / 0.2031 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n37.json) | 0.1756 / 0.1949 |
| `metal/bf16/dot/dot_or_vector/n38` | matrix | 38 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n38.json) | 0.1753 / 0.2053 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n38.json) | 0.1993 / 0.1932 |
| `metal/bf16/dot/dot_or_vector/n39` | matrix | 39 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n39.json) | 0.2162 / 0.2037 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n39.json) | 0.2101 / 0.1909 |
| `metal/bf16/dot/dot_or_vector/n40` | matrix | 40 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n40.json) | 0.181 / 0.2051 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n40.json) | 0.1665 / 0.1921 |
| `metal/bf16/dot/dot_or_vector/n41` | matrix | 41 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n41.json) | 0.1824 / 0.1979 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n41.json) | 0.1887 / 0.1914 |
| `metal/bf16/dot/dot_or_vector/n42` | matrix | 42 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n42.json) | 0.1924 / 0.206 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n42.json) | 0.1746 / 0.1925 |
| `metal/bf16/dot/dot_or_vector/n43` | matrix | 43 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n43.json) | 0.1785 / 0.2052 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n43.json) | 0.1851 / 0.1929 |
| `metal/bf16/dot/dot_or_vector/n44` | matrix | 44 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n44.json) | 0.1861 / 0.2046 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n44.json) | 0.1661 / 0.1928 |
| `metal/bf16/dot/dot_or_vector/n45` | matrix | 45 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n45.json) | 0.181 / 0.2017 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n45.json) | 0.206 / 0.1929 |
| `metal/bf16/dot/dot_or_vector/n46` | matrix | 46 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n46.json) | 0.1944 / 0.2048 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n46.json) | 0.1706 / 0.1903 |
| `metal/bf16/dot/dot_or_vector/n47` | matrix | 47 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n47.json) | 0.2152 / 0.2065 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n47.json) | 0.1861 / 0.1927 |
| `metal/bf16/dot/dot_or_vector/n48` | matrix | 48 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n48.json) | 0.1906 / 0.2031 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n48.json) | 0.1842 / 0.1907 |
| `metal/bf16/dot/dot_or_vector/n63` | matrix | 63 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n63.json) | 0.1906 / 0.1968 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n63.json) | 0.1777 / 0.1923 |
| `metal/bf16/dot/dot_or_vector/n64` | matrix | 64 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n64.json) | 0.1886 / 0.2062 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n64.json) | 0.1875 / 0.192 |
| `metal/bf16/dot/dot_or_vector/n65` | matrix | 65 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n65.json) | 0.187 / 0.2014 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n65.json) | 0.1951 / 0.1932 |
| `metal/bf16/dot/dot_or_vector/n127` | matrix | 127 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n127.json) | 0.2097 / 0.1971 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n127.json) | 0.2037 / 0.1948 |
| `metal/bf16/dot/dot_or_vector/n128` | matrix | 128 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n128.json) | 0.1893 / 0.2062 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n128.json) | 0.1717 / 0.1921 |
| `metal/bf16/dot/dot_or_vector/n129` | matrix | 129 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n129.json) | 0.1794 / 0.2038 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n129.json) | 0.1797 / 0.1937 |
| `metal/bf16/dot/dot_or_vector/n255` | matrix | 255 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n255.json) | 0.1712 / 0.1981 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n255.json) | 0.1703 / 0.1927 |
| `metal/bf16/dot/dot_or_vector/n256` | matrix | 256 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n256.json) | 0.1922 / 0.2043 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n256.json) | 0.1789 / 0.1945 |
| `metal/bf16/dot/dot_or_vector/n257` | matrix | 257 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n257.json) | 0.1875 / 0.2038 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n257.json) | 0.1761 / 0.1924 |
| `metal/bf16/dot/dot_or_vector/n511` | matrix | 511 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n511.json) | 0.2084 / 0.1951 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n511.json) | 0.2123 / 0.193 |
| `metal/bf16/dot/dot_or_vector/n512` | matrix | 512 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n512.json) | 0.1626 / 0.1702 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n512.json) | 0.168 / 0.1925 |
| `metal/bf16/dot/dot_or_vector/n513` | matrix | 513 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n513.json) | 0.1532 / 0.1832 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n513.json) | 0.1712 / 0.191 |
| `metal/bf16/dot/dot_or_vector/n1023` | matrix | 1023 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n1023.json) | 0.1724 / 0.1874 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n1023.json) | 0.1742 / 0.1933 |
| `metal/bf16/dot/dot_or_vector/n1024` | matrix | 1024 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n1024.json) | 0.1716 / 0.1887 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n1024.json) | 0.1952 / 0.1926 |
| `metal/bf16/dot/dot_or_vector/n1025` | matrix | 1025 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n1025.json) | 0.1761 / 0.1857 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n1025.json) | 0.1689 / 0.192 |
| `metal/bf16/dot/dot_or_vector/n2048` | matrix | 2048 | dense | correct | [passed](operator_peer/metal__bf16__dot__dot_or_vector__n2048.json) | 0.1707 / 0.1866 | [passed](program_peer/metal__bf16__dot__dot_or_vector__n2048.json) | 0.1781 / 0.1902 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
