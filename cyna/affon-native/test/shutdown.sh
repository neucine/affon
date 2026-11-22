#!/bin/sh
set -eu

native=${AFFON_CYNA_NATIVE:-cyna/affon-native/build/affon-cyna-native}
printf '\032\000\000\000\026\000\000\000{"command":"shutdown"}' | "$native"
