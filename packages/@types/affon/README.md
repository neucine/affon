# Affon runtime declarations

The root tsconfig loads this package through typeRoots and types: ["affon"].
index.d.ts references both Affon modules and Hao standard modules explicitly.
No wildcard std:* declaration is used.

The std:ffi, std:ffi/c, std:http, std:plot, std:runtime and std:util declarations
are copied from Hao's types/std directory at revision
a5b07cea0f196329825536d481246bcdd72f997e.
Keep these declarations synchronized with the Hao runtime dependency when
upgrading it. The copies keep this package usable without sibling-checkout paths.
