# Program Inspector

This zero-build browser app is the first interactive consumer of
`@affon/inspector`. It opens serialized `ProgramInspection` JSON and keeps its
structure tree, collapsed dependency graph, and exact details panel synchronized.

From the repository root, serve the checkout over HTTP and open the route:

```sh
python3 -m http.server 8080
```

Then visit `http://localhost:8080/apps/program-inspector/`. The app has a small
arithmetic inspection built in; use **Open inspection JSON** or drag a JSON file
onto the page to inspect an exported Program. No Program execution, Session, or
browser framework is required.

The current viewer deliberately hides captured constant values. It shows their
presence and metadata, while the serialized inspection remains the source of
truth.

Generate the two-layer decoder LM inspection used as a complex manual fixture:

```sh
RUNTIME_PACKAGE_PATH="$PWD/packages" ./zig-out/bin/affon run \
  apps/program-inspector/generate-decoder-inspection.ts
```

The command writes `/private/tmp/decoder-lm.affon-inspection.json` with constants
summarized for safe local import.

The viewer labels composition scopes as structural groupings derived from node
paths. For schema version 1 it also shows the exact child-Program bindings and
outputs captured automatically during composition. Legacy unversioned JSON
still opens with path-derived scopes, but cannot show those missing interfaces.
No authoring annotations or visualization metadata are required.
