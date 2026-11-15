# First Program in one minute

Install Affon on a [supported platform](../runtime/install.md), then get the
example and run it from the repository root:

```sh
git clone https://github.com/neucine/affon.git
cd affon
```

```sh
affon examples/first-program.ts
```

The script uses three local numbers and the CPU backend, so it requires no model
weights, Python environment, network request, or GPU. It declares the input
shape and dtype, inspects the authored training Program, compiles it for a
Session, and trains one parameter with SGD. A successful run prints the input
contract, graph size, an optimizer transition, a decreasing loss, and three
predictions close to `[2, 4, 6]`. It raises an error if the loss has not fallen
by at least 100 times.

The repository supplies the [example file](../../examples/first-program.ts);
the installed runtime supplies `affon:core`, `affon:ops`, and `affon:optim`.
To inspect the full semantic graph, print
`JSON.stringify(training.inspect(), null, 2)` in the
script. The inspection describes the Program; it does not claim runtime timing
or accelerator placement. See [backend support](../core/backend-support.md)
for how to check a specific GPU workload.
