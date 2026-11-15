const assert = require('node:assert/strict');
const native = require('./zig-out/lib/ownership.node');

function sample(label, session) {
  const result = { label, ...native.stats(session), rss: process.memoryUsage().rss };
  console.log(JSON.stringify(result));
  return result;
}

async function main() {
  const session = native.openSession();
  const count = Number(process.env.TENSOR_COUNT || 262144); // 1 MiB f32 tensor
  const iterations = Number(process.env.ITERATIONS || 500);
  sample('start', session);

  for (let i = 0; i < iterations; i++) {
    const tensor = native.newTensor(session, count);
    native.closeTensor(tensor);
  }
  const explicit = sample('after-explicit', session);
  assert.equal(explicit.liveTensors, 0);
  assert.equal(explicit.activeBytes, 0);

  function allocateAndDrop() {
    const tensors = [];
    for (let i = 0; i < iterations; i++) tensors.push(native.newTensor(session, count));
    sample('before-drop', session);
  }
  allocateAndDrop();
  sample('after-drop-before-gc', session);
  if (global.gc) {
    for (let i = 0; i < 3; i++) {
      global.gc();
      await new Promise(resolve => setImmediate(resolve));
    }
    const collected = sample('after-forced-gc-and-turns', session);
    assert.equal(collected.liveTensors, 0);
    assert.equal(collected.activeBytes, 0);
  }
  native.closeSession(session);
  sample('after-close', session);
}

main().catch(error => { console.error(error); process.exitCode = 1; });
