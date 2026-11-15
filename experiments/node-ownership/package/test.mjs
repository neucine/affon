import assert from 'node:assert/strict';
import { Session, withSession } from '@affon/core';

const session = new Session();
for (let i = 0; i < 500; i++) {
  session.run(() => {
    session.zeros(262144);
  });
}
assert.deepEqual(session.stats(), { activeBytes: 0, liveTensors: 0 });

let escaped;
assert.throws(() => session.run(() => {
  escaped = session.zeros(262144);
  throw new Error('early exit');
}), /early exit/);
assert.equal(escaped.disposed, true);
assert.deepEqual(session.stats(), { activeBytes: 0, liveTensors: 0 });

await session.run(async () => {
  session.zeros(262144);
  await Promise.resolve();
  assert.equal(session.stats().liveTensors, 1);
});
assert.deepEqual(session.stats(), { activeBytes: 0, liveTensors: 0 });

let releaseFirst;
const first = session.run(async () => {
  session.zeros(262144);
  await new Promise(resolve => { releaseFirst = resolve; });
});
await session.run(async () => {
  session.zeros(262144);
  await Promise.resolve();
});
assert.equal(session.stats().liveTensors, 1);
releaseFirst();
await first;
assert.deepEqual(session.stats(), { activeBytes: 0, liveTensors: 0 });

const kept = session.run(() => session.zeros(262144));
assert.equal(session.stats().liveTensors, 1);
kept.dispose();
kept.dispose();
assert.deepEqual(session.stats(), { activeBytes: 0, liveTensors: 0 });

const result = session.run(() => ({ output: session.zeros(262144) }));
assert.equal(result.output.disposed, false);
result.output.dispose();
assert.deepEqual(session.stats(), { activeBytes: 0, liveTensors: 0 });

session.dispose();
session.dispose();
assert.throws(() => session.zeros(1), /disposed/);

const closed = withSession(owner => {
  owner.run(() => owner.zeros(262144));
  return owner;
});
assert.equal(closed.disposed, true);
console.log('scoped ownership passed');
