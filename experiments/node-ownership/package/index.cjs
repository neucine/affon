'use strict';

const native = require('../zig-out/lib/ownership.node');
const { AsyncLocalStorage } = require('node:async_hooks');
const activeScope = new AsyncLocalStorage();

class Tensor {
  #handle;
  #disposed = false;
  #session;

  constructor(session, handle) {
    this.#session = session;
    this.#handle = handle;
  }

  get disposed() { return this.#disposed; }

  dispose() {
    if (this.#disposed) return;
    native.closeTensor(this.#handle);
    this.#disposed = true;
    this.#session = null;
  }

  [Symbol.dispose]() { this.dispose(); }
}

class Scope {
  #session;
  #tensors = new Set();
  #disposed = false;

  constructor(session) { this.#session = session; }

  get disposed() { return this.#disposed; }

  own(tensor) {
    if (this.#disposed) throw new Error('Scope has been disposed');
    this.#tensors.add(tensor);
  }

  retainReturned(value, seen = new WeakSet()) {
    if (value instanceof Tensor) {
      this.#tensors.delete(value);
    } else if (value && typeof value === 'object' && !seen.has(value)) {
      seen.add(value);
      for (const child of Object.values(value)) this.retainReturned(child, seen);
    }
  }

  zeros(elements) {
    if (this.#disposed) throw new Error('Scope has been disposed');
    const tensor = this.#session.zeros(elements);
    this.own(tensor);
    return tensor;
  }

  transfer(tensor) {
    if (this.#disposed) throw new Error('Scope has been disposed');
    if (!this.#tensors.delete(tensor)) throw new Error('Tensor is not owned by this scope');
    return tensor;
  }

  dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const tensor of Array.from(this.#tensors).reverse()) tensor.dispose();
    this.#tensors.clear();
  }

  [Symbol.dispose]() { this.dispose(); }
}

function runScoped(resource, callback) {
  try {
    const result = callback(resource);
    if (result && typeof result.then === 'function') return Promise.resolve(result).finally(() => resource.dispose());
    resource.dispose();
    return result;
  } catch (error) {
    resource.dispose();
    throw error;
  }
}

class Session {
  #handle = native.openSession();
  #disposed = false;

  get disposed() { return this.#disposed; }

  zeros(elements) {
    if (this.#disposed) throw new Error('Session has been disposed');
    if (!Number.isSafeInteger(elements) || elements <= 0 || elements > 16 * 1024 * 1024)
      throw new RangeError('elements must be an integer between 1 and 16777216');
    const active = activeScope.getStore();
    if (active && active.session === this && active.scope.disposed)
      throw new Error('Run scope has been disposed');
    const tensor = new Tensor(this, native.newTensor(this.#handle, elements));
    if (active && active.session === this) active.scope.own(tensor);
    return tensor;
  }

  run(callback) {
    if (this.#disposed) throw new Error('Session has been disposed');
    const scope = new Scope(this);
    return activeScope.run({ session: this, scope }, () => {
      try {
        const result = callback();
        if (result && typeof result.then === 'function') {
          return Promise.resolve(result).then(
            value => { scope.retainReturned(value); return value; },
          ).finally(() => scope.dispose());
        }
        scope.retainReturned(result);
        scope.dispose();
        return result;
      } catch (error) {
        scope.dispose();
        throw error;
      }
    });
  }

  scope(callback) {
    if (this.#disposed) throw new Error('Session has been disposed');
    return runScoped(new Scope(this), callback);
  }

  stats() { return native.stats(this.#handle); }

  dispose() {
    if (this.#disposed) return;
    native.closeSession(this.#handle);
    this.#disposed = true;
  }

  [Symbol.dispose]() { this.dispose(); }
}

function withSession(callback) { return runScoped(new Session(), callback); }

module.exports = { Session, withSession };
