if (typeof globalThis.AffonError !== 'function') {
  globalThis.AffonError = class AffonError extends Error {
    constructor(code, message) {
      super(message)
      this.name = 'AffonError'
      this.code = code
    }
  }
}
