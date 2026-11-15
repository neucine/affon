export declare class Tensor {
  readonly disposed: boolean;
  dispose(): void;
  [Symbol.dispose](): void;
}

export declare class Scope {
  readonly disposed: boolean;
  /** Allocate a one dimensional f32 CPU tensor owned by this scope. */
  zeros(elements: number): Tensor;
  /** Remove a tensor from automatic scope disposal. The caller then owns it. */
  transfer(tensor: Tensor): Tensor;
  dispose(): void;
  [Symbol.dispose](): void;
}

export declare class Session {
  readonly disposed: boolean;
  zeros(elements: number): Tensor;
  /** Release tensors created in the callback, except tensors returned from it. */
  run<T>(callback: () => T): T;
  scope<T>(callback: (scope: Scope) => T): T;
  stats(): { activeBytes: number; liveTensors: number };
  dispose(): void;
  [Symbol.dispose](): void;
}

export declare function withSession<T>(callback: (session: Session) => T): T;
