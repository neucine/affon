/**
 * Immutable optimizer descriptors consumed by affon:compute optimize().
 * Creating a descriptor does not allocate state or mutate parameters.
 *
 * @summary Configure SGD, Adam, or AdamW training transforms.
 */
declare module "affon:optim" {
  /** Stochastic gradient descent configuration. */
  export type SGD = Readonly<{
    /** Descriptor discriminator. */
    kind: "sgd";
    /** Step size applied to each parameter update. */
    learning_rate: number;
    /** Momentum coefficient. Zero selects plain SGD. */
    momentum: number;
  }>;
  /** Adam configuration. */
  export type Adam = Readonly<{ kind: "adam"; learning_rate: number; beta1: number; beta2: number; epsilon: number }>;
  /** AdamW configuration with decoupled weight decay. */
  export type AdamW = Readonly<{ kind: "adamw"; learning_rate: number; beta1: number; beta2: number; epsilon: number; weight_decay: number }>;
  /** An optimizer that updates parameters on each call. */
  export type BaseOptimizer = SGD | Adam | AdamW;
  /** An immutable, step-based learning-rate schedule. */
  export type LRSchedule =
    | Readonly<{ kind: "constant"; learning_rate: number }>
    | Readonly<{ kind: "linear" | "cosine"; start: number; end: number; steps: number }>
    | Readonly<{ kind: "step"; base: number; gamma: number; every: number; steps?: number }>
    | Readonly<{ kind: "warmup_cosine"; start: number; peak: number; end: number; warmup_steps: number; total_steps: number }>
    | Readonly<{ kind: "sequence"; schedules: readonly LRSchedule[] }>;
  /** A base optimizer whose learning rate is selected from a schedule at each update. */
  export type ScheduledOptimizer = Readonly<{ kind: "scheduled"; optimizer: BaseOptimizer; schedule: LRSchedule }>;
  /** An optimizer that produces an update whenever it receives gradients. */
  export type UpdatingOptimizer = BaseOptimizer | ScheduledOptimizer;
  /** Gradient-accumulation wrapper around an updating optimizer. */
  export type AccumulatingOptimizer = Readonly<{ kind: "accumulate"; optimizer: UpdatingOptimizer; steps: number }>;
  /** Any optimizer descriptor accepted by optimize(). */
  export type Optimizer = UpdatingOptimizer | AccumulatingOptimizer;

  /** Step-based schedule factories. */
  export const schedules: Readonly<{
    constant(learning_rate: number): LRSchedule;
    linear(options: { start: number; end: number; steps: number }): LRSchedule;
    cosine(options: { start: number; end: number; steps: number }): LRSchedule;
    step(options: { base: number; gamma: number; every: number; steps?: number }): LRSchedule;
    warmupCosine(options: { start: number; peak: number; end: number; warmup_steps: number; total_steps: number }): LRSchedule;
    sequence(...parts: readonly LRSchedule[]): LRSchedule;
  }>;

  /**
   * Create an immutable SGD descriptor.
   * @semantics Defaults to learning_rate 0.01 and momentum 0.
   */
  export function sgd(options?: { learning_rate?: number; momentum?: number }): SGD;
  /**
   * Create an immutable Adam descriptor.
   * @semantics Defaults to learning_rate 0.001, beta1 0.9, beta2 0.999, and epsilon 1e-8.
   */
  export function adam(options?: { learning_rate?: number; beta1?: number; beta2?: number; epsilon?: number }): Adam;
  /**
   * Create an immutable AdamW descriptor with decoupled weight decay.
   * @semantics Uses Adam defaults and a default weight_decay of 0.01.
   */
  export function adamw(options?: { learning_rate?: number; beta1?: number; beta2?: number; epsilon?: number; weight_decay?: number }): AdamW;
  /** Select the base optimizer's learning rate from a schedule on each update. */
  export function scheduled(optimizer: BaseOptimizer, schedule: LRSchedule): ScheduledOptimizer;
  /**
   * Accumulate and average gradients over multiple runs before updating parameters.
   * @semantics steps must be an integer of at least 2.
   */
  export function accumulate(optimizer: UpdatingOptimizer, options: { steps: number }): AccumulatingOptimizer;
}
