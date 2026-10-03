declare module "affon:optim" {
  export type SGD = Readonly<{
    kind: "sgd";
    learning_rate: number;
    momentum: number;
  }>;
  export type Adam = Readonly<{ kind: "adam"; learning_rate: number; beta1: number; beta2: number; epsilon: number }>;
  export type AdamW = Readonly<{ kind: "adamw"; learning_rate: number; beta1: number; beta2: number; epsilon: number; weight_decay: number }>;
  export type Optimizer = SGD | Adam | AdamW;

  export function sgd(options?: { learning_rate?: number; momentum?: number }): SGD;
  export function adam(options?: { learning_rate?: number; beta1?: number; beta2?: number; epsilon?: number }): Adam;
  export function adamw(options?: { learning_rate?: number; beta1?: number; beta2?: number; epsilon?: number; weight_decay?: number }): AdamW;
}
