import { add, cat, mul, reshape } from 'affon:ops'
import {
  Tensor,
  Session,
  gradient,
  optimize,
  program,
  type Executable,
  type FormalTensor,
  type Program,
  type ProgramArguments,
  type TensorSpec,
} from "affon:compute"
import { adam, sgd, type Adam, type Optimizer, type SGD } from "affon:optim"
import * as compute from "affon:compute"
import * as optimModule from "affon:optim"

// @ts-expect-error Compute implementation modules are not part of the public type graph.
import * as internalCompute from "affon:_internal/compute/program"
void internalCompute

// @ts-expect-error Losses are authored through a Program builder's p.nn namespace.
import { cross_entropy as computeCrossEntropy } from "affon:compute"
// @ts-expect-error Optimizers belong to affon:optim.
import { adam as computeAdam } from "affon:compute"
void computeCrossEntropy
void computeAdam

// @ts-expect-error Formal tensors are created only by a ProgramBuilder callback.
compute.FormalTensor
// @ts-expect-error Program builders are provided only by program().
compute.ProgramBuilder
// @ts-expect-error Executables are created only by Session.compile().
compute.Executable
// @ts-expect-error Execution state is created only by Session.initialize().
compute.ExecutionState

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
function assertType<T extends true>() {}

assertType<IsExact<"default" extends keyof typeof compute ? true : false, false>>()
assertType<IsExact<"default" extends keyof typeof optimModule ? true : false, false>>()

const image = Tensor.f32([32, 784], { axes: ["batch", "feature"] })
// @ts-expect-error bool is not executable by the compute core.
Tensor.bool([2])
const classifier = program("classifier", p => {
  const value = p.argument("image", image)
  return p.nn.linear(value, { name: "projection", out_features: 10 })
})
const loss = program("classifier_loss", p => {
  const logits = classifier(p.argument("image", image))
  return p.nn.cross_entropy(logits, p.argument("labels", Tensor.i64([32])))
})
const derivatives = gradient(loss, ["classifier_projection_weight", "classifier_projection_bias"])
const optimizer = adam({ learning_rate: 0.001 })
const momentumOptimizer = sgd({ momentum: 0.9 })
const training_step = optimize(loss, optimizer)
const session = new Session({ device: "cpu" })
const runtimeTensor: Tensor = session.tensor([1, 2, 3])
const executable = session.compile(training_step)
const executionState = session.initialize(training_step)
const inferenceExecutable = session.compile(classifier)
const inferenceOutput = inferenceExecutable.run({ image: runtimeTensor })
const paired = program("paired", p => {
  const value = p.argument("value", Tensor.f32([3]))
  return [value, value] as const
})
const pairedOutput = session.compile(paired).run({ value: runtimeTensor })
const eagerSum = add(runtimeTensor, runtimeTensor)

assertType<IsExact<typeof image, TensorSpec>>()
assertType<IsExact<typeof runtimeTensor, Tensor>>()
assertType<typeof classifier extends Program ? true : false>()
assertType<typeof loss extends Program ? true : false>()
assertType<IsExact<typeof derivatives, Program>>()
assertType<typeof optimizer extends Optimizer ? true : false>()
assertType<IsExact<typeof optimizer, Adam>>()
assertType<IsExact<typeof momentumOptimizer, SGD>>()
assertType<IsExact<typeof executable, Executable<FormalTensor>>>()
assertType<IsExact<typeof inferenceOutput, Tensor>>()
assertType<IsExact<typeof pairedOutput, readonly Tensor[]>>()
assertType<IsExact<typeof eagerSum, Tensor>>()
assertType<IsExact<(typeof executable.argument_names)[number], string>>()
assertType<IsExact<(typeof executable.argument_specs)[number], import("affon:compute").ProgramFormal>>()
assertType<IsExact<typeof executable.disposed, boolean>>()
assertType<IsExact<Parameters<typeof executable.run>[0], ProgramArguments>>()
assertType<IsExact<(typeof executionState.parameters)[string], Tensor>>()
assertType<IsExact<(typeof executionState.model_state)[string], Tensor>>()
assertType<IsExact<(typeof executionState.optimizer_state)[string], Tensor | number>>()
assertType<IsExact<(typeof executionState.rng_state)[string], number>>()

program("composed", p => {
  const value = p.argument("image", image)
  const ordinary = classifier(value)
  const explicit = p.use(classifier, { as: "second", image: value })
  void explicit
  return ordinary as FormalTensor
})

program("roles", p => {
  const value = p.argument("value", Tensor.f32([4]))
  const scale = p.parameter("scale", Tensor.f32([4]))
  const running = p.state("running", Tensor.f32([4]))
  const epsilon = p.constant("epsilon", 1e-5, Tensor.f32([]))
  return add(add(mul(value, scale), running), epsilon)
})

program("ops", p => {
  const value = p.argument("value", Tensor.f32([2]))
  const joined = cat([value, value])
  const sum = add(value, value)
  assertType<IsExact<typeof joined, FormalTensor>>()
  assertType<IsExact<typeof sum, FormalTensor>>()
  // @ts-expect-error Tensor operations are public through affon:ops, not FormalTensor methods.
  value.add(value)
  // @ts-expect-error Program builders declare values; operations live in affon:ops.
  p.cat([value, value])
  return reshape(joined, [2, 2])
})

// @ts-expect-error Program hyperparameters are ordinary JavaScript closure values.
program("no_hyperparameters", p => p.hyperparameter("width", 4))

// @ts-expect-error New optimizer options use snake_case.
optimize(loss, adam({ learningRate: 0.001 }))
