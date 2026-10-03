import { add, argmax, cat, cross_entropy, mean_squared_error, mul, reshape, stack } from 'affon:ops'
import {
  Tensor,
  Session,
  gradient,
  metrics,
  optimize,
  program,
  update_parameters,
  type Callable,
  type Executable,
  type FormalTensor,
  type Program,
  type ProgramArguments,
  type TensorSpec,
} from "affon:compute"
import { accumulate, adam, scheduled, schedules, sgd, type AccumulatingOptimizer, type Adam, type Optimizer, type ScheduledOptimizer, type SGD } from "affon:optim"
import * as compute from "affon:compute"
import {
  binary_cross_entropy,
  binary_cross_entropy_with_logits,
  cross_entropy as crossEntropyLoss,
  linear,
  mean_squared_error as meanSquaredErrorLoss,
  type LossCallable,
} from "affon:nn"

// @ts-expect-error The removed compute compatibility module must stay unavailable.
import "affon:compute/legacy"
// @ts-expect-error The removed neural-network compatibility module must stay unavailable.
import "affon:nn/legacy"
import * as optimModule from "affon:optim"

// @ts-expect-error Compute implementation modules are not part of the public type graph.
import * as internalCompute from "affon:_internal/compute/program"
void internalCompute

// @ts-expect-error Tensor operations belong to affon:ops.
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
const defaultValue = Tensor.from([1, 2, 3])
const defaultZeros = Tensor.zeros([2, 3], { dtype: "f64" })
const defaultOnes = Tensor.ones([2, 3])
const defaultFull = Tensor.full([2, 3], 4)
const defaultRange = Tensor.arange(0, 3)
const defaultRandom = Tensor.randn([2, 3], { seed: 7 })
// @ts-expect-error bool is not executable by the compute core.
Tensor.bool([2])
const classifier = program("classifier", p => {
  const value = p.argument("image", image)
  return linear({ out_features: 10 })({ x: value }, "projection")
})
const loss = program("classifier_loss", p => {
  const logits = classifier({ image: p.argument("image", image) })
  return cross_entropy(logits, p.argument("labels", Tensor.i64([32])))
})
const derivatives = gradient(loss, ["classifier.projection.weight", "classifier.projection.bias"])
const explicitUpdate = update_parameters(loss, derivatives, sgd({ learning_rate: 0.01 }))
const optimizer = adam({ learning_rate: 0.001 })
const momentumOptimizer = sgd({ momentum: 0.9 })
const accumulatingOptimizer = accumulate(optimizer, { steps: 4 })
const scheduledOptimizer = scheduled(optimizer, schedules.cosine({ start: 0.001, end: 0.0001, steps: 100 }))
const scheduledAccumulatingOptimizer = accumulate(scheduledOptimizer, { steps: 4 })
const separateLoss = program("cross_entropy_loss", p => cross_entropy(
  p.argument("logits", Tensor.f32([32, 10])),
  p.argument("labels", Tensor.i64([32])),
))
const training_step = optimize(classifier, separateLoss, optimizer)
const reusableTrainingStep = optimize(classifier, separateLoss, optimizer)
const builtInLoss: LossCallable = crossEntropyLoss()
const builtInTrainingStep = optimize(classifier, builtInLoss, optimizer)
optimize(classifier, builtInLoss, accumulatingOptimizer)
optimize(classifier, builtInLoss, scheduledAccumulatingOptimizer)
optimize(classifier, meanSquaredErrorLoss(), optimizer)
optimize(classifier, binary_cross_entropy(), optimizer)
optimize(classifier, binary_cross_entropy_with_logits(), optimizer)
// @ts-expect-error optimize always requires model, loss, and optimizer.
optimize(loss, optimizer)
const session = new Session({ device: "cpu" })
const runtimeTensor: Tensor = session.tensor([1, 2, 3])
const executable = session.compile(training_step)
const reusableExecutable = session.compile(reusableTrainingStep)
const builtInExecutable = session.compile(builtInTrainingStep)
const executionState = session.initialize(training_step)
session.compile(explicitUpdate)
const inferenceExecutable = session.compile(classifier)
const inferenceOutput = inferenceExecutable.run({ image: runtimeTensor })
const paired = program("paired", p => {
  const value = p.argument("value", Tensor.f32([3]))
  return [value, value] as const
})
const pairedOutput = session.compile(paired).run({ value: runtimeTensor })
const evaluatedSum = add(runtimeTensor, runtimeTensor)
const evaluatedMetric: number = metrics.mean_absolute_error(runtimeTensor, runtimeTensor)
const evaluatedLoss = mean_squared_error(runtimeTensor, runtimeTensor)
const evaluatedIndices = argmax(runtimeTensor)
const evaluatedStack = stack([runtimeTensor, runtimeTensor])
void evaluatedMetric

assertType<IsExact<typeof image, TensorSpec>>()
assertType<IsExact<typeof defaultValue, Tensor>>()
assertType<IsExact<typeof defaultZeros, Tensor>>()
assertType<IsExact<typeof defaultOnes, Tensor>>()
assertType<IsExact<typeof defaultFull, Tensor>>()
assertType<IsExact<typeof defaultRange, Tensor>>()
assertType<IsExact<typeof defaultRandom, Tensor>>()
// @ts-expect-error The default Session cannot be replaced or configured in code.
Session.default = new Session()
assertType<IsExact<typeof runtimeTensor, Tensor>>()
assertType<typeof classifier extends Program ? true : false>()
assertType<typeof loss extends Program ? true : false>()
assertType<IsExact<typeof derivatives, Program>>()
assertType<typeof optimizer extends Optimizer ? true : false>()
assertType<IsExact<typeof optimizer, Adam>>()
assertType<IsExact<typeof momentumOptimizer, SGD>>()
assertType<IsExact<typeof accumulatingOptimizer, AccumulatingOptimizer>>()
assertType<IsExact<typeof scheduledOptimizer, ScheduledOptimizer>>()
assertType<IsExact<typeof executable, Executable<FormalTensor>>>()
assertType<IsExact<typeof reusableExecutable, Executable<FormalTensor>>>()
assertType<IsExact<typeof builtInExecutable, Executable<FormalTensor>>>()
assertType<IsExact<typeof inferenceOutput, Tensor>>()
assertType<IsExact<typeof pairedOutput, readonly Tensor[]>>()
assertType<IsExact<typeof evaluatedSum, Tensor>>()
assertType<IsExact<typeof evaluatedLoss, Tensor>>()
assertType<IsExact<typeof evaluatedIndices, Tensor>>()
assertType<IsExact<typeof evaluatedStack, Tensor>>()
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
  // @ts-expect-error Programs compose through named bindings.
  classifier(value)
  const ordinary = classifier({ image: value })
  const explicit = classifier({ image: value }, "second")
  void explicit
  return ordinary as FormalTensor
})

const projection: Callable<{ x: FormalTensor }, FormalTensor> = linear({ out_features: 4 })
program("layer_factory", p => projection({ x: p.argument("x", Tensor.f32([2, 3])) }, "head"))

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
  // @ts-expect-error Neural-network layers are ordinary factories from affon:nn.
  p.nn.linear(value, { name: "removed", out_features: 2 })
  // @ts-expect-error Program composition is expressed by calling the Program.
  p.use(classifier, { as: "removed", image: value })
  return reshape(joined, [2, 2])
})

// @ts-expect-error Program hyperparameters are ordinary JavaScript closure values.
program("no_hyperparameters", p => p.hyperparameter("width", 4))

// @ts-expect-error New optimizer options use snake_case.
optimize(classifier, separateLoss, adam({ learningRate: 0.001 }))
