# Optim Concepts

Beginner-friendly mental model for optimizer steps in `affon:compute`.

## Naming note

Current optimizer-related public names are:

- `grad(loss, params)`
- `clear_grad(params)`
- `clip_grad_norm(params, max_norm)`
- `sgd(...)`
- `adam(...)`
- `adamw(...)`

`clear_grad`, `clip_grad_norm`, and `no_grad` are part of the current snake_case public training surface. Use the exported names exactly as they appear.

This guide answers:
- what gradients are
- what an optimizer actually changes
- why `clear_grad()` exists
- how learning-rate schedules fit the training loop
- when gradient clipping matters

## 1. Big picture

Training has two halves:

1. compute how wrong the model is
2. update parameters so it becomes less wrong next time

The optimizer is the second half.

Typical flow:

```txt
forward
loss
grad(loss, params)
step(params)
clear_grad(params)
```

## 2. What `grad(...)` produces

After:

```ts
grad(loss, params)
```

each trainable parameter has a gradient.

Example:

```txt
grad(loss, params)
weight.grad
bias.grad
```

A gradient tells you:
- if this parameter increases slightly, will the loss go up or down?
- how strongly does that parameter affect the current loss?

You can think of it as the local slope of the loss with respect to the parameter.

## 3. What `optimizer.step()` does

The optimizer reads the current gradients and updates parameters.

Very roughly:

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

Where:
- $\theta$ is a parameter
- $\eta$ is the learning rate
- $\nabla_\theta L$ is the gradient of the loss

That is the core idea behind all optimizers.

Current device rule:

- `sgd(...)` can update parameters on `"cpu"` or `"metal"`
- when an SGD parameter lives on `"metal"` but its gradient is still on `"cpu"`, the update path materializes the gradient onto the parameter device before applying the step
- all optimizers skip parameters whose `grad` is `null`

Different optimizers mainly differ in:
- how they scale the update
- whether they use momentum / running averages
- whether they add weight decay

## 4. Why `clear_grad()` exists

Gradients are stored on parameters.

So if you do:

```ts
grad(loss, params)
grad(loss2, params)
```

without clearing gradients, the gradients accumulate.

That is sometimes useful on purpose, but most training loops want fresh gradients per batch.

So the usual pattern is:

```ts
clear_grad(params)
const pred = model(x)
const loss = criterion(pred, target)
grad(loss, params)
step(params)
```

Mental model:
- `grad(...)` writes gradients
- `step(...)` consumes gradients
- `clear_grad(...)` clears them before the next training iteration
- today, Metal training is partial: covered backward paths including `abs`, `relu`, `gelu`, `clamp`, `reshape`, `cat`, `stack`, `gather`, `topk`, and supported axis reductions can now keep gradients on Metal, but some autograd helpers still materialize through CPU

## 5. Core terms

- `gradient`
  The derivative stored on each parameter after `grad(...)`.

- `learning rate`
  The step size used by the optimizer.

- `optimizer step`
  One parameter update using the current gradients.

- `training iteration`
  One full batch pass:
  `forward -> loss -> grad(loss, params) -> step(params)`

- `epoch`
  One full pass over the training dataset.

## 6. SGD

`sgd(...)`

Basic intuition:
- follow the negative gradient direction
- take a small step

Without momentum, SGD is the simplest optimizer.

With momentum, it also remembers some of the recent update direction, which can make learning smoother and faster.

Use when:
- you want the simplest optimizer
- you want predictable behavior
- you are learning the basics
- you may want Metal parameter updates today

## 7. Adam

`adam(...)`

Adam tracks running averages of:
- the gradient
- the squared gradient

This gives it an adaptive step size per parameter.

Rough intuition:
- parameters with large noisy gradients get scaled more carefully
- parameters with small gradients can still move meaningfully

Use when:
- you want a strong default optimizer
- you want less tuning than plain SGD

## 8. Learning-rate schedules

A schedule changes the learning rate over training.

In `affon:compute`, a schedule is a function of training context:

```txt
{ epoch, step }
```

Mental model:
- the optimizer says how to update
- the schedule says how big the learning rate should be at this point in training

## 9. Applying a schedule

`affon:compute` exposes schedule helpers directly:

```ts
const schedule = schedules.sequence(
  schedules.linear({
    start: 0,
    end: 1e-3,
    duration: Duration.steps(100),
  }),
  schedules.cosine({
    start: 1e-3,
    end: 5e-4,
    duration: Duration.steps(900),
  }),
  schedules.constant(5e-4),
)
```

You can either assign `step.lr` manually, or wrap a step function:

```ts
const step = adam({ lr: 1e-3 })
const scheduledStep = scheduled(step, schedule)
```

Then in the loop:

```ts
scheduledStep.epoch(epoch)
scheduledStep(params)
```

This is useful when:
- learning rate should warm up
- decay should happen after a fixed number of steps
- workflow config should translate into a plain compute schedule callback or scheduled step wrapper

## 11. Gradient clipping

`clip_grad_norm(params, max_norm)`

This rescales gradients when their global norm is too large.

Why this matters:
- some models can produce exploding gradients
- recurrent models are a classic case
- very large gradients can make training unstable

Mental model:
- compute the overall gradient size
- if it is too large, scale all gradients down proportionally
- keep the direction, reduce the magnitude

## 12. One complete training loop

```ts
for (let epoch = 0; epoch < epochs; epoch++) {
  clear_grad(params)
  const pred = model(x)
  const loss = criterion(pred, target)
  grad(loss, params)
  step(params)
}
```

With schedule and clipping:

```ts
for (let epoch = 0; epoch < epochs; epoch++) {
  step.lr = schedule({ epoch, step: globalStep })
  clear_grad(params)
  const pred = model(x)
  const loss = criterion(pred, target)
  grad(loss, params)
  clip_grad_norm(params, 1.0)
  step(params)
}
```

## 13. Quick memory aid

- `grad(loss, params)`
  compute gradients

- `clear_grad(params)`
  clear old gradients

- `step(params)`
  update parameters

- `schedule`
  decide the learning rate over time

- `clip_grad_norm()`
  keep gradients from becoming too large
