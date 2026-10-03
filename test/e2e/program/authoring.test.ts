import { add, cross_entropy, mean, reshape } from "affon:ops"
import { describe, expect, test } from "std:test"
import { Tensor, gradient, optimize, program } from "affon:compute"
import { adam } from "affon:optim"

describe("Program authoring and transforms", () => {
  test("authors, composes, transforms, and inspects one symbolic pipeline", () => {
    const imageSpec = Tensor.f32([2, 4], { axes: ["batch", "feature"] })
    const model = program("classifier", p => {
      const image = p.argument("image", imageSpec)
      return p.nn.linear(image, { name: "head", out_features: 3 })
    })
    const composed = program("ensemble", p => {
      const image = p.argument("image", imageSpec)
      const first = model(image)
      const second = p.use(model, { as: "second", image })
      return add(first, second as typeof first)
    })
    const loss = program("ensemble_loss", p => {
      const image = p.argument("image", imageSpec)
      const labels = p.argument("labels", Tensor.i64([2]))
      return cross_entropy(composed(image), labels)
    })
    const optimizationLoss = program("ensemble_optimization_loss", p => cross_entropy(
      p.argument("logits", Tensor.f32([2, 3])),
      p.argument("labels", Tensor.i64([2])),
    ))
    const gradients = gradient(loss, ["ensemble_classifier_head_weight", "ensemble_second_head_weight"])
    const step = optimize(composed, optimizationLoss, adam({ learning_rate: 0.001 }))
    const fasterStep = optimize(composed, optimizationLoss, adam({ learning_rate: 0.01 }))

    expect(loss.inspect().arguments.map(value => [value.name, value.spec.dtype, value.spec.shape])).toEqual([
      ["image", "f32", [2, 4]],
      ["labels", "i64", [2]],
    ])
    expect(loss.inspect().parameters.map(value => value.name)).toEqual([
      "ensemble_classifier_head_weight",
      "ensemble_classifier_head_bias",
      "ensemble_second_head_weight",
      "ensemble_second_head_bias",
    ])
    expect(gradients.inspect().outputs.length).toBe(2)
    expect(step.inspect().transitions[0].kind).toBe("optimize")
    expect(step.provenance === fasterStep.provenance).toBe(false)
  })

  test("enforces exact composition bindings and deeply immutable inspection metadata", () => {
    const child = program("composition_child", p => add(
      p.argument("left", Tensor.f32([2])),
      p.argument("right", Tensor.f32([2])),
    ))
    expect(() => program("extra_positional", p => {
      const value = p.argument("value", Tensor.f32([2]))
      return (child as any)(value, value, value)
    })).toThrow("expects 2 arguments")
    expect(() => program("extra_named", p => {
      const value = p.argument("value", Tensor.f32([2]))
      return p.use(child, { as: "child", left: value, right: value, extra: value })
    })).toThrow("unknown composition_child argument")

    const scalar = program("composition_scalar", p => mean(p.argument("value", Tensor.f32([2]))))
    const derivative = gradient(scalar, "value")
    expect(() => program("composed_gradient", p => derivative(p.argument("value", Tensor.f32([2]))))).toThrow("only authored Programs")

    const shaped = program("immutable_metadata", p => reshape(p.argument("value", Tensor.f32([2, 2])), [4]))
    const options = shaped.inspect().nodes.at(-1)!.options as any
    expect(Object.isFrozen(options)).toBe(true)
    expect(Object.isFrozen(options.shape)).toBe(true)
  })

})
