import { add, cross_entropy, mean, reshape } from "affon:ops"
import { describe, expect, test } from "std:test"
import { Tensor, gradient, optimize, program } from "affon:compute"
import { linear } from "affon:nn"
import { adam } from "affon:optim"

describe("Program authoring and transforms", () => {
  test("authors, composes, transforms, and inspects one symbolic pipeline", () => {
    const imageSpec = Tensor.f32([2, 4], { axes: ["batch", "feature"] })
    const model = program("classifier", p => {
      const image = p.argument("image", imageSpec)
      return linear({ out_features: 3 })({ x: image }, "head")
    })
    const composed = program("ensemble", p => {
      const image = p.argument("image", imageSpec)
      const first = model({ image })
      const second = model({ image }, "second")
      return add(first, second as typeof first)
    })
    const loss = program("ensemble_loss", p => {
      const image = p.argument("image", imageSpec)
      const labels = p.argument("labels", Tensor.i64([2]))
      return cross_entropy(composed({ image }), labels)
    })
    const optimizationLoss = program("ensemble_optimization_loss", p => cross_entropy(
      p.argument("logits", Tensor.f32([2, 3])),
      p.argument("labels", Tensor.i64([2])),
    ))
    const gradients = gradient(loss, ["ensemble.classifier.head.weight", "ensemble.second.head.weight"])
    const step = optimize(composed, optimizationLoss, adam({ learning_rate: 0.001 }))
    const fasterStep = optimize(composed, optimizationLoss, adam({ learning_rate: 0.01 }))

    expect(loss.inspect().arguments.map(value => [value.name, value.spec.dtype, value.spec.shape])).toEqual([
      ["image", "f32", [2, 4]],
      ["labels", "i64", [2]],
    ])
    expect(loss.inspect().parameters.map(value => value.name)).toEqual([
      "ensemble.classifier.head.weight",
      "ensemble.classifier.head.bias",
      "ensemble.second.head.weight",
      "ensemble.second.head.bias",
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
    expect(() => program("positional_binding", p => {
      const value = p.argument("value", Tensor.f32([2]))
      return (child as any)(value)
    })).toThrow("named bindings")
    expect(() => program("extra_named", p => {
      const value = p.argument("value", Tensor.f32([2]))
      return child({ left: value, right: value, extra: value }, "child")
    })).toThrow("unknown composition_child argument")

    const scalar = program("composition_scalar", p => mean(p.argument("value", Tensor.f32([2]))))
    const derivative = gradient(scalar, "value")
    expect(() => program("composed_gradient", p => derivative({ value: p.argument("value", Tensor.f32([2])) }))).toThrow("only authored Programs")

    const shaped = program("immutable_metadata", p => reshape(p.argument("value", Tensor.f32([2, 2])), [4]))
    const options = shaped.inspect().nodes.at(-1)!.options as any
    expect(Object.isFrozen(options)).toBe(true)
    expect(Object.isFrozen(options.shape)).toBe(true)
  })

  test("derives immutable nested composition paths without changing value-node identities", () => {
    const leaf = program("path_leaf", p => add(
      p.argument("value", Tensor.f32([2])),
      p.parameter("bias", Tensor.f32([2])),
    ))
    const block = program("path_block", p => leaf({
      value: p.argument("value", Tensor.f32([2])),
    }, "projection"))
    const model = program("path_model", p => block({
      value: p.argument("value", Tensor.f32([2])),
    }, "layer.0"))

    const inspection = model.inspect()
    expect(inspection.schema_version).toBe(1)
    expect(inspection.constant_values).toBe("inline")
    expect(inspection.provenance_id.startsWith("affon:")).toBe(true)
    expect(inspection.nodes.map(node => node.id)).toEqual(inspection.nodes.map((_, index) => index))
    expect(inspection.nodes.find(node => node.name === "layer.0.projection.bias")!.path).toEqual([
      { program: "path_block", instance: "layer.0" },
      { program: "path_leaf", instance: "projection" },
    ])
    expect(inspection.nodes.at(-1)!.path).toEqual([
      { program: "path_block", instance: "layer.0" },
      { program: "path_leaf", instance: "projection" },
    ])
    expect(Object.isFrozen(inspection.nodes.at(-1)!.path)).toBe(true)
    expect(Object.isFrozen(inspection.nodes.at(-1)!.path[0])).toBe(true)
    expect(inspection.arguments[0].name).toBe("value")
    expect(inspection.nodes[inspection.outputs[0]].operands).toEqual([0, 1])
    expect(inspection.components.length).toBe(2)
    expect(inspection.components[0].program).toBe("path_block")
    expect(inspection.components[0].bindings).toEqual({ value: 0 })
    expect(inspection.components[0].outputs).toEqual([inspection.outputs[0]])
    expect(inspection.components[1].path).toEqual([
      { program: "path_block", instance: "layer.0" },
      { program: "path_leaf", instance: "projection" },
    ])
    expect(Object.isFrozen(inspection.components)).toBe(true)
    expect(Object.isFrozen(inspection.components[0].bindings)).toBe(true)
  })

  test("captures exact multi-output component interfaces without author annotations", () => {
    const child = program("interface_child", p => {
      const left = p.argument("left", Tensor.f32([2]))
      const right = p.argument("right", Tensor.f32([2]))
      return [add(left, right), add(right, right)]
    })
    const parent = program("interface_parent", p => {
      const x = p.argument("x", Tensor.f32([2]))
      const y = p.argument("y", Tensor.f32([2]))
      return child({ left: x, right: y }, "pair")
    })
    const inspection = parent.inspect()
    expect(inspection.components).toEqual([{
      id: "component:interface_child@pair#0",
      program: "interface_child",
      instance: "pair",
      path: [{ program: "interface_child", instance: "pair" }],
      bindings: { left: 0, right: 1 },
      outputs: inspection.outputs,
    }])

    const repeated = program("repeated_interface", p => {
      const x = p.argument("x", Tensor.f32([2]))
      const first = child({ left: x, right: x }, "shared")
      const second = child({ left: x, right: x }, "shared")
      return add(first[0], second[0])
    }).inspect()
    expect(repeated.components.map(component => component.id)).toEqual([
      "component:interface_child@shared#0",
      "component:interface_child@shared#1",
    ])
  })

  test("authors and composes a shape-specialized child from a callable", () => {
    const reusable = ({ value }: { value: import("affon:compute").FormalTensor }, name = "reusable") => {
      const width = value.spec.shape[0]
      const child = program("shape_specialized_child", p => add(
        p.argument("value", Tensor.f32([width])),
        p.parameter("bias", Tensor.f32([width])),
      ))
      return child({ value }, name)
    }
    const parent = program("shape_specialized_parent", p => reusable({
      value: p.argument("value", Tensor.f32([3])),
    }, "component"))

    expect(parent.inspect().parameters.map(value => value.name)).toEqual(["component.bias"])
    expect(parent.inspect().nodes.at(-1)!.path).toEqual([
      { program: "shape_specialized_child", instance: "component" },
    ])
  })

})
