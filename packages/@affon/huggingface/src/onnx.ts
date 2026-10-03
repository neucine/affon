import { load_graph } from '../../onnx/src/index.ts'

/** ONNX is an execution backend, not an HF architecture. Preparation is explicit. */
export type OnnxModelOptions = {
  task: 'image-classification' | 'audio-classification'
  backend: 'onnx'
  graph_dir: string
  input_name?: string
  output_name?: string
}
export function load_onnx_classifier(
  config: Record<string, any>,
  options: OnnxModelOptions,
) {
  if (!['image-classification', 'audio-classification'].includes(options.task))
    throw Error('Unsupported ONNX classification task')
  if (!options.graph_dir)
    throw Error('ONNX backend requires a prepared graph_dir')
  const graph = load_graph(options.graph_dir)
  const inputs = Object.keys(graph.graph.inputs)
  if (inputs.length !== 1)
    throw Error('HF classification requires one graph input')
  const input = options.input_name ?? inputs[0]
  const output =
    options.output_name ??
    (graph.graph.outputs.length === 1 ? graph.graph.outputs[0] : undefined)
  if (input !== inputs[0] || !output || !graph.graph.outputs.includes(output))
    throw Error('Invalid or ambiguous ONNX input/output binding')
  const input_shape = graph.graph.inputs[input]
  const output_shape =
    graph.graph.nodes.find((node) => node.output === output)?.shape ??
    graph.graph.constants[output]
  const valid_input =
    options.task === 'image-classification'
      ? input_shape.length === 4 && input_shape[1] === 3
      : config.model_type === 'audio-spectrogram-transformer' &&
        input_shape.length === 3 &&
        input_shape[1] === config.max_length &&
        input_shape[2] === config.num_mel_bins
  if (
    !valid_input ||
    !output_shape ||
    output_shape.length !== 2 ||
    output_shape[0] !== input_shape[0]
  ) {
    throw Error(
      'HF classifier requires task-compatible input dimensions and [batch, classes] logits',
    )
  }
  const labels = config.id2label
  if (
    labels &&
    (typeof labels !== 'object' ||
      Object.keys(labels).length !== output_shape[1] ||
      Array.from({ length: output_shape[1] }, (_, i) => labels[String(i)]).some(
        (x) => typeof x !== 'string',
      ))
  ) {
    throw Error('HF labels do not match ONNX class count')
  }
  return {
    config,
    backend: 'onnx' as const,
    parameters: graph.parameters,
    output_names: graph.output_names,
    input_name: input,
    output_name: output,
    input_shape: [...input_shape],
    forward: graph.forward,
  }
}
export type OnnxClassifier = ReturnType<typeof load_onnx_classifier>
export type OnnxImageClassifier = OnnxClassifier

export type OnnxAudioClassifier = OnnxClassifier
