import { run } from 'std:process'

export type CandidateBackend = 'cpu' | 'metal' | 'cuda'
export type DecoderProof = {
  backend: CandidateBackend
  loss: number
  resumedParameter: number[]
  explanation: string
}

export async function runDecoderProof(backend: CandidateBackend = 'cpu'): Promise<DecoderProof> {
  const result = await run({ cmd: './zig-out/bin/affon-candidate-canary', args: [backend] })
  return JSON.parse(result.stdout) as DecoderProof
}
