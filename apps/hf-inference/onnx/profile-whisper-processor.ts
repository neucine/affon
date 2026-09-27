import fs from 'std:fs'
import { getEnv } from 'std:process'
import { load_whisper_processor } from '../../../packages/@affon/huggingface/src/processors/whisper.ts'
import { decode_wav } from '../../../packages/@affon/huggingface/src/processors/shared/audio.ts'
const bytes = new Uint8Array(
  JSON.parse(fs.readFileSync(getEnv('PROFILE_AUDIO_BYTES')!)),
)
const audio = decode_wav(bytes)
const p = load_whisper_processor(
  `${getEnv('AFFON_WHISPER_DIR') ?? '/private/tmp/affon-onnx-whisper'}/source`,
  'metal',
)
p.process(audio.samples, audio.sampling_rate)
const runs: Record<string, number>[] = []
for (let i = 0; i < 3; i++)
  p.process(audio.samples, audio.sampling_rate, (t) => runs.push(t))
fs.writeFileSync(getEnv('PROFILE_OUTPUT')!, JSON.stringify({ runs }, null, 2))
