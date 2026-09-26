import { snapshot_download } from './hub.ts'
import type { HubOptions } from './hub.ts'
import { load_model } from './model.ts'
import type { ModelOptions, NativeModelTask } from './model.ts'
import { load_processor } from './processor.ts'

/** Load a native model and its processor from a pinned public Hub snapshot.
 * @param model_id Hub repository ID.
 * @param options Task, immutable revision, cache directory, device, and offline policy.
 * @returns Local snapshot path, native model, and corresponding processor.
 */
export async function from_pretrained<T extends NativeModelTask>(model_id: string, options: HubOptions & ModelOptions<T>) {
  const directory = await snapshot_download(model_id, options)
  const processor = load_processor(directory, options)
  const model = load_model(directory, options)
  return { directory, model, processor }
}
