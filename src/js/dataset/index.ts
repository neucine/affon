import 'affon:errors'
import { read, DataLoader, tabular } from 'affon:dataset/tabular.ts'
import { readText, text, TextDataLoader, EncodedTextDataLoader, PaddedTextDataLoader, TensorizedTextDataLoader } from 'affon:dataset/text.ts'

const mod: any = {}
mod.read = read
mod.DataLoader = DataLoader
mod.tabular = tabular
mod.text = text

export { read, readText, DataLoader, tabular, text, TextDataLoader, EncodedTextDataLoader, PaddedTextDataLoader, TensorizedTextDataLoader }
export default mod
