import compute from 'affon:compute'

const x = compute.tensor([[1, 2], [3, 4]])

const _toString: string = x.toString()
const _reprText: string = x.repr()

void _toString
void _reprText
