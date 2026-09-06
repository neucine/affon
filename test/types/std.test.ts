import { tensor } from 'affon:compute'
import { figure, plot, show } from 'std:plot'
import { inspect } from 'std:util'
import { get } from 'std:http'
import runtime from 'std:runtime'
import ffi from 'std:ffi'
import c from 'std:ffi/c'

figure({ width: 600, height: 400 })
plot(tensor([1, 2, 3]))
plot([1, 2], [3, 4], { label: 'series' })
const svg: string = show().repr().data
const description: string = inspect({ svg })
const name: 'hao' = runtime.name
const response = get('https://example.com')
void [ffi, c, description, name, response]

// @ts-expect-error Plot dimensions must be numeric.
figure({ width: 'wide' })
// @ts-expect-error Inspect returns a string.
const invalid: number = inspect({})
// @ts-expect-error Unknown std modules must not silently become any.
import missing from 'std:nonexistent'
