import { describe, test, expect } from 'std:test'
import { axes, tensor } from 'affon:compute'
import { inspect } from 'std:util'

describe('compute repr', () => {
  test('renders a small 1D tensor', () => {
    const vec = tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    expect(vec.repr()).toMatchInlineSnapshot(`
      [5 Tensor, dtype=f32]
        0  1  2  3  4  
      [ 1  2  3  4  5  ]
    `)
  })

  test('renders axis names in text repr when present', () => {
    const mat = tensor([[1.0, 2.0], [3.0, 4.0]], {
      axes: [axes.batch, axes.feature],
    })

    expect(mat.repr()).toMatchInlineSnapshot(`
      [2×2 Tensor, dtype=f32, axes=[batch, feature]]
           0  1
        ┌        ┐
      0 │  1  2  │
      1 │  3  4  │
        └        ┘
    `)
  })

  test('renders a small 2D tensor in html mode', () => {
    const mat = tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    const html = mat.repr({ mode: 'html' }) as { mime: string; data: string }
    expect(html.mime).toBe('text/html')
    expect(html.data).toMatchInlineSnapshot(`
      <table class="repr">
      <tr><th></th><th>0</th><th>1</th><th>2</th></tr>
      <tr><td class="repr-idx">0</td><td>1</td><td>0</td><td>0</td></tr>
      <tr><td class="repr-idx">1</td><td>0</td><td>1</td><td>0</td></tr>
      <tr><td class="repr-idx">2</td><td>0</td><td>0</td><td>1</td></tr>
      </table>
    `)
  })

  test('renders axis names in html repr when present', () => {
    const mat = tensor([[1.0, 2.0]], {
      axes: ['batch', 'feature<&>'],
    })

    const html = mat.repr({ mode: 'html' }) as { mime: string; data: string }
    expect(html.mime).toBe('text/html')
    expect(html.data).toMatchInlineSnapshot(`
      <div class="repr-meta">1×2 Tensor, dtype=f32, axes=[batch, feature&lt;&amp;&gt;]</div>
      <table class="repr">
      <tr><th></th><th>0</th><th>1</th></tr>
      <tr><td class="repr-idx">0</td><td>1</td><td>2</td></tr>
      </table>
    `)
  })

  test('renders a sparse 2D tensor', () => {
    const mat = tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    expect(mat.repr({ sparse: true })).toMatchInlineSnapshot(`
      [3×3 Tensor, dtype=f32]
           0  1  2
        ┌           ┐
      0 │  1        │
      1 │     1     │
      2 │        1  │
        └           ┘
    `)
  })

  test('renders a small 4D tensor in html mode', () => {
    const t4d = tensor([
      [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
      [[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]],
    ])

    const html = t4d.repr({ mode: 'html' }) as { mime: string; data: string }
    expect(html.mime).toBe('text/html')
    expect(html.data).toMatchInlineSnapshot(`
      <table class="repr repr-nd">
      <tr><td class="repr-idx">(0,0)</td><td>
      <table class="repr repr-inner">
      <tr><th></th><th>0</th><th>1</th><th>2</th></tr>
      <tr><td class="repr-idx">0</td><td>1</td><td>2</td><td>3</td></tr>
      <tr><td class="repr-idx">1</td><td>4</td><td>5</td><td>6</td></tr>
      </table>
      </td></tr>
      <tr><td class="repr-idx">(0,1)</td><td>
      <table class="repr repr-inner">
      <tr><th></th><th>0</th><th>1</th><th>2</th></tr>
      <tr><td class="repr-idx">0</td><td>7</td><td>8</td><td>9</td></tr>
      <tr><td class="repr-idx">1</td><td>10</td><td>11</td><td>12</td></tr>
      </table>
      </td></tr>
      <tr><td class="repr-idx">(1,0)</td><td>
      <table class="repr repr-inner">
      <tr><th></th><th>0</th><th>1</th><th>2</th></tr>
      <tr><td class="repr-idx">0</td><td>13</td><td>14</td><td>15</td></tr>
      <tr><td class="repr-idx">1</td><td>16</td><td>17</td><td>18</td></tr>
      </table>
      </td></tr>
      <tr><td class="repr-idx">(1,1)</td><td>
      <table class="repr repr-inner">
      <tr><th></th><th>0</th><th>1</th><th>2</th></tr>
      <tr><td class="repr-idx">0</td><td>19</td><td>20</td><td>21</td></tr>
      <tr><td class="repr-idx">1</td><td>22</td><td>23</td><td>24</td></tr>
      </table>
      </td></tr>
      </table>
    `)
  })

  test('uses compact toString and inspect by default', () => {
    const vec = tensor([1, 2, 3])
    const mat = tensor([[1, 2], [3, 4]])

    expect(String(vec)).toBe('Tensor([1, 2, 3], dtype=f32)')
    expect(inspect(vec)).toContain('[3 Tensor, dtype=f32]')
    expect(String(mat)).toBe('Tensor([[1, 2], [3, 4]], dtype=f32)')
    expect(inspect(mat)).toContain('[2×2 Tensor, dtype=f32]')
  })
})
