import { test, expect } from 'std:test'
import telemetry, {type MetricSnapshot} from 'std:telemetry'
import { create_phase_profiler, metric_changes } from '../src/benchmark/telemetry.ts'
const metric = (name:string, value:number, kind:MetricSnapshot['kind']='counter'):MetricSnapshot => ({id:0,scope:'compute.execution',name,value,kind,unit:'count',count:0,sum:0,min:0,max:0})
test('profile distinguishes counter deltas, gauge snapshots and unavailable GPU timing', () => {
  const before = [metric('metal_command_count',10),metric('metal_command_gpu_valid_count',10),metric('peak_bytes',100,'gauge')]
  const after = [metric('metal_command_count',15),metric('metal_command_gpu_valid_count',14),metric('peak_bytes',100,'gauge'),metric('transfer_to_host_bytes',64)]
  const result = metric_changes(before,after)
  expect(result.counters['compute.execution.metal_command_count']).toBe(5)
  expect(result.counters['compute.execution.transfer_to_host_bytes']).toBe(64)
  expect(result.gauges['compute.execution.peak_bytes']).toEqual({before:100,after:100})
  expect(result.metal_gpu_timing_complete).toBe(false)
  after[1].value = 15
  expect(metric_changes(before,after).metal_gpu_timing_complete).toBe(true)
  expect(metric_changes([],[]).metal_gpu_timing_complete).toBe(null)
})
test('phase profiler records real telemetry and ends failed sync/async scopes', async () => {
  const profile = create_phase_profiler()
  const counter = telemetry.counter({scope:'compute.execution',name:'profile_test_count',unit:'count'})
  expect(profile.sync('profile.test.ok',()=>{counter.add(3);return 42})).toBe(42)
  expect(profile.phases[0].counters['compute.execution.profile_test_count']).toBe(3)
  expect(()=>profile.sync('profile.test.error',()=>{throw Error('expected')})).toThrow('expected')
  try { await profile.async('profile.test.async_error',async()=>{throw Error('expected')}) } catch {}
  expect(await profile.async('profile.test.async_ok',async()=>7)).toBe(7)
  expect(profile.phases.map(p=>p.status)).toEqual(['ok','err','err','ok'])
})
