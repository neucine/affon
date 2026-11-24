import { getEnv, run } from 'std:process'

function workspaceRoot(): string {
  const cwd = getEnv('AFFON_ROOT') ?? getEnv('PWD') ?? '.'
  const windows = cwd.match(/^\/([A-Za-z])\/(.*)$/)
  return windows ? `${windows[1].toUpperCase()}:/${windows[2]}` : cwd
}

function uniqueName(prefix: string): string {
  return `${workspaceRoot()}/build/test-tmp/${prefix}-${Date.now()}-${Math.floor(Math.random() * 1e9)}`
}

export async function createTempDirectory(prefix: string): Promise<string> {
  const path = uniqueName(prefix)
  await run({ cmd: 'mkdir', args: ['-p', path] })
  return path
}

export async function createTempPath(prefix: string): Promise<string> {
  await run({ cmd: 'mkdir', args: ['-p', `${workspaceRoot()}/build/test-tmp`] })
  return uniqueName(prefix)
}

export async function removeTempPath(path: string): Promise<void> {
  await run({ cmd: 'rm', args: ['-rf', path] })
}
