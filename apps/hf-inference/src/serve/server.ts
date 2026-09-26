// Run from the repository root with Affon.
import http from 'std:http'
import { load_config } from './config.ts'
import { load_models } from '../inference/models.ts'
import { create_handler } from './http/routes.ts'

const config = load_config()
console.log('Loading configured inference models…')
const models = await load_models(config)

http.serve({
  hostname: '127.0.0.1',
  port: config.port,
  maxBodyBytes: 16 * 1024 * 1024,
  maxResponseBytes: 21 * 1024 * 1024,
  requestTimeoutMs: 300000,
  handler: create_handler(config, models),
})
console.log(
  `Affon native model playground ready: ${config.origin} (${config.device})`,
)
