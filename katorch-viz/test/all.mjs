/**
 * 一次跑完全部测试。
 *   node test/all.mjs
 */
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))

const suites = [
  ['数据层：往返一致 + 图操作', 'roundtrip.mjs'],
  ['数据层：形状推导（与 Python 对拍）', 'synthetic.mjs'],
  ['Host 半：HTTP 路由 + agent 工具', 'host-smoke.mjs'],
  ['Client 半：加载 + apply + 渲染', 'client-smoke.mjs'],
]

let failed = 0
for (const [label, file] of suites) {
  console.log(`\n${'='.repeat(64)}\n${label}  (${file})\n${'='.repeat(64)}`)
  const res = spawnSync(process.execPath, [path.join(here, file)], { stdio: 'inherit' })
  if (res.status !== 0) failed += 1
}

console.log(`\n${'='.repeat(64)}`)
if (failed) {
  console.log(`${failed} / ${suites.length} 组测试没通过`)
  process.exit(1)
}
console.log('全部测试通过')
