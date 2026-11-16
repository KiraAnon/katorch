/**
 * Host 冒烟测试 —— 不依赖 DSH 重启，直接 import index.js 的 apply()，
 * 用假 ctx 拿到它注册的 HTTP 路由和 agent 工具，然后真打一遍。
 *
 * 覆盖：/state、/open、/analyze、/save、/browse、write-const，以及 katorch_viz 工具。
 * 跑法：node test/host-smoke.mjs
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { EventEmitter } from 'node:events'
import { fileURLToPath } from 'node:url'

import { apply } from '../index.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const SAMPLE = process.env.KATORCH_SAMPLE || 'F:\\KAtorch\\PC编译\\wsl-sysroot\\n1\\g.txt'

let pass = 0
const failures = []
const ok = (tag, cond, detail) => {
  if (cond) { pass += 1; console.log(`  PASS ${tag}`) }
  else { failures.push(`${tag}${detail ? `: ${detail}` : ''}`); console.log(`  FAIL ${tag}${detail ? `: ${detail}` : ''}`) }
}

// ---- 假 ctx ---------------------------------------------------------------
const routes = []
const tools = []
const effects = []

const ctx = {
  effect(fn, label) {
    effects.push(label || 'effect')
    const dispose = fn()
    return () => { if (typeof dispose === 'function') dispose() }
  },
  webServer: {
    register(route) { routes.push(route); return () => {} },
  },
  tools: {
    register(def) { tools.push(def); return () => {} },
  },
}

console.log('=== 跑 apply ===')
try {
  apply(ctx, {
    defaultPath: SAMPLE,
    constRewrite: [['/data/data/com.termux/files/home/npt/', 'F:/KAtorch/artifacts/npt/']],
  })
} catch (err) {
  console.log(`  FAIL apply：${err.stack || err.message}`)
  process.exit(1)
}
ok('apply 没抛异常', true)
ok('注册了 HTTP 路由', routes.length === 1, `${routes.length} 条`)
ok('路由挂在 /katorch-viz 前缀', routes[0] && routes[0].path === '/katorch-viz' && routes[0].kind === 'prefix',
  routes[0] && `${routes[0].kind} ${routes[0].path}`)
ok('注册了 agent 工具 katorch_viz', tools.some((x) => x.name === 'katorch_viz'))

// ---- 假 req / res ---------------------------------------------------------
function mkReq(method, url, body) {
  const req = new EventEmitter()
  req.method = method
  req.url = url
  if (body !== undefined) {
    process.nextTick(() => {
      req.emit('data', Buffer.from(JSON.stringify(body)))
      req.emit('end')
    })
  } else {
    process.nextTick(() => req.emit('end'))
  }
  return req
}

function mkRes() {
  return {
    code: null,
    headers: null,
    body: '',
    writeHead(code, headers) { this.code = code; this.headers = headers },
    end(text) { this.body = text },
    done: false,
  }
}

async function call(method, route, body) {
  const routePath = '/katorch-viz' + route
  const req = mkReq(method, routePath, body)
  const res = mkRes()
  // handler 是 (req,res)=>void，内部 async —— 等它把 res.end 调完
  await new Promise((resolve) => {
    const origEnd = res.end.bind(res)
    res.end = (text) => { origEnd(text); res.done = true; resolve() }
    Promise.resolve(routes[0].handler(req, res)).catch((e) => { res.body = JSON.stringify({ error: String(e) }); resolve() })
    setTimeout(() => resolve(), 8000)
  })
  let json = null
  try { json = JSON.parse(res.body) } catch { /* 非 JSON */ }
  return { code: res.code, json, raw: res.body }
}

// ---- /state ---------------------------------------------------------------
console.log('\n=== GET /state ===')
{
  const r = await call('GET', '/state')
  ok('200', r.code === 200, `得 ${r.code}`)
  ok('返回 ok', r.json && r.json.ok === true)
  ok('带 config', r.json && r.json.config && 'constRewrite' in r.json.config)
  ok('带编译能力探测', r.json && r.json.compiler && typeof r.json.compiler.available === 'boolean')
}

// ---- /open ----------------------------------------------------------------
console.log('\n=== POST /open ===')
let opened = null
{
  const r = await call('POST', '/open', { path: SAMPLE })
  opened = r.json
  ok('200', r.code === 200, `得 ${r.code}`)
  ok('返回 ok', opened && opened.ok === true, opened && opened.error)
  if (opened && opened.ok) {
    const original = fs.readFileSync(SAMPLE, 'utf8')
    ok('原文逐字符一致', opened.text === original)
    ok('报出字节数', opened.bytes === original.length, `${opened.bytes} vs ${original.length}`)
    const metaCount = Object.keys(opened.constMeta || {}).length
    ok('列出常量 operand', metaCount > 0, `${metaCount} 个`)
    const missing = Object.values(opened.constMeta).filter((m) => !m.exists).length
    console.log(`  常量 operand ${metaCount} 个，本机缺失 ${missing} 个（手机路径，正常）`)
    ok('常量重映射生效', Object.values(opened.constMeta).some((m) => m.local.startsWith('F:/KAtorch/artifacts/npt/')),
      Object.values(opened.constMeta)[0]?.local)
  }
}

// ---- /analyze -------------------------------------------------------------
console.log('\n=== POST /analyze ===')
{
  const original = fs.readFileSync(SAMPLE, 'utf8')
  const r = await call('POST', '/analyze', { text: original, consts: {} })
  ok('200', r.code === 200, `得 ${r.code}`)
  ok('返回算子数', r.json && r.json.opCount === 284, r.json && String(r.json.opCount))
  ok('统计里有 opcode 分布', r.json && Array.isArray(r.json.stats.byOpcode) && r.json.stats.byOpcode.length > 3)
  const errs = (r.json.problems || []).filter((p) => p.level === 'error')
  ok('没有结构错误', errs.length === 0, errs.slice(0, 2).map((p) => p.text).join(' | '))
  ok('形状校验因缺常量降级为 info',
    (r.json.problems || []).some((p) => p.level === 'info'),
    '（本机没有 .bin，属预期）')

  // 故意破坏 id 连续性，必须被抓到
  const broken = original.replace('OPD 10 8 3 1 1 2048 0 0 o', 'OPD 110 8 3 1 1 2048 0 0 o')
  const r2 = await call('POST', '/analyze', { text: broken, consts: {} })
  ok('id 不连续被抓到',
    r2.json && r2.json.errorCount > 0 && r2.json.problems.some((p) => p.text.includes('id 不连续')),
    r2.json && r2.json.problems.filter((p) => p.level === 'error').map((p) => p.text).join(' | '))
}

// ---- /save ----------------------------------------------------------------
console.log('\n=== POST /save ===')
{
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'katorch-viz-'))
  const target = path.join(tmpDir, 'g.txt')
  const original = fs.readFileSync(SAMPLE, 'utf8')
  const r = await call('POST', '/save', { path: target, text: original })
  ok('200', r.code === 200, `得 ${r.code}`)
  ok('文件写出来了', fs.existsSync(target))
  ok('内容一致', fs.readFileSync(target, 'utf8') === original)
  ok('报了字节数', r.json && r.json.bytes === original.length)

  // 第二次写要留下备份
  const r2 = await call('POST', '/save', { path: target, text: original.replace('END\n', 'END\n') })
  const backups = fs.readdirSync(tmpDir).filter((n) => n.includes('.bak-'))
  ok('覆盖时留了 .bak', backups.length >= 1, backups.join(','))
  fs.rmSync(tmpDir, { recursive: true, force: true })
}

// ---- /browse --------------------------------------------------------------
console.log('\n=== GET /browse ===')
{
  const dir = path.dirname(SAMPLE)
  const r = await call('GET', `/browse?path=${encodeURIComponent(dir)}`)
  ok('200', r.code === 200, `得 ${r.code}`)
  ok('列出了 g.txt', r.json && (r.json.files || []).some((f) => f.name === 'g.txt'),
    r.json && JSON.stringify((r.json.files || []).map((f) => f.name)))
}

// ---- agent 工具 -----------------------------------------------------------
console.log('\n=== agent 工具 katorch_viz ===')
const tool = tools.find((x) => x.name === 'katorch_viz')
{
  const r = await tool.execute({ action: 'roundtrip', path: SAMPLE })
  ok('roundtrip 判定 PASS', r.verdict === 'PASS', JSON.stringify({ identical: r.identical, errs: r.structuralErrors }))
  ok('roundtrip 报 identical', r.identical === true)
  ok('roundtrip 报算子数', r.importedOps === 284, String(r.importedOps))

  const r2 = await tool.execute({ action: 'stats', path: SAMPLE })
  ok('stats 报 opCount', r2.opCount === 284, String(r2.opCount))
  ok('stats 报常量字节', typeof r2.constBytes === 'string' && r2.constBytes.length > 0, r2.constBytes)
  ok('stats 报 opcode 分布', Array.isArray(r2.byOpcode) && r2.byOpcode[0].name === 'RESHAPE',
    JSON.stringify(r2.byOpcode && r2.byOpcode.slice(0, 3)))

  const r3 = await tool.execute({ action: 'list', path: path.dirname(SAMPLE) })
  ok('list 列出目录', Array.isArray(r3.files) && r3.files.length >= 1)

  const r4 = await tool.execute({
    action: 'analyze',
    text: 'OPD 0 8 2 4 8 0 0 i\nOPD 1 8 2 4 8 0 0 o\nIO 1 0 1 1\nOP 19 1 0 1 1\nEND\n',
  })
  ok('analyze 小图通过', r4.errorCount === 0, r4.problems.filter((p) => p.level === 'error').map((p) => p.text).join(' | '))

  // id 不连续的图 export 时应当被重编号修好（这正是需求要的）
  const gapText = 'OPD 0 8 2 4 8 0 0 i\nOPD 5 8 2 4 8 0 0 o\nIO 1 0 1 5\nOP 19 1 0 1 5\nEND\n'
  const tmpGap = path.join(os.tmpdir(), `kv-gap-${Date.now()}.txt`)
  const rg = await tool.execute({ action: 'export', path: tmpGap, text: gapText })
  ok('id 不连续时 export 会重编号', rg.ok === true, JSON.stringify(rg).slice(0, 200))
  if (rg.ok) {
    const out = fs.readFileSync(tmpGap, 'utf8')
    ok('重编号后 id 连续',
      out.includes('OPD 0 ') && out.includes('OPD 1 ') && !out.includes('OPD 5 '), out.replace(/\n/g, ' | '))
    ok('OP 行的引用也跟着改', out.includes('OP 19 1 0 1 1'), out.split('\n').find((l) => l.startsWith('OP ')))
    fs.rmSync(tmpGap, { force: true })
  }

  // 引用根本不存在的 operand —— 这个重编号也修不好，必须拒绝写盘
  const badText = 'OPD 0 8 2 4 8 0 0 i\nOPD 1 8 2 4 8 0 0 o\nIO 1 0 1 1\nOP 19 1 99 1 1\nEND\n'
  const tmp2 = path.join(os.tmpdir(), `kv-export-${Date.now()}.txt`)
  const r5 = await tool.execute({ action: 'export', path: tmp2, text: badText })
  ok('export 拒绝写坏图', r5.ok === false && r5.refused === true, JSON.stringify(r5).slice(0, 200))
  ok('坏图没有被写到磁盘', !fs.existsSync(tmp2))

  // 正常 export
  const goodText = 'OPD 0 8 2 4 8 0 0 i\nOPD 1 8 2 4 8 0 0 o\nIO 1 0 1 1\nOP 19 1 0 1 1\nEND\n'
  const r6 = await tool.execute({ action: 'export', path: tmp2, text: goodText })
  ok('export 写成功', r6.ok === true, JSON.stringify(r6).slice(0, 200))
  if (r6.ok) {
    ok('写出的内容和输入一致', fs.readFileSync(tmp2, 'utf8') === goodText)
    fs.rmSync(tmp2, { force: true })
  }
}

console.log(`\n=== 结果：${pass} 通过 / ${failures.length} 失败 ===`)
if (failures.length) {
  for (const f of failures) console.log(`  !! ${f}`)
  process.exit(1)
}
