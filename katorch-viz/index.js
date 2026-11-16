/**
 * KAtorch 可视化编译器 —— Host 半。
 *
 * 职责边界（刻意做窄）：
 *   图的解析 / 编辑 / 形状推导 / 导出全在 core/（纯数据层，浏览器和 Node 都能跑）。
 *   Host 只做两件 Client 做不到的事：
 *     1. 读写磁盘（g.txt 与 .bin 常量）
 *     2. 起 HTTP 路由，把上面这些暴露给 Web UI，同时把同一组操作注册成 agent 工具
 *
 * 所以两边跑的是同一份 core 代码，不会出现「界面显示的和导出的不是一份」。
 */
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'

import { parseGtxt, checkStructure } from './core/parser.js'
import { serialize, normalizeIds, stats, statsByGroups, preview } from './core/export.js'
import { validateGraph } from './core/shape.js'
import { paramSlots, operandBytes, opName } from './core/spec.js'

export const name = 'katorch-viz'
export const inject = ['webServer', 'tools']

const API = '/katorch-viz'

const DEFAULTS = {
  /** 打开界面时预填的路径 */
  defaultPath: '',
  /** 最近一次打开的路径（运行时记忆） */
  pythonPath: 'python',
  /**
   * 常量路径重映射：图里写的是手机上的 /data/data/com.termux/...，
   * 本机若有一份拷贝，在这里指定 [前缀, 本地前缀] 的对应关系。
   */
  constRewrite: [
    ['/data/data/com.termux/files/home/npt/', 'F:/KAtorch/artifacts/npt/'],
  ],
  /** 读常量时单个文件的上限（形状参数都很小，权重不读） */
  constReadLimit: 65536,
}

/** 形状推导只关心这些位置的常量（axis / perm / shape / pad / stride / adj），都很大不了 */
const isShapeParam = (graph, op, index) => paramSlots(op.opcode).includes(index)

// --------------------------------------------------------------------------- //
// 小工具
// --------------------------------------------------------------------------- //

function readBody(req, limit = 32 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    req.on('data', (chunk) => {
      size += chunk.length
      if (size > limit) {
        reject(new Error(`请求体过大（>${Math.round(limit / 1024 / 1024)}MB）`))
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks)))
    req.on('error', reject)
  })
}

function sendJson(res, code, body) {
  const text = JSON.stringify(body)
  res.writeHead(code, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
  })
  res.end(text)
}

const fmtBytes = (n) => {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${n} B`
}

// --------------------------------------------------------------------------- //
// 插件
// --------------------------------------------------------------------------- //

export function apply(ctx, config) {
  const cfg = { ...DEFAULTS, ...(config || {}) }
  if (Array.isArray(cfg.constRewrite)) {
    cfg.constRewrite = cfg.constRewrite
      .filter((pair) => Array.isArray(pair) && pair.length === 2)
      .map(([from, to]) => [String(from), String(to)])
  } else {
    cfg.constRewrite = []
  }

  let lastPath = cfg.defaultPath || ''

  /** 把图里的手机路径映射到本机路径 */
  function resolveConstPath(p) {
    if (!p) return p
    for (const [from, to] of cfg.constRewrite) {
      if (from && p.startsWith(from)) return to + p.slice(from.length)
    }
    return p
  }

  /**
   * 读整份图：原文 + 形状参数常量。
   * @returns {Promise<{ok:true, path:string, text:string, consts:object, constMeta:object}>}
   */
  async function openFile(filePath) {
    const abs = path.resolve(String(filePath))
    const stat = await fsp.stat(abs).catch(() => null)
    if (!stat) throw new Error(`文件不存在：${abs}`)
    if (!stat.isFile()) throw new Error(`不是文件：${abs}`)
    const text = await fsp.readFile(abs, 'utf8')

    const { graph } = parseGtxt(text)

    // 需要读哪些常量：op 的参数位引用的 operand
    const wanted = new Set()
    for (const op of graph.ops) {
      const slots = paramSlots(op.opcode)
      if (!slots.length) continue
      for (const i of slots) {
        const oid = op.ins[i]
        if (oid != null) wanted.add(oid)
      }
    }

    const consts = {}
    const constMeta = {}
    for (const el of graph.operandById.values()) {
      if (el.tailKind !== 'w' || !el.tailPath) continue
      const local = resolveConstPath(el.tailPath)
      const entry = {
        oid: el.id,
        declared: el.tailPath,
        local,
        exists: false,
        size: null,
        off: el.tailOff,
        len: el.tailLen,
      }
      try {
        const st = await fsp.stat(local)
        entry.exists = true
        entry.size = st.size
        if (wanted.has(el.id) && (!el.tailLen || el.tailLen <= cfg.constReadLimit)) {
          const buf = await fsp.readFile(local)
          const slice = el.tailOff != null && el.tailLen != null
            ? buf.subarray(el.tailOff, el.tailOff + el.tailLen)
            : buf
          if (slice.length % 4 === 0 && slice.length <= cfg.constReadLimit) {
            const ints = new Array(slice.length / 4)
            for (let i = 0; i < ints.length; i += 1) ints[i] = slice.readInt32LE(i * 4)
            consts[el.id] = ints
          }
        }
      } catch {
        entry.exists = false
      }
      constMeta[el.id] = entry
    }

    lastPath = abs
    return {
      ok: true,
      path: abs,
      text,
      consts,
      constMeta,
      bytes: text.length,
      mtime: stat.mtimeMs,
    }
  }

  /** 保存前自检：解析 + 结构 + 形状 */
  function inspectText(text) {
    const { graph, warnings } = parseGtxt(text)
    const structural = checkStructure(graph)
    return { graph, warnings, structural }
  }

  /** 完整的验收信息（工具的 validate / export 都用它） */
  function analyze(text, consts) {
    const { graph, warnings } = parseGtxt(text)
    const structural = checkStructure(graph)
    const constMap = new Map()
    for (const [oid, ints] of Object.entries(consts || {})) {
      const buf = Buffer.alloc(ints.length * 4)
      ints.forEach((v, i) => buf.writeInt32LE(v | 0, i * 4))
      constMap.set(Number(oid), buf)
    }
    const shapeProblems = validateGraph(graph, constMap)
    const st = stats(graph)
    const errors = [
      ...structural.filter((p) => p.level === 'error'),
      ...shapeProblems.filter((p) => p.level === 'error'),
    ]
    return {
      graph,
      warnings,
      stats: st,
      problems: [...structural, ...shapeProblems],
      errorCount: errors.length,
      opCount: graph.ops.length,
    }
  }

  /** 写文件，默认留一份 .bak-<时间戳> */
  async function saveFile(filePath, text, { backup = true } = {}) {
    const abs = path.resolve(String(filePath))
    await fsp.mkdir(path.dirname(abs), { recursive: true })
    let backedUpTo = null
    const exists = await fsp.stat(abs).catch(() => null)
    if (backup && exists && exists.isFile()) {
      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
      backedUpTo = `${abs}.bak-${stamp}`
      await fsp.copyFile(abs, backedUpTo)
    }
    await fsp.writeFile(abs, text, 'utf8')
    lastPath = abs
    return { path: abs, bytes: Buffer.byteLength(text, 'utf8'), backedUpTo }
  }

  /** 写 .bin 常量（P1：改 RESHAPE 目标形状 / TRANSPOSE perm 这类） */
  async function writeConst(filePath, ints, { off = null, len = null, backup = true } = {}) {
    const abs = path.resolve(resolveConstPath(String(filePath)))
    const buf = Buffer.alloc(ints.length * 4)
    ints.forEach((v, i) => buf.writeInt32LE(v | 0, i * 4))

    if (off == null && len == null) {
      let backedUpTo = null
      if (backup) {
        const exists = await fsp.stat(abs).catch(() => null)
        if (exists) {
          const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
          backedUpTo = `${abs}.bak-${stamp}`
          await fsp.copyFile(abs, backedUpTo)
        }
      }
      await fsp.mkdir(path.dirname(abs), { recursive: true })
      await fsp.writeFile(abs, buf)
      return { path: abs, bytes: buf.length, backedUpTo }
    }

    // 打包文件的片段：只覆盖这一段，其余不动
    let base = Buffer.alloc(0)
    const exists = await fsp.stat(abs).catch(() => null)
    if (exists) base = await fsp.readFile(abs)
    const end = (off || 0) + (len != null ? len : buf.length)
    const out = Buffer.concat([base, Buffer.alloc(Math.max(0, end - base.length))])
    buf.copy(out, off || 0, 0, Math.min(buf.length, end - (off || 0)))
    await fsp.writeFile(abs, out)
    return { path: abs, bytes: buf.length, off, len, inPlace: true }
  }

  /** 列目录，找出里面的 g.txt 一类文件 */
  async function browse(dirPath) {
    const abs = path.resolve(String(dirPath || '.'))
    const entries = await fsp.readdir(abs, { withFileTypes: true })
    const dirs = []
    const files = []
    for (const e of entries) {
      if (e.name.startsWith('.')) continue
      if (e.isDirectory()) {
        dirs.push({ name: e.name, path: path.join(abs, e.name) })
      } else if (/\.txt$/i.test(e.name) || /g\.txt$/i.test(e.name)) {
        const st = await fsp.stat(path.join(abs, e.name)).catch(() => null)
        files.push({ name: e.name, path: path.join(abs, e.name), size: st ? st.size : null })
      }
    }
    dirs.sort((a, b) => a.name.localeCompare(b.name))
    files.sort((a, b) => a.name.localeCompare(b.name))
    return { path: abs, parent: path.dirname(abs), dirs, files }
  }

  /**
   * 调 host.c 编译（P2）。需要本机有编好的 host 可执行文件与厂商库，
   * 手机上（Termux）才有；Windows 这边一般没有，所以只做能力探测。
   */
  function probeCompiler() {
    const candidates = [
      cfg.hostBinary,
      'F:\\KAtorch\\编译器\\host',
      'F:\\KAtorch\\编译器\\host.exe',
    ].filter(Boolean)
    for (const c of candidates) {
      try {
        if (fs.statSync(c).isFile()) return { available: true, path: c }
      } catch { /* 继续找 */ }
    }
    return {
      available: false,
      hint: '本机没有 host 可执行文件。真机编译请在 Termux 里跑 tools/verify-on-phone.sh，'
        + '或在 cordis.patch.yml 里给本插件配 hostBinary 指向编好的 host。',
    }
  }

  // ------------------------------------------------------------------------- //
  // HTTP 路由
  // ------------------------------------------------------------------------- //

  async function handler(req, res) {
    const url = new URL(req.url || '/', 'http://localhost')
    const route = url.pathname.slice(API.length) || '/'
    try {
      if (req.method === 'GET' && (route === '/' || route === '/state')) {
        return sendJson(res, 200, {
          ok: true,
          config: { defaultPath: cfg.defaultPath, constRewrite: cfg.constRewrite },
          lastPath,
          compiler: probeCompiler(),
        })
      }

      if (req.method === 'POST' && route === '/open') {
        const body = JSON.parse((await readBody(req)).toString('utf8') || '{}')
        return sendJson(res, 200, await openFile(body.path))
      }

      if (req.method === 'GET' && route === '/browse') {
        const p = url.searchParams.get('path') || cfg.defaultPath || 'F:\\KAtorch'
        return sendJson(res, 200, { ok: true, ...(await browse(p)) })
      }

      // 纯文本自检（不落盘）：把当前编辑内容送进来，回结构与形状问题
      if (req.method === 'POST' && route === '/analyze') {
        const body = JSON.parse((await readBody(req)).toString('utf8') || '{}')
        const result = analyze(body.text, body.consts)
        return sendJson(res, 200, {
          ok: true,
          opCount: result.opCount,
          stats: result.stats,
          problems: result.problems,
          warnings: result.warnings,
          errorCount: result.errorCount,
        })
      }

      if (req.method === 'POST' && route === '/save') {
        const body = JSON.parse((await readBody(req)).toString('utf8') || '{}')
        if (typeof body.text !== 'string') return sendJson(res, 400, { ok: false, error: '缺 text' })
        const target = body.path || lastPath
        if (!target) return sendJson(res, 400, { ok: false, error: '缺 path' })
        const saved = await saveFile(target, body.text, { backup: body.backup !== false })
        return sendJson(res, 200, { ok: true, ...saved })
      }

      if (req.method === 'POST' && route === '/write-const') {
        const body = JSON.parse((await readBody(req)).toString('utf8') || '{}')
        if (!Array.isArray(body.path ? body.ints : null)) {
          return sendJson(res, 400, { ok: false, error: '需要 path 与 ints 数组' })
        }
        const written = await writeConst(body.path, body.ints, {
          off: body.off == null ? null : Number(body.off),
          len: body.len == null ? null : Number(body.len),
          backup: body.backup !== false,
        })
        return sendJson(res, 200, { ok: true, ...written })
      }

      if (req.method === 'POST' && route === '/compile') {
        const compiler = probeCompiler()
        if (!compiler.available) return sendJson(res, 501, { ok: false, error: compiler.hint })
        return sendJson(res, 501, {
          ok: false,
          error: '编译回调尚未接线（P2）。先在 Termux 里跑 tools/verify-on-phone.sh。',
        })
      }

      return sendJson(res, 404, { ok: false, error: `未知接口：${req.method} ${route}` })
    } catch (err) {
      return sendJson(res, 500, { ok: false, error: String(err && err.message ? err.message : err) })
    }
  }

  ctx.effect(() => ctx.webServer.register({
    kind: 'prefix',
    path: API,
    handler: (req, res) => { handler(req, res) },
  }), 'katorch-viz: http routes')

  // ------------------------------------------------------------------------- //
  // agent 工具 —— 与 UI 走同一套 Host 实现
  // ------------------------------------------------------------------------- //

  ctx.effect(() => ctx.tools.register({
    name: 'katorch_viz',
    description: [
      'KAtorch 可视化编译器的数据面：读写并校验 lowering 出来的 NPU 文本图 g.txt。',
      'action=open 读取 g.txt（返回原文 + 形状参数常量 + 统计）；action=analyze 校验一段图文本'
        + '（结构 + id 连续性 + 形状推导，对应 shape_infer.validate_txt）；action=roundtrip 做往返自检'
        + '（解析→导出→与原文逐字节比对）；action=export 把文本写回磁盘（自动留 .bak）；'
        + 'action=list 列目录找 g.txt；action=stats 只看统计。',
      '图格式：OPD 声明 operand / OP 算子 / IO 输入输出 / END。operand id 必须 0..N-1 连续，'
        + '否则 host.c 编译返回 -304。',
    ].join(' '),
    parameters: {
      type: 'object',
      properties: {
        action: {
          type: 'string',
          enum: ['open', 'analyze', 'roundtrip', 'export', 'list', 'stats'],
        },
        path: { type: 'string', description: '文件或目录的绝对路径（action=open/export/list/stats）' },
        text: { type: 'string', description: '图文本（action=analyze；缺省时用 path 读）' },
        consts: {
          type: 'object',
          description: 'action=analyze 用的形状参数常量表 {oid: [int32,...]}；不给则不推导',
          additionalProperties: true,
        },
      },
      required: ['action'],
      additionalProperties: false,
    },
    output: {
      schema: { type: 'object', additionalProperties: true },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
    },
    async execute(args) {
      const action = args && args.action

      if (action === 'list') {
        return await browse(args.path || cfg.defaultPath || 'F:\\KAtorch')
      }

      if (action === 'open' || action === 'stats') {
        if (!args.path) throw new Error('open/stats 需要 path')
        const opened = await openFile(args.path)
        const result = analyze(opened.text, opened.consts)
        const summary = {
          path: opened.path,
          bytes: opened.bytes,
          opCount: result.opCount,
          operandCount: result.stats.operandCount,
          constOperands: result.stats.constOperands,
          constBytes: fmtBytes(result.stats.constBytes),
          byOpcode: result.stats.byOpcode.slice(0, 20),
          errorCount: result.errorCount,
          problems: result.problems.filter((p) => p.level === 'error').slice(0, 20),
          warnings: result.warnings,
          constsRead: Object.keys(opened.consts).length,
          constsMissing: Object.values(opened.constMeta).filter((m) => !m.exists).length,
        }
        if (action === 'stats') return summary
        return {
          ...summary,
          text: opened.text,
          consts: opened.consts,
          constMeta: opened.constMeta,
        }
      }

      if (action === 'analyze') {
        const text = args.text != null ? String(args.text) : (args.path ? await fsp.readFile(path.resolve(args.path), 'utf8') : '')
        if (!text) throw new Error('analyze 需要 text 或 path')
        const result = analyze(text, args.consts)
        return {
          opCount: result.opCount,
          stats: result.stats,
          errorCount: result.errorCount,
          problems: result.problems,
          warnings: result.warnings,
        }
      }

      if (action === 'roundtrip') {
        if (!args.path) throw new Error('roundtrip 需要 path')
        const abs = path.resolve(args.path)
        const original = await fsp.readFile(abs, 'utf8')
        const { graph, warnings } = parseGtxt(original)
        const exported = serialize(graph)
        const identical = exported === original
        const structural = checkStructure(graph).filter((p) => p.level === 'error')
        return {
          path: abs,
          bytes: Buffer.byteLength(original, 'utf8'),
          importedOps: graph.ops.length,
          importedOperands: graph.operandById.size,
          exportedBytes: Buffer.byteLength(exported, 'utf8'),
          identical,
          firstDiff: identical ? null : firstDiffIndex(exported, original),
          structuralErrors: structural,
          warnings,
          verdict: identical && structural.length === 0 ? 'PASS' : 'FAIL',
        }
      }

      if (action === 'export') {
        if (!args.path) throw new Error('export 需要 path')
        const text = args.text != null ? String(args.text) : await fsp.readFile(path.resolve(args.path), 'utf8')
        // 导出前强制重编号 + 自检
        const { graph } = parseGtxt(text)
        normalizeIds(graph)
        const finalText = serialize(graph)
        const result = analyze(finalText)
        if (result.errorCount > 0) {
          return {
            ok: false,
            refused: true,
            reason: `自检发现 ${result.errorCount} 个错误，没有写盘`,
            problems: result.problems.filter((p) => p.level === 'error'),
          }
        }
        const saved = await saveFile(args.path, finalText)
        return { ok: true, ...saved, opCount: result.opCount }
      }

      throw new Error(`未知 action: ${action}`)
    },
  }), 'katorch-viz: agent tool')

  /** 提供给 client 的预览（P2 对比模式会用） */
  void preview
  void statsByGroups
  void operandBytes
  void opName
  void isShapeParam
}

function firstDiffIndex(a, b) {
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i += 1) if (a[i] !== b[i]) return i
  return n
}
