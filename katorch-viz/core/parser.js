/**
 * g.txt 解析器。
 *
 * 设计要点（往返一致性的来源）：
 *   每个元素都保留 raw（原始行文本，不含换行符）与 tokens。
 *   导出时只要这一行没被改过、且 id 没重编号，就原样吐出 raw。
 *   所以「导入 → 什么都不动 → 导出」天然逐字节一致，不依赖我猜对数字格式。
 *
 * host.c 的解析是宽容的（strtol 跳空白），这里也按宽容解析，
 * 但只支持 host.c 认识的那几种行；其它行原样保留，不拒绝。
 */

/** 按 \n 拆行，保留每行自己的行尾（含可能的 \r）。 */
export function splitLines(text) {
  const out = []
  let i = 0
  while (i < text.length) {
    const j = text.indexOf('\n', i)
    if (j < 0) {
      out.push({ body: text.slice(i), eol: '' })
      break
    }
    out.push({ body: text.slice(i, j), eol: '\n' })
    i = j + 1
  }
  return out
}

function tokenize(body) {
  const s = body.replace(/\r$/, '').trim()
  if (!s) return []
  return s.split(/\s+/)
}

const isInt = (s) => /^[+-]?\d+$/.test(s)

/**
 * 解析 OPD 行。
 * 返回 null 表示这行形状不对（调用方按 unknown 处理）。
 */
function parseOpd(tokens) {
  if (tokens.length < 6) return null
  const id = parseInt(tokens[1], 10)
  const type = parseInt(tokens[2], 10)
  const nd = parseInt(tokens[3], 10)
  if (!Number.isFinite(id) || !Number.isFinite(type) || !Number.isFinite(nd) || nd < 0) return null
  if (tokens.length < 4 + nd) return null
  const dims = []
  for (let i = 0; i < nd; i += 1) {
    const v = parseInt(tokens[4 + i], 10)
    if (!Number.isFinite(v)) return null
    dims.push(v)
  }
  const scaleTok = tokens[4 + nd]
  const zpTok = tokens[5 + nd]
  const scale = scaleTok == null ? 0 : Number(scaleTok)
  const zp = zpTok == null ? 0 : parseInt(zpTok, 10)

  const tailTokens = tokens.slice(6 + nd)
  let tailKind = 'none'
  let tailValue = null
  let tailPath = null
  let tailOff = null
  let tailLen = null

  const head = tailTokens[0]
  if (head === 'a') {
    tailKind = 'a'
    tailValue = tailTokens.length > 1 ? parseInt(tailTokens[1], 10) : 0
    if (!Number.isFinite(tailValue)) tailValue = 0
  } else if (head === 'w') {
    tailKind = 'w'
    // w <path>            整个文件就是常量
    // w <path> <off> <len> 从打包文件里取一段
    tailPath = tailTokens.length > 1 ? tailTokens[1] : ''
    if (tailTokens.length >= 4 && isInt(tailTokens[2]) && isInt(tailTokens[3])) {
      tailOff = parseInt(tailTokens[2], 10)
      tailLen = parseInt(tailTokens[3], 10)
    }
  } else if (head === 'i') {
    tailKind = 'i'
  } else if (head === 'o') {
    tailKind = 'o'
  }

  return {
    id, type, nd, dims,
    scale: Number.isFinite(scale) ? scale : 0,
    zp: Number.isFinite(zp) ? zp : 0,
    tailKind, tailValue, tailPath, tailOff, tailLen,
    scaleRaw: scaleTok == null ? '0' : scaleTok,
    zpRaw: zpTok == null ? '0' : zpTok,
  }
}

function parseOp(tokens) {
  if (tokens.length < 4) return null
  const opcode = parseInt(tokens[1], 10)
  const nIn = parseInt(tokens[2], 10)
  if (!Number.isFinite(opcode) || !Number.isFinite(nIn) || nIn < 0 || nIn > 64) return null
  if (tokens.length < 3 + nIn + 1) return null
  const ins = []
  for (let i = 0; i < nIn; i += 1) ins.push(parseInt(tokens[3 + i], 10))
  const nOut = parseInt(tokens[3 + nIn], 10)
  if (!Number.isFinite(nOut) || nOut < 0 || nOut > 64) return null
  if (tokens.length < 4 + nIn + nOut) return null
  const outs = []
  for (let i = 0; i < nOut; i += 1) outs.push(parseInt(tokens[4 + nIn + i], 10))
  if (ins.some((v) => !Number.isFinite(v)) || outs.some((v) => !Number.isFinite(v))) return null
  return { opcode, ins, outs }
}

function parseIo(tokens) {
  if (tokens.length < 3) return null
  const nIn = parseInt(tokens[1], 10)
  if (!Number.isFinite(nIn) || nIn < 0 || nIn > 256) return null
  if (tokens.length < 2 + nIn + 1) return null
  const ins = []
  for (let i = 0; i < nIn; i += 1) ins.push(parseInt(tokens[2 + i], 10))
  const nOut = parseInt(tokens[2 + nIn], 10)
  if (!Number.isFinite(nOut) || nOut < 0 || nOut > 256) return null
  if (tokens.length < 3 + nIn + nOut) return null
  const outs = []
  for (let i = 0; i < nOut; i += 1) outs.push(parseInt(tokens[3 + nIn + i], 10))
  if (ins.some((v) => !Number.isFinite(v)) || outs.some((v) => !Number.isFinite(v))) return null
  return { ins, outs }
}

function parseQch(tokens) {
  // QCH <opid> <channelDim> <count> <scales_f32_path>
  if (tokens.length < 5) return null
  const opid = parseInt(tokens[1], 10)
  const channelDim = parseInt(tokens[2], 10)
  const count = parseInt(tokens[3], 10)
  if (!Number.isFinite(opid) || !Number.isFinite(channelDim) || !Number.isFinite(count)) return null
  // path 之后可能还有内容，host.c 取到行尾
  return { opid, channelDim, count, path: tokens.slice(4).join(' ') }
}

/**
 * 解析整份 g.txt。
 * @param {string} text
 * @returns {{graph: object, warnings: string[]}}
 */
export function parseGtxt(text) {
  const warnings = []
  const elements = []
  const operandById = new Map()
  const ops = []
  let io = null
  let end = null

  for (const { body, eol } of splitLines(text)) {
    const tokens = tokenize(body)
    if (tokens.length === 0) {
      elements.push({ kind: 'raw', raw: body, eol })
      continue
    }
    const head = tokens[0]

    if (head === 'OPD') {
      const p = parseOpd(tokens)
      if (!p) {
        warnings.push(`无法解析的 OPD 行：${body.trim()}`)
        elements.push({ kind: 'unknown', raw: body, eol, tokens })
        continue
      }
      const el = {
        kind: 'operand', raw: body, eol, tokens,
        ...p,
        dirty: false,
        // 运行时补充
        opIndex: -1,        // 由哪个 op 产出（-1 = 图输入或纯常量）
        consumers: [],      // 被哪些 op 消费
      }
      elements.push(el)
      if (operandById.has(p.id)) {
        warnings.push(`operand id ${p.id} 重复出现；后者覆盖前者`)
      }
      operandById.set(p.id, el)
      continue
    }

    if (head === 'OP') {
      const p = parseOp(tokens)
      if (!p) {
        warnings.push(`无法解析的 OP 行：${body.trim()}`)
        elements.push({ kind: 'unknown', raw: body, eol, tokens })
        continue
      }
      const el = { kind: 'op', raw: body, eol, tokens, ...p, dirty: false, index: ops.length }
      elements.push(el)
      ops.push(el)
      continue
    }

    if (head === 'IO') {
      const p = parseIo(tokens)
      if (!p) {
        warnings.push(`无法解析的 IO 行：${body.trim()}`)
        elements.push({ kind: 'unknown', raw: body, eol, tokens })
        continue
      }
      const el = { kind: 'io', raw: body, eol, tokens, ...p, dirty: false }
      elements.push(el)
      if (io) warnings.push('出现了多条 IO 行；host.c 只认最后一条')
      io = el
      continue
    }

    if (head === 'QCH') {
      const p = parseQch(tokens)
      if (!p) {
        warnings.push(`无法解析的 QCH 行：${body.trim()}`)
        elements.push({ kind: 'unknown', raw: body, eol, tokens })
        continue
      }
      elements.push({ kind: 'qch', raw: body, eol, tokens, ...p, dirty: false })
      continue
    }

    if (head === 'END') {
      elements.push({ kind: 'end', raw: body, eol, tokens })
      end = true
      continue
    }

    if (head.startsWith('#')) {
      elements.push({ kind: 'raw', raw: body, eol })
      continue
    }

    warnings.push(`未知行类型「${head}」：${body.trim()}`)
    elements.push({ kind: 'unknown', raw: body, eol, tokens })
  }

  const graph = {
    elements,
    operandById,
    ops,
    io,
    hasEnd: !!end,
    newline: detectNewline(text),
  }

  linkGraph(graph)

  // 未改动前，整份文本应该能原样重建
  if (!graph.hasEnd) warnings.push('缺少 END 行')
  if (!io) warnings.push('缺少 IO 行 —— host.c 会以 -107 拒绝这份图')

  return { graph, warnings }
}

function detectNewline(text) {
  const i = text.indexOf('\n')
  if (i > 0 && text[i - 1] === '\r') return '\r\n'
  return '\n'
}

/**
 * 建立 operand -> 生产者 / 消费者的双向索引。编辑后要重新调用。
 */
export function linkGraph(graph) {
  for (const el of graph.operandById.values()) {
    el.opIndex = -1
    el.consumers = []
  }
  graph.ops.forEach((op, index) => {
    for (const oid of op.outs) {
      const el = graph.operandById.get(oid)
      if (el) el.opIndex = index
    }
    for (const oid of op.ins) {
      const el = graph.operandById.get(oid)
      if (el) el.consumers.push(index)
    }
  })
  return graph
}

/**
 * 跑一遍 host.c 会做的结构性检查（不需要 NPU 库）。
 * -304 那一类错误（id 不连续）就是这里抓不到、只能靠导出重编号保证的。
 * @returns {Array<{level:'error'|'warn', text:string, element?:object}>}
 */
export function checkStructure(graph) {
  const problems = []

  // 1. operand id 必须 0..N-1、连续、且按出现顺序递增
  let expect = 0
  for (const el of graph.elements) {
    if (el.kind !== 'operand') continue
    if (el.id !== expect) {
      problems.push({
        level: 'error',
        text: `operand id 不连续：第 ${expect + 1} 个 OPD 写的是 ${el.id}，应为 ${expect} —— host.c 的 M_addOp 按出现顺序编号，`
          + `M_setVal 用文件里的 id，不一致会返回 -304`,
        element: el,
      })
      expect = el.id
    }
    expect += 1
  }

  // 2. OP 引用的 operand 必须存在
  for (const op of graph.ops) {
    for (const oid of op.ins) {
      if (!graph.operandById.has(oid)) {
        problems.push({ level: 'error', text: `${opLabel(op)} 引用了不存在的输入 operand ${oid}`, element: op })
      }
    }
    for (const oid of op.outs) {
      if (!graph.operandById.has(oid)) {
        problems.push({ level: 'error', text: `${opLabel(op)} 引用了不存在的输出 operand ${oid}`, element: op })
      }
    }
    for (const oid of op.outs) {
      const el = graph.operandById.get(oid)
      if (el && el.opIndex !== op.index) {
        problems.push({
          level: 'error',
          text: `operand ${oid} 被多个算子写（${opLabel(graph.ops[el.opIndex] || op)} 与 ${opLabel(op)}）`,
          element: op,
        })
      }
    }
  }

  // 3. IO 引用的 operand 必须存在
  if (graph.io) {
    for (const oid of graph.io.ins) {
      if (!graph.operandById.has(oid)) {
        problems.push({ level: 'error', text: `IO 的输入 ${oid} 没有 OPD 声明`, element: graph.io })
      }
    }
    for (const oid of graph.io.outs) {
      if (!graph.operandById.has(oid)) {
        problems.push({ level: 'error', text: `IO 的输出 ${oid} 没有 OPD 声明`, element: graph.io })
      }
    }
  }

  // 4. nd 与 dims 长度一致
  for (const el of graph.operandById.values()) {
    if (el.dims.length !== el.nd) {
      problems.push({ level: 'error', text: `operand ${el.id} 的 nd=${el.nd} 但给了 ${el.dims.length} 个维度`, element: el })
    }
  }

  return problems
}

function opLabel(op) {
  if (!op) return 'OP ?'
  return `#${op.index} OP ${op.opcode}`
}
