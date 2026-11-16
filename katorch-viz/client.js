/**
 * KAtorch 可视化编译器 —— Client 半（自动生成，别手改）。
 * 源码在 core/ 与 ui/，用 node build.mjs 重新生成。
 * 生成时间：2026-11-16T10:53:07.005Z
 */
window.__ModuleLoader__.load({
  id: '@local/katorch-viz',
  factory(require) {
    const React = require('react')

  // ===== core/spec.js =====
/**
 * KAtorch 图格式规格表 —— 唯一一份，parser / shape / export 都从这里取。
 *
 * 权威来源：F:\KAtorch\编译器\host.c 的 build_from_file() 与 shape_infer.py。
 * 改这里之前先回去看那两个文件，不要凭印象改。
 */

/** opcode -> 名称（shape_infer.py:NAMES + 算子清单里出现过的） */
const OP_NAMES = {
  0: 'ADD',
  1: 'AVGPOOL',
  2: 'CONCAT',
  3: 'CONV_2D',
  6: 'DEQUANTIZE',
  9: 'FC',
  14: 'LOGISTIC',
  17: 'MAXPOOL',
  18: 'MUL',
  19: 'RELU',
  20: 'RELU1',
  21: 'RELU6',
  22: 'RESHAPE',
  25: 'SOFTMAX',
  28: 'TANH',
  30: 'DIV',
  31: 'MEAN',
  36: 'SUB',
  37: 'TRANSPOSE',
  38: 'ABS',
  45: 'CAST',
  49: 'EXP',
  60: 'LOG',
  64: 'LOG_SOFTMAX',
  65: 'MAXIMUM',
  66: 'MINIMUM',
  67: 'NEG',
  72: 'QUANTIZE',
  77: 'REDUCE_MAX',
  80: 'REDUCE_SUM',
  83: 'RSQRT',
  85: 'SIN',
  88: 'SQRT',
  102: 'BATCH_MATMUL',
}

function opName(code) {
  return OP_NAMES[code] || `OP_${code}`
}

/** 分类：形状推导用（shape_infer.py 原样搬过来） */
const UNARY = new Set([6, 14, 19, 20, 21, 25, 28, 38, 45, 49, 60, 64, 67, 72, 83, 85, 88])
const BINARY = new Set([0, 18, 30, 36, 65, 66])
const REDUCE = new Set([31, 77, 80])

const ADD = 0
const CONCAT = 2
const CONV = 3
const FC = 9
const RESHAPE = 22
const TRANSPOSE = 37
const SOFTMAX = 25
const BATCH_MATMUL = 102

/** operand type 码 -> 元信息 */
const TYPES = {
  0: { name: 'FLOAT32', bytes: 4, tensor: false },
  1: { name: 'INT32', bytes: 4, tensor: false },
  3: { name: 'TENSOR_FLOAT32', bytes: 4, tensor: true },
  4: { name: 'TENSOR_INT32', bytes: 4, tensor: true },
  5: { name: 'QUANT8_ASYMM', bytes: 1, tensor: true },
  7: { name: 'TENSOR_QUANT16_SYMM', bytes: 2, tensor: true },
  8: { name: 'TENSOR_FLOAT16', bytes: 2, tensor: true },
  11: { name: 'QUANT8_SYMM_PER_CHANNEL', bytes: 1, tensor: true },
}

/** 短名（节点上显示用） */
const TYPE_SHORT = {
  0: 'f32', 1: 'i32', 3: 'f32', 4: 'i32', 5: 'u8', 7: 'i16', 8: 'f16', 11: 'u8pc',
}

function typeBytes(type) {
  const t = TYPES[type]
  return t ? t.bytes : 0
}

function typeName(type) {
  const t = TYPES[type]
  return t ? t.name : `TYPE_${type}`
}

function typeShort(type) {
  return TYPE_SHORT[type] || `t${type}`
}

/** 一个 operand 声明占多少字节（nelem * elem_size） */
function operandBytes(el) {
  if (!el || !el.dims) return 0
  let n = 1
  for (const d of el.dims) n *= d
  return n * typeBytes(el.type)
}

/**
 * 常量（a / w）的 int32 值。形状推导时 axis / perm / shape 都从这儿来。
 * data 是外部读进来的 Buffer/Uint8Array（可选）。
 */
function constInts(el, data) {
  if (!el) return null
  if (el.tailKind === 'a') return [el.tailValue]
  if (el.tailKind !== 'w') return null
  if (!data || data.length === 0 || data.length % 4 !== 0) return null
  const n = data.length / 4
  const out = new Array(n)
  const view = new DataView(data.buffer ? data.buffer : data, data.byteOffset || 0, data.byteLength)
  for (let i = 0; i < n; i += 1) out[i] = view.getInt32(i * 4, true)
  return out
}

/**
 * 该 opcode 的「参数 operand 下标」——这些位置的常量才需要读文件。
 * 返回 [] 表示不需要常量数据。
 */
function paramSlots(opcode) {
  if (REDUCE.has(opcode)) return [1]
  if (opcode === TRANSPOSE) return [1]
  if (opcode === RESHAPE) return [1]
  if (opcode === CONCAT) return [2]
  if (opcode === BATCH_MATMUL) return [2, 3]
  if (opcode === CONV) return [3, 4, 5, 6, 7, 8]
  return []
}

/** 每个算子期望的输入个数（算子表里的「输入个数」，缺省 null = 可变） */
const ARITY = {
  0: 3, 2: 3, 9: 4, 14: 1, 18: 3, 19: 1, 22: 2, 25: 2, 28: 1, 30: 3,
  31: 3, 36: 3, 37: 2, 38: 1, 45: 1, 49: 1, 60: 1, 64: 1, 65: 3, 66: 3,
  67: 1, 77: 3, 80: 3, 83: 1, 85: 1, 88: 1, 102: 4,
}

/**
 * 节点上的已知坑标注（需求 §10）。
 * 返回 [{level:'warn'|'error', text}]
 */
function pitfalls(opEl, graph) {
  const out = []
  const code = opEl.opcode
  const ins = opEl.ins || []
  const get = (i) => (graph && graph.operandById ? graph.operandById.get(ins[i]) : undefined)

  const want = ARITY[code]
  if (want != null && ins.length !== want) {
    out.push({
      level: 'warn',
      text: `${opName(code)} 期望 ${want} 个输入，实际 ${ins.length} 个`,
    })
  }

  if (code === FC) {
    const act = get(3)
    // 依据：需求 §10「FC 的 act 写成 type 1 → 静默通过，结果全错」
    // 真实图（n1/g.txt 的 `OP 9 4 14 15 16 9 1 17`）里 act 是 OPD 9 = type 4 的 `a 0`。
    // 所以正解是 type 4（TENSOR_INT32），写成 type 1（INT32 标量）会静默出错。
    if (act && act.type === 1) {
      out.push({
        level: 'error',
        text: 'FC 的 act 是 type 1(INT32 标量)，应为 type 4(TENSOR_INT32) —— 编译会静默通过，结果全错',
      })
    }
  }

  if (code === CONV) {
    for (let i = 3; i <= 8; i += 1) {
      const p = get(i)
      if (!p) continue
      if (p.type !== 1) {
        out.push({ level: 'error', text: `CONV_2D 的 pad/stride(输入${i}) 必须是 type 1，现在是 type ${p.type}` })
      } else if (p.nd !== 0) {
        out.push({ level: 'error', text: `CONV_2D 的 pad/stride(输入${i}) 必须 nd=0，现在是 nd=${p.nd}` })
      }
    }
  }

  if (code === SOFTMAX) {
    const x = get(0)
    if (x && x.type === 8) {
      out.push({
        level: 'warn',
        text: '原生 SOFTMAX 吃 fp16 时输出恒为均匀分布 1/N —— 只有 int16 正确',
      })
    }
  }

  return out
}


  // ===== core/parser.js =====
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
function splitLines(text) {
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
function parseGtxt(text) {
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
function linkGraph(graph) {
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
function checkStructure(graph) {
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


  // ===== core/shape.js =====
/**
 * 形状推导 —— 从 F:\KAtorch\编译器\shape_infer.py 的第 62~156 行逐条移植。
 *
 * 为什么这层必须有：adapter 不校验 operand 声明的形状和算子实际输出。
 * 声明错了它不报错，只按你声明的尺寸分配 buffer，算子只写实际那部分，
 * 剩下的字节是未初始化内存 → 数值全错但 rc 全 0。
 *
 * 与 Python 版的差别：
 *   Python 版 validate_txt 从 .bin 文件读 axis/perm/shape；
 *   这里数据由调用方预先读好塞进 constData（浏览器读不了磁盘，由 Host 半读）。
 */



class ShapeError extends Error {}

/**
 * 单个算子的形状推导。
 * @param {number} opcode
 * @param {number[][]} ins 输入张量形状
 * @param {{axis?:number, perm?:number[], shape?:number[], adj_x?:number, adj_y?:number, keepdim?:boolean, pad?:number[], stride?:number[]}} kw
 * @returns {number[][]} 输出形状列表
 */
function infer(opcode, ins, kw = {}) {
  const shapes = ins.map((s) => Array.from(s))

  if (opcode === FC) {
    // (x, w, b, act)；x 最后一维 == w 最后一维
    const x = shapes[0]
    const w = shapes[1]
    if (!x || !w) throw new ShapeError('FC 需要至少 2 个输入')
    if (x[x.length - 1] !== w[w.length - 1]) {
      throw new ShapeError(`FC: x[-1]=${x[x.length - 1]} != w[-1]=${w[w.length - 1]}`)
    }
    return [[...x.slice(0, -1), w[0]]]
  }

  if (BINARY.has(opcode)) {
    if (shapes.length < 2) throw new ShapeError('二元算子需要至少 2 个输入')
    return [broadcast(shapes[0], shapes[1])]
  }

  if (REDUCE.has(opcode)) {
    // (x, axis, act)
    const axisRaw = kw.axis
    if (axisRaw == null) throw new ShapeError(`${opName(opcode)} 需要 axis`)
    const x = shapes[0]
    if (!x) throw new ShapeError(`${opName(opcode)} 需要输入`)
    const nd = x.length
    let axis = axisRaw
    if (axis < 0) axis += nd
    if (!(axis >= 0 && axis < nd)) throw new ShapeError(`${opName(opcode)}: axis=${axisRaw} 越界 (nd=${nd})`)
    const out = [...x]
    if (kw.keepdim !== false) out[axis] = 1
    else out.splice(axis, 1)
    return [out]
  }

  if (UNARY.has(opcode)) {
    if (!shapes[0]) throw new ShapeError(`${opName(opcode)} 需要一个输入`)
    return [shapes[0]]
  }

  if (opcode === TRANSPOSE) {
    const perm = kw.perm
    if (!perm) throw new ShapeError('TRANSPOSE 需要 perm')
    const x = shapes[0]
    if (!x) throw new ShapeError('TRANSPOSE 需要输入')
    if (perm.length !== x.length) {
      throw new ShapeError(`TRANSPOSE: perm 长度 ${perm.length} != 张量维度 ${x.length}`)
    }
    for (const p of perm) {
      if (!(p >= 0 && p < x.length)) throw new ShapeError(`TRANSPOSE: perm 元素 ${p} 越界 (nd=${x.length})`)
    }
    return [perm.map((p) => x[p])]
  }

  if (opcode === RESHAPE) {
    const shape = kw.shape
    if (!shape) throw new ShapeError('RESHAPE 需要 shape')
    const x = shapes[0]
    if (!x) throw new ShapeError('RESHAPE 需要输入')
    const t0 = x.reduce((a, b) => a * b, 1)
    const t1 = shape.reduce((a, b) => a * b, 1)
    if (t0 !== t1) {
      throw new ShapeError(`RESHAPE: [${x}] 有 ${t0} 个元素, [${shape}] 有 ${t1} 个`)
    }
    return [Array.from(shape)]
  }

  if (opcode === CONCAT) {
    const axis = kw.axis
    if (axis == null) throw new ShapeError('CONCAT 需要 axis')
    const a = shapes[0]
    const b = shapes[1]
    if (!a || !b) throw new ShapeError('CONCAT 需要两个张量')
    if (a.length !== b.length) throw new ShapeError(`CONCAT: 维度不同 [${a}] vs [${b}]`)
    const ax = axis < 0 ? axis + a.length : axis
    if (!(ax >= 0 && ax < a.length)) throw new ShapeError(`CONCAT: axis=${axis} 越界 (nd=${a.length})`)
    const out = [...a]
    out[ax] += b[ax]
    return [out]
  }

  if (opcode === BATCH_MATMUL) {
    const ax = Number(kw.adj_x || 0)
    const ay = Number(kw.adj_y || 0)
    const a = shapes[0]
    const b = shapes[1]
    if (!a || !b) throw new ShapeError('BATCH_MATMUL 需要两个输入')
    if (a.length !== 3 || b.length !== 3) {
      throw new ShapeError(`BATCH_MATMUL 需要 3D 输入，收到 [${a}] / [${b}]`)
    }
    const B = a[0]
    const M = ax ? a[2] : a[1]
    const K = ax ? a[1] : a[2]
    const K2 = ay ? b[2] : b[1]
    const N = ay ? b[1] : b[2]
    if (K !== K2) throw new ShapeError(`BATCH_MATMUL: 内积维 ${K} vs ${K2}`)
    if (b[0] !== B && b[0] !== 1) throw new ShapeError(`BATCH_MATMUL: batch ${B} vs ${b[0]}`)
    return [[B, M, N]]
  }

  if (opcode === CONV) {
    // x: NHWC [N,H,W,C]，w: OHWC [O,KH,KW,C]
    const x = shapes[0]
    const w = shapes[1]
    if (!x || !w) throw new ShapeError('CONV_2D 需要 x 和 w')
    if (x.length !== 4 || w.length !== 4) throw new ShapeError(`CONV_2D 需要 4D：x=[${x}] w=[${w}]`)
    if (x[3] !== w[3]) throw new ShapeError(`CONV_2D: 输入通道 ${x[3]} != 权重通道 ${w[3]}`)
    const pad = kw.pad || [1, 1, 1, 1]   // (l, r, t, b)
    const stride = kw.stride || [1, 1]   // (w, h)
    const H = x[1]
    const W = x[2]
    const OH = Math.floor((H + pad[2] + pad[3] - w[1]) / stride[1]) + 1
    const OW = Math.floor((W + pad[0] + pad[1] - w[2]) / stride[0]) + 1
    if (OH <= 0 || OW <= 0) throw new ShapeError(`CONV_2D: 输出尺寸非法 OH=${OH} OW=${OW}`)
    return [[x[0], OH, OW, w[0]]]
  }

  throw new ShapeError(`未实现的算子 ${opName(opcode)} (${opcode})`)
}

/** numpy 风格广播 */
function broadcast(a, b) {
  let x = Array.from(a)
  let y = Array.from(b)
  const n = Math.max(x.length, y.length)
  x = [...Array(n - x.length).fill(1), ...x]
  y = [...Array(n - y.length).fill(1), ...y]
  const out = []
  for (let i = 0; i < n; i += 1) {
    if (x[i] === y[i] || x[i] === 1 || y[i] === 1) out.push(Math.max(x[i], y[i]))
    else throw new ShapeError(`broadcast 不兼容: [${a}] vs [${b}] (维 ${i}: ${x[i]} vs ${y[i]})`)
  }
  return out
}

/**
 * 从一个 op 元素取出推导所需的 kw（从它的参数 operand 的常量值）。
 * 常量读不到时返回 {kw, missing:[...]}，调用方据此降级。
 */
function opKw(opEl, graph, constData) {
  const kw = {}
  const missing = []
  const dataOf = (i) => {
    const oid = opEl.ins[i]
    if (oid == null) return null
    const el = graph.operandById.get(oid)
    if (!el) return null
    return constInts(el, constData ? constData.get(oid) : null)
  }

  const code = opEl.opcode
  if (REDUCE.has(code) || CONCAT === code) {
    const i = code === CONCAT ? 2 : 1
    const v = dataOf(i)
    if (v && v.length) kw.axis = v[0]
    else missing.push(`输入${i} 的 axis 常量`)
  }
  if (code === TRANSPOSE) {
    const v = dataOf(1)
    if (v) kw.perm = v
    else missing.push('输入1 的 perm 常量')
  }
  if (code === RESHAPE) {
    const v = dataOf(1)
    if (v) kw.shape = v
    else missing.push('输入1 的 shape 常量')
  }
  if (code === BATCH_MATMUL) {
    const vx = dataOf(2)
    const vy = dataOf(3)
    kw.adj_x = vx && vx.length ? vx[0] : 0
    kw.adj_y = vy && vy.length ? vy[0] : 0
    if (!vx) missing.push('输入2 的 adj_x 常量')
    if (!vy) missing.push('输入3 的 adj_y 常量')
  }
  if (code === CONV) {
    const pad = []
    const stride = []
    for (let i = 3; i <= 6; i += 1) {
      const v = dataOf(i)
      pad.push(v && v.length ? v[0] : 1)
      if (!v) missing.push(`输入${i} 的 pad 常量`)
    }
    for (let i = 7; i <= 8; i += 1) {
      const v = dataOf(i)
      stride.push(v && v.length ? v[0] : 1)
      if (!v) missing.push(`输入${i} 的 stride 常量`)
    }
    kw.pad = pad
    kw.stride = stride
  }
  return { kw, missing }
}

/**
 * 推导一个 op 的输出形状。
 * @returns {{ok:true, shapes:number[][], missing:string[]} | {ok:false, error:string, missing:string[]}}
 */
function inferOp(opEl, graph, constData) {
  const { kw, missing } = opKw(opEl, graph, constData)
  const inShapes = []
  for (const oid of opEl.ins) {
    const el = graph.operandById.get(oid)
    if (!el) return { ok: false, error: `输入 operand ${oid} 不存在`, missing }
    inShapes.push(el.dims)
  }
  try {
    const shapes = infer(opEl.opcode, inShapes, kw)
    return { ok: true, shapes, missing }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err), missing }
  }
}

/**
 * 全图校验：每个 op 的声明输出形状是否等于推导形状。
 * 对应 shape_infer.py 的 validate_txt()。
 * @returns {Array<{level:string, text:string, opIndex?:number, oid?:number}>}
 */
function validateGraph(graph, constData) {
  const problems = []
  for (const op of graph.ops) {
    const res = inferOp(op, graph, constData)
    if (!res.ok) {
      // 常量缺失时只是「推不出来」，不是错误；真正的不一致才报
      if (res.missing.length) {
        problems.push({
          level: 'info',
          text: `${opName(op.opcode)}(#${op.index}) 跳过形状校验：${res.missing.join('、')} 读不到`,
          opIndex: op.index,
        })
      } else {
        problems.push({
          level: 'error',
          text: `${opName(op.opcode)}(#${op.index}) 形状推导失败：${res.error}`,
          opIndex: op.index,
        })
      }
      continue
    }
    if (res.shapes.length !== op.outs.length) {
      problems.push({
        level: 'error',
        text: `${opName(op.opcode)}(#${op.index}) 推导出 ${res.shapes.length} 个输出，图里声明了 ${op.outs.length} 个`,
        opIndex: op.index,
      })
      continue
    }
    op.outs.forEach((oid, i) => {
      const el = graph.operandById.get(oid)
      if (!el) return
      const declared = el.dims
      const want = res.shapes[i]
      if (declared.length !== want.length || declared.some((v, k) => v !== want[k])) {
        problems.push({
          level: 'error',
          text: `${opName(op.opcode)}(#${op.index}) 输出 ${oid} 声明 [${declared}] != 推导 [${want}] —— adapter 会静默出错`,
          opIndex: op.index,
          oid,
        })
      }
    })
  }
  return problems
}

/**
 * 在一条链上把上游形状变更传播到下游。
 * 每一轮重推导，直到无变化或达到轮数上限。
 * @returns {{changed:number[], problems:object[]}}
 */
function propagateShapes(graph, constData, maxRounds = 12) {
  const changed = []
  let problems = []
  for (let round = 0; round < maxRounds; round += 1) {
    let dirty = false
    for (const op of graph.ops) {
      const res = inferOp(op, graph, constData)
      if (!res.ok || res.missing.length) continue
      if (res.shapes.length !== op.outs.length) continue
      op.outs.forEach((oid, i) => {
        const el = graph.operandById.get(oid)
        if (!el) return
        const want = res.shapes[i]
        if (el.dims.length === want.length && el.dims.every((v, k) => v === want[k])) return
        el.dims = Array.from(want)
        el.nd = want.length
        el.dirty = true
        dirty = true
        changed.push(oid)
      })
    }
    if (!dirty) break
    if (round === maxRounds - 1) {
      problems.push({ level: 'warn', text: `形状传播 ${maxRounds} 轮仍未收敛，图里可能有环` })
    }
  }
  problems = problems.concat(validateGraph(graph, constData).filter((p) => p.level === 'error'))
  return { changed, problems }
}


  // ===== core/export.js =====
/**
 * 导出：重编号 + 序列化。
 *
 * 硬约束（host.c）：operand id 必须是 0..N-1、连续、且按 OPD 出现顺序递增。
 * 原因：M_addOp 按 OPD 出现顺序分配内部索引，M_setVal 用文件里写的 id。
 * 两者不一致 → 编译返回 -304。
 *
 * 另一条硬约束（往返一致性）：没动过的行必须原样吐出 raw。
 * 所以这里只重建 dirty 的行。
 */



/**
 * 重新连续编号（幂等）。
 * 已经连续时什么都不改，因此不影响逐字节往返。
 * @returns {{changed:boolean, map:Map<number,number>}}
 */
function normalizeIds(graph) {
  const map = new Map()
  let next = 0
  let changed = false

  for (const el of graph.elements) {
    if (el.kind !== 'operand') continue
    map.set(el.id, next)
    if (next !== el.id) changed = true
    next += 1
  }

  if (!changed) {
    // 仍然要保证 operandById 与 elements 一致（删除操作后可能残留）
    const fresh = new Map()
    for (const el of graph.elements) {
      if (el.kind === 'operand') fresh.set(el.id, el)
    }
    graph.operandById = fresh
    return { changed: false, map }
  }

  for (const el of graph.elements) {
    if (el.kind === 'operand') {
      const nid = map.get(el.id)
      if (nid !== el.id) {
        el.id = nid
        el.dirty = true
      }
    } else if (el.kind === 'op' || el.kind === 'io') {
      let touched = false
      el.ins = el.ins.map((oid) => {
        const n = map.has(oid) ? map.get(oid) : oid
        if (n !== oid) touched = true
        return n
      })
      el.outs = el.outs.map((oid) => {
        const n = map.has(oid) ? map.get(oid) : oid
        if (n !== oid) touched = true
        return n
      })
      if (touched) el.dirty = true
    } else if (el.kind === 'qch') {
      if (map.has(el.opid) && map.get(el.opid) !== el.opid) {
        el.opid = map.get(el.opid)
        el.dirty = true
      }
    }
  }

  const fresh = new Map()
  for (const el of graph.elements) {
    if (el.kind === 'operand') fresh.set(el.id, el)
  }
  graph.operandById = fresh
  return { changed: true, map }
}

// --------------------------------------------------------------------------- //
// 单行格式化
// --------------------------------------------------------------------------- //

function formatOperand(el) {
  const parts = ['OPD', String(el.id), String(el.type), String(el.nd)]
  for (const d of el.dims) parts.push(String(d))
  parts.push(el.scaleRaw != null ? el.scaleRaw : String(el.scale))
  parts.push(el.zpRaw != null ? el.zpRaw : String(el.zp))
  if (el.tailKind === 'a') {
    parts.push('a', String(el.tailValue == null ? 0 : el.tailValue))
  } else if (el.tailKind === 'w') {
    parts.push('w', String(el.tailPath || ''))
    if (el.tailOff != null && el.tailLen != null) parts.push(String(el.tailOff), String(el.tailLen))
  } else if (el.tailKind === 'i') {
    parts.push('i')
  } else if (el.tailKind === 'o') {
    parts.push('o')
  }
  return parts.join(' ')
}

function formatOp(el) {
  const parts = ['OP', String(el.opcode), String(el.ins.length)]
  for (const oid of el.ins) parts.push(String(oid))
  parts.push(String(el.outs.length))
  for (const oid of el.outs) parts.push(String(oid))
  return parts.join(' ')
}

function formatIo(el) {
  const parts = ['IO', String(el.ins.length)]
  for (const oid of el.ins) parts.push(String(oid))
  parts.push(String(el.outs.length))
  for (const oid of el.outs) parts.push(String(oid))
  return parts.join(' ')
}

function formatQch(el) {
  return ['QCH', String(el.opid), String(el.channelDim), String(el.count), el.path].join(' ')
}

function formatElement(el) {
  switch (el.kind) {
    case 'operand': return el.dirty ? formatOperand(el) : el.raw
    case 'op': return el.dirty ? formatOp(el) : el.raw
    case 'io': return el.dirty ? formatIo(el) : el.raw
    case 'qch': return el.dirty ? formatQch(el) : el.raw
    case 'end': return el.raw
    case 'unknown':
    case 'raw':
    default: return el.raw
  }
}

/**
 * 序列化整张图。
 * @param {object} graph
 * @param {{renumber?:boolean}} opts renumber 默认 true（导出就是要交付给 host.c）
 */
function serialize(graph, opts = {}) {
  const renumber = opts.renumber !== false
  if (renumber) normalizeIds(graph)

  const out = []
  for (const el of graph.elements) {
    out.push(formatElement(el))
    // eol 由 parse 带来（'\n' / '\r\n' / 末行 ''）；新建元素已带 graph.newline
    out.push(el.eol == null ? graph.newline : el.eol)
  }
  return out.join('')
}

/** 只生成文本，不改图（用于预览） */
function preview(graph) {
  return graph.elements.map((el) => formatElement(el)).join('\n') + '\n'
}

// --------------------------------------------------------------------------- //
// 图的编辑操作
// --------------------------------------------------------------------------- //

/** 删除一个算子及其独占的输出 operand */
function deleteOp(graph, opIndex) {
  const op = graph.ops[opIndex]
  if (!op) throw new Error(`没有第 ${opIndex} 个算子`)

  const ioIds = new Set()
  if (graph.io) {
    for (const oid of graph.io.ins) ioIds.add(oid)
    for (const oid of graph.io.outs) ioIds.add(oid)
  }

  const removeIds = new Set()
  for (const oid of op.outs) {
    const el = graph.operandById.get(oid)
    if (!el) continue
    const usedElsewhere = el.consumers.some((i) => i !== opIndex)
    if (!usedElsewhere && !ioIds.has(oid)) removeIds.add(oid)
  }

  graph.elements = graph.elements.filter((el) => {
    if (el === op) return false
    if (el.kind === 'operand' && removeIds.has(el.id)) return false
    return true
  })
  graph.ops = graph.ops.filter((_, i) => i !== opIndex)
  graph.ops.forEach((o, i) => { o.index = i })
  for (const el of graph.elements) {
    if (el.kind === 'operand') el.consumers = el.consumers.filter((i) => i !== opIndex).map((i) => (i > opIndex ? i - 1 : i))
  }
  // 只有被重编号波及的行才会变 dirty —— normalizeIds 负责标
  normalizeIds(graph)
  relink(graph)
  return { removedOperands: [...removeIds] }
}

/** 复制一个算子：在原位置后面插一份，输出用新 operand */
function duplicateOp(graph, opIndex) {
  const op = graph.ops[opIndex]
  if (!op) throw new Error(`没有第 ${opIndex} 个算子`)

  // 新输出 operand 先用临时 id（远大于现有最大值），normalizeIds 会把它压回连续区间
  const PHANTOM = 1000000
  const newEls = op.outs.map((oid, i) => {
    const src = graph.operandById.get(oid)
    if (!src) throw new Error(`输出 operand ${oid} 不存在`)
    return {
      ...src,
      id: PHANTOM + i,
      dims: [...src.dims],
      consumers: [],
      opIndex: -1,
      dirty: true,
      raw: '',
      eol: graph.newline,
    }
  })

  const newOp = {
    ...op,
    ins: [...op.ins],
    outs: newEls.map((e) => e.id),
    dirty: true,
    raw: '',
    eol: graph.newline,
    index: -1,
  }

  // OPD 必须排在所有 OP 之前（host.c 流式解析：OP 引用时 operand 必须已 addOperand）
  const insertOpAt = graph.elements.indexOf(op)
  graph.elements.splice(insertOpAt + 1, 0, newOp)
  const lastOperandIdx = graph.elements.reduce((acc, el, i) => (el.kind === 'operand' ? i : acc), 0)
  graph.elements.splice(lastOperandIdx + 1, 0, ...newEls)

  graph.ops.splice(opIndex + 1, 0, newOp)
  graph.ops.forEach((o, i) => { o.index = i })
  normalizeIds(graph)
  relink(graph)
  return { opIndex: opIndex + 1 }
}

/** 改 operand 的形状 */
function setOperandDims(graph, oid, dims) {
  const el = graph.operandById.get(oid)
  if (!el) throw new Error(`operand ${oid} 不存在`)
  el.dims = dims.map((d) => Math.max(0, Math.round(Number(d)) || 0))
  el.nd = el.dims.length
  el.dirty = true
  return el
}

/** 改一个算子的输入（连接编辑 P0 不做，但底层留着） */
function setOpInputs(graph, opIndex, ins) {
  const op = graph.ops[opIndex]
  if (!op) throw new Error(`没有第 ${opIndex} 个算子`)
  op.ins = ins.map((v) => Math.round(Number(v)))
  op.dirty = true
  relink(graph)
  return op
}

/** 重建 op.index 与 operand 双向索引 */
function relink(graph) {
  graph.ops.forEach((o, i) => { o.index = i })
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

// --------------------------------------------------------------------------- //
// 统计
// --------------------------------------------------------------------------- //

/**
 * @returns {{
 *   opCount:number, operandCount:number, constOperands:number,
 *   constBytes:number, operandBytes:number,
 *   byOpcode: Array<{code:number,name:string,count:number}>,
 *   opcodes: Map<number,number>,
 *   constPaths: number,
 * }}
 */
function stats(graph) {
  const byCode = new Map()
  for (const op of graph.ops) byCode.set(op.opcode, (byCode.get(op.opcode) || 0) + 1)

  let constBytes = 0
  let constOperands = 0
  const constPaths = new Set()
  let totalOperandBytes = 0

  for (const el of graph.operandById.values()) {
    const bytes = operandBytes(el)
    totalOperandBytes += bytes
    if (el.tailKind === 'w') {
      constOperands += 1
      constPaths.add(el.tailPath)
      // 短格式：整个文件；长格式：off/len 指明长度
      constBytes += el.tailLen != null ? el.tailLen : bytes
    } else if (el.tailKind === 'a') {
      constOperands += 1
      constBytes += 4
    }
  }

  const byOpcode = [...byCode.entries()]
    .map(([code, count]) => ({ code, name: opName(code), count }))
    .sort((a, b) => b.count - a.count || a.code - b.code)

  return {
    opCount: graph.ops.length,
    operandCount: graph.operandById.size,
    constOperands,
    constBytes,
    operandBytes: totalOperandBytes,
    byOpcode,
    uniqueConstPaths: constPaths.size,
  }
}

/** 每个分组（层）的算子数与常量大小 */
function statsByGroups(graph, groups) {
  return groups.map((g) => {
    let opCount = 0
    let bytes = 0
    for (let i = g.start; i < g.end; i += 1) {
      const op = graph.ops[i]
      if (!op) continue
      opCount += 1
      for (const oid of op.outs) {
        const el = graph.operandById.get(oid)
        if (el) bytes += operandBytes(el)
      }
      for (const oid of op.ins) {
        const el = graph.operandById.get(oid)
        if (el && el.tailKind === 'w' && el.consumers.length === 1) {
          bytes += el.tailLen != null ? el.tailLen : operandBytes(el)
        }
      }
    }
    return { ...g, opCount, bytes }
  })
}

/** 节点副标题：`256×2048 · f16` */
function shapeLabel(el) {
  if (!el) return '?'
  const dims = el.dims.length ? el.dims.join('×') : 'scalar'
  return `${dims} · ${typeShort(el.type)}`
}


  // ===== ui/i18n.js =====
/**
 * 文案。走宿主的 locale 服务，这样跟随界面语言。
 * key 很少，够用就行。
 */
const zh = {
  'nav': 'KAtorch 图',
  'title': 'KAtorch 可视化编译器',
  'path.ph': 'g.txt 的绝对路径，例如 F:\\KAtorch\\PC编译\\wsl-sysroot\\n1\\g.txt',
  'open': '打开',
  'browse': '浏览',
  'export': '导出',
  'save': '写回',
  'check': '自检',
  'group': '分组',
  'group.auto': '自动按层',
  'group.size': '按算子数',
  'group.none': '不分组',
  'group.size.ph': '每层算子数',
  'expandAll': '全部展开',
  'collapseAll': '全部折叠',
  'outline': '大纲',
  'inspect': '检查器',
  'stats': '统计',
  'problems': '问题',
  'noProblems': '自检通过',
  'dirty': '有未导出的改动',
  'empty.title': '还没有打开图',
  'empty.hint': '填 g.txt 路径后点「打开」，或从大纲里选一个组。',
  'menu.expand': '展开这一组',
  'menu.collapse': '折叠这一组',
  'menu.duplicate': '复制算子',
  'menu.delete': '删除算子',
  'menu.focus': '定位到画布中央',
  'needOpen': '先打开一个 g.txt',
  'exported': '已写回',
  'checked': '自检完成',
  'constsWritten': '常量文件已写回',
}

const en = {
  'nav': 'KAtorch Graph',
  'title': 'KAtorch Visual Compiler',
  'path.ph': 'absolute path to g.txt',
  'open': 'Open',
  'browse': 'Browse',
  'export': 'Export',
  'save': 'Write back',
  'check': 'Validate',
  'group': 'Groups',
  'group.auto': 'Auto by layer',
  'group.size': 'By op count',
  'group.none': 'No groups',
  'group.size.ph': 'ops per layer',
  'expandAll': 'Expand all',
  'collapseAll': 'Collapse all',
  'outline': 'Outline',
  'inspect': 'Inspector',
  'stats': 'Stats',
  'problems': 'Problems',
  'noProblems': 'Validation passed',
  'dirty': 'Unsaved changes',
  'empty.title': 'No graph loaded',
  'empty.hint': 'Enter a g.txt path and press Open.',
  'menu.expand': 'Expand group',
  'menu.collapse': 'Collapse group',
  'menu.duplicate': 'Duplicate op',
  'menu.delete': 'Delete op',
  'menu.focus': 'Center on canvas',
  'needOpen': 'Open a g.txt first',
  'exported': 'Written back',
  'checked': 'Validation done',
  'constsWritten': 'Const files written',
}

const DICTS = { zh, en }

let translate = null

function setTranslator(fn) {
  translate = fn
}

function t(key) {
  if (!translate) return zh[key] || key
  const got = translate(key)
  return got == null || got === key ? (zh[key] || key) : got
}


  // ===== ui/theme.js =====
/**
 * 样式 —— 只用宿主主题 token（--dsw-alias-*），不写死颜色。
 * 类名统一 kv- 前缀，避免和宿主/其它插件打架。
 */
const CSS = `
.kv-root {
  display: flex; flex-direction: column; height: 100%; min-height: 0;
  background: var(--dsw-alias-bg-base); color: var(--dsw-alias-label-primary);
  font-size: 13px; position: relative;
}
.kv-toolbar {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 9px 12px; border-bottom: 1px solid var(--dsw-alias-border-l1);
  background: var(--dsw-alias-bg-layer-1); flex: none;
}
.kv-title { font-weight: 600; font-size: 13.5px; margin-right: 4px; white-space: nowrap; }
.kv-path {
  flex: 1 1 260px; min-width: 180px; box-sizing: border-box;
  background: var(--dsw-alias-bg-base); color: var(--dsw-alias-label-primary);
  border: 1px solid var(--dsw-alias-border-l1); border-radius: 7px;
  padding: 6px 9px; font-size: 12px; font-family: ui-monospace, Consolas, monospace;
}
.kv-path:focus { outline: none; border-color: var(--dsw-alias-brand-primary); }
.kv-btn {
  appearance: none; cursor: pointer; border-radius: 7px; padding: 6px 11px; font-size: 12px;
  border: 1px solid var(--dsw-alias-border-l1); background: var(--dsw-alias-bg-base);
  color: var(--dsw-alias-label-primary); white-space: nowrap;
}
.kv-btn:hover:not(:disabled) { background: var(--dsw-alias-bg-layer-2); border-color: var(--dsw-alias-border-l2); }
.kv-btn:disabled { opacity: .45; cursor: default; }
.kv-btn-primary { background: var(--dsw-alias-brand-primary); border-color: var(--dsw-alias-brand-primary); color: #fff; font-weight: 500; }
.kv-btn-primary:hover:not(:disabled) { filter: brightness(1.08); }
.kv-btn-danger { color: var(--dsw-alias-state-error-primary); border-color: var(--dsw-alias-state-error-primary); }
.kv-select {
  background: var(--dsw-alias-bg-base); color: var(--dsw-alias-label-primary);
  border: 1px solid var(--dsw-alias-border-l1); border-radius: 7px; padding: 6px 8px; font-size: 12px;
}
.kv-spacer { flex: 1 1 auto; }

.kv-body { display: flex; flex: 1 1 auto; min-height: 0; }
.kv-outline {
  width: 216px; flex: none; display: flex; flex-direction: column; min-height: 0;
  border-right: 1px solid var(--dsw-alias-border-l1); background: var(--dsw-alias-bg-layer-1);
}
.kv-side {
  width: 316px; flex: none; display: flex; flex-direction: column; min-height: 0;
  border-left: 1px solid var(--dsw-alias-border-l1); background: var(--dsw-alias-bg-layer-1);
}
.kv-center { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; min-height: 0; }

.kv-panel-head {
  display: flex; align-items: center; gap: 6px; padding: 8px 11px; flex: none;
  border-bottom: 1px solid var(--dsw-alias-border-l1);
  font-size: 12px; font-weight: 600; color: var(--dsw-alias-label-secondary);
}
.kv-panel-body { overflow: auto; flex: 1 1 auto; min-height: 0; padding: 8px; }
.kv-panel-body.pad0 { padding: 0; }

.kv-group-row {
  display: flex; align-items: center; gap: 6px; padding: 6px 8px; border-radius: 7px;
  cursor: pointer; font-size: 12px; line-height: 1.35;
}
.kv-group-row:hover { background: var(--dsw-alias-bg-layer-2); }
.kv-group-row.active { background: var(--dsw-alias-bg-layer-2); box-shadow: inset 2px 0 0 var(--dsw-alias-brand-primary); }
.kv-group-name { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kv-group-meta { font-size: 10.5px; color: var(--dsw-alias-label-secondary); white-space: nowrap; }
.kv-caret { width: 12px; text-align: center; color: var(--dsw-alias-label-secondary); flex: none; }

.kv-canvas-wrap { position: relative; flex: 1 1 auto; min-height: 0; overflow: hidden; }
.kv-canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.kv-zoom-chip {
  position: absolute; right: 10px; bottom: 10px; font-size: 11px; padding: 3px 8px; border-radius: 999px;
  background: var(--dsw-alias-bg-layer-2); color: var(--dsw-alias-label-secondary);
  border: 1px solid var(--dsw-alias-border-l1); pointer-events: none;
}
.kv-canvas-hint {
  position: absolute; left: 10px; bottom: 10px; font-size: 11px; color: var(--dsw-alias-label-secondary);
  pointer-events: none; opacity: .75;
}

.kv-kv { display: grid; grid-template-columns: 88px 1fr; gap: 3px 8px; font-size: 12px; padding: 2px 2px 8px; }
.kv-kv dt { color: var(--dsw-alias-label-secondary); }
.kv-kv dd { margin: 0; word-break: break-all; font-family: ui-monospace, Consolas, monospace; font-size: 11.5px; }

.kv-sub { font-size: 11px; font-weight: 600; color: var(--dsw-alias-label-secondary); margin: 10px 0 5px; letter-spacing: .02em; }
.kv-sub:first-child { margin-top: 2px; }

.kv-tensor {
  display: flex; align-items: center; gap: 6px; padding: 4px 7px; border-radius: 6px;
  border: 1px solid var(--dsw-alias-border-l1); background: var(--dsw-alias-bg-base);
  font-size: 11.5px; margin-bottom: 4px; cursor: pointer;
}
.kv-tensor:hover { border-color: var(--dsw-alias-border-l2); }
.kv-tid { font-family: ui-monospace, Consolas, monospace; color: var(--dsw-alias-label-secondary); flex: none; min-width: 26px; }
.kv-tshape { flex: 1 1 auto; font-family: ui-monospace, Consolas, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kv-ttype { color: var(--dsw-alias-label-secondary); flex: none; font-size: 10.5px; }
.kv-tconst { color: var(--dsw-alias-brand-primary); flex: none; font-size: 10px; }

.kv-badge {
  display: inline-block; font-size: 10.5px; padding: 1px 6px; border-radius: 999px; flex: none;
  border: 1px solid var(--dsw-alias-border-l1); color: var(--dsw-alias-label-secondary);
}
.kv-badge.err { color: var(--dsw-alias-state-error-primary); border-color: var(--dsw-alias-state-error-primary); }
.kv-badge.warn { color: var(--dsw-alias-state-warn-primary); border-color: var(--dsw-alias-state-warn-primary); }
.kv-badge.ok { color: var(--dsw-alias-state-success-primary); border-color: var(--dsw-alias-state-success-primary); }

.kv-note {
  font-size: 11.5px; line-height: 1.5; padding: 7px 9px; border-radius: 7px; margin: 6px 0;
  border: 1px solid var(--dsw-alias-border-l1); color: var(--dsw-alias-label-secondary);
  background: var(--dsw-alias-bg-base);
}
.kv-note.err {
  border-color: var(--dsw-alias-state-error-primary); color: var(--dsw-alias-state-error-primary);
  background: color-mix(in srgb, var(--dsw-alias-state-error-primary) 8%, transparent);
}
.kv-note.warn {
  border-color: var(--dsw-alias-state-warn-primary); color: var(--dsw-alias-state-warn-primary);
  background: color-mix(in srgb, var(--dsw-alias-state-warn-primary) 8%, transparent);
}
.kv-note.ok {
  border-color: var(--dsw-alias-state-success-primary); color: var(--dsw-alias-state-success-primary);
  background: color-mix(in srgb, var(--dsw-alias-state-success-primary) 8%, transparent);
}

.kv-input {
  box-sizing: border-box; background: var(--dsw-alias-bg-base); color: var(--dsw-alias-label-primary);
  border: 1px solid var(--dsw-alias-border-l1); border-radius: 6px; padding: 4px 7px;
  font-size: 11.5px; font-family: ui-monospace, Consolas, monospace; width: 100%;
}
.kv-input:focus { outline: none; border-color: var(--dsw-alias-brand-primary); }

.kv-dims { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }
.kv-dims .kv-input { width: 58px; }

.kv-foot {
  flex: none; max-height: 168px; display: flex; flex-direction: column;
  border-top: 1px solid var(--dsw-alias-border-l1); background: var(--dsw-alias-bg-layer-1);
}
.kv-foot.collapsed { max-height: 33px; }
.kv-problem {
  display: flex; gap: 7px; align-items: flex-start; padding: 4px 10px; font-size: 11.5px;
  line-height: 1.45; cursor: pointer; border-radius: 5px;
}
.kv-problem:hover { background: var(--dsw-alias-bg-layer-2); }
.kv-problem .lv { flex: none; width: 14px; text-align: center; font-weight: 700; }
.kv-problem.err .lv { color: var(--dsw-alias-state-error-primary); }
.kv-problem.warn .lv { color: var(--dsw-alias-state-warn-primary); }
.kv-problem.info .lv { color: var(--dsw-alias-label-secondary); }

.kv-tabs { display: flex; gap: 2px; padding: 0 8px; border-bottom: 1px solid var(--dsw-alias-border-l1); flex: none; }
.kv-tab {
  appearance: none; background: none; border: none; cursor: pointer; padding: 7px 9px; font-size: 12px;
  color: var(--dsw-alias-label-secondary); border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.kv-tab.active { color: var(--dsw-alias-label-primary); border-bottom-color: var(--dsw-alias-brand-primary); }

.kv-bar { display: flex; align-items: center; gap: 6px; font-size: 11.5px; margin-bottom: 3px; }
.kv-bar-name { width: 96px; flex: none; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kv-bar-track { flex: 1 1 auto; height: 7px; border-radius: 999px; background: var(--dsw-alias-bg-layer-2); overflow: hidden; }
.kv-bar-fill { height: 100%; background: var(--dsw-alias-brand-primary); border-radius: 999px; }
.kv-bar-num { width: 34px; text-align: right; flex: none; color: var(--dsw-alias-label-secondary); }

.kv-menu {
  position: fixed; z-index: 60; min-width: 152px; padding: 4px;
  background: var(--dsw-alias-bg-layer-2); border: 1px solid var(--dsw-alias-border-l1);
  border-radius: 9px; box-shadow: 0 8px 28px rgba(0,0,0,.28);
}
.kv-menu button {
  display: block; width: 100%; text-align: left; appearance: none; background: none; border: none;
  color: var(--dsw-alias-label-primary); font-size: 12px; padding: 6px 9px; border-radius: 6px; cursor: pointer;
}
.kv-menu button:hover:not(:disabled) { background: var(--dsw-alias-bg-base); }
.kv-menu button:disabled { opacity: .4; cursor: default; }
.kv-menu .sep { height: 1px; background: var(--dsw-alias-border-l1); margin: 4px 2px; }

.kv-empty { padding: 22px 14px; text-align: center; color: var(--dsw-alias-label-secondary); font-size: 12px; line-height: 1.6; }

.kv-toast {
  position: absolute; left: 50%; top: 58px; transform: translateX(-50%); z-index: 50;
  padding: 7px 14px; border-radius: 9px; font-size: 12px; max-width: 78%;
  background: var(--dsw-alias-bg-layer-2); border: 1px solid var(--dsw-alias-border-l1);
  box-shadow: 0 8px 26px rgba(0,0,0,.24);
}
.kv-toast.err { border-color: var(--dsw-alias-state-error-primary); color: var(--dsw-alias-state-error-primary); }
.kv-toast.ok { border-color: var(--dsw-alias-state-success-primary); color: var(--dsw-alias-state-success-primary); }

.kv-dirty { color: var(--dsw-alias-state-warn-primary); font-size: 11.5px; white-space: nowrap; }
.kv-mono { font-family: ui-monospace, Consolas, monospace; }
.kv-muted { color: var(--dsw-alias-label-secondary); }
`


  // ===== ui/layout.js =====
/**
 * 布局计算 —— 纯函数，不碰 DOM，可以被单测。
 *
 * 两套视图：
 *   折叠视图：每个分组（层）是一个组节点，纵向排列
 *   展开视图：组内算子按「数据流深度」分列，列内按文件顺序堆叠
 *
 * 需求 §7.2 要的「1 万节点流畅」靠两件事：折叠（28 层变成 28 个节点）+ 视口裁剪。
 * 折叠后画布上只有几十个节点，虚拟化基本不需要；展开一层也才几百个。
 */

const NODE_W = 150
const NODE_H = 46
const GROUP_W = 236
const GROUP_H = 58
const COL_GAP = 60
const ROW_GAP = 14
const GROUP_GAP = 26
const GROUP_HEAD = 34

/**
 * 自动检测重复周期（DiT 每层的算子序列是一样的）。
 * @param {number[]} codes opcode 序列
 * @param {number} minRate 匹配率下限
 * @returns {{period:number, rate:number}|null}
 */
function detectPeriod(codes, minRate = 0.9) {
  const n = codes.length
  if (n < 16) return null
  let best = null
  const maxPeriod = Math.floor(n / 2)
  for (let p = 4; p <= maxPeriod; p += 1) {
    let match = 0
    const total = n - p
    if (total <= 0) break
    for (let i = 0; i < total; i += 1) if (codes[i] === codes[i + p]) match += 1
    const rate = match / total
    if (rate >= minRate && (!best || rate > best.rate + 1e-9)) {
      best = { period: p, rate }
      // 更小的周期已经足够好就不再找了（最小周期更贴近「一层」）
      break
    }
  }
  return best
}

/**
 * 生成分组。
 * @param {object} graph
 * @param {{mode:'period'|'size'|'none', size?:number}} opts
 * @returns {Array<{id:number, start:number, end:number, label:string, kind:string}>}
 */
function makeGroups(graph, opts = {}) {
  const n = graph.ops.length
  const mode = opts.mode || 'period'
  const groups = []

  const push = (start, end, kind) => {
    if (end <= start) return
    groups.push({ id: groups.length, start, end, kind, label: '' })
  }

  if (n === 0) return groups

  if (mode === 'none') {
    push(0, n, 'whole')
  } else if (mode === 'size') {
    const size = Math.max(1, Math.floor(opts.size || 284))
    for (let i = 0; i < n; i += size) push(i, Math.min(n, i + size), 'size')
  } else {
    const codes = graph.ops.map((op) => op.opcode)
    const det = detectPeriod(codes)
    if (det && det.period < n) {
      for (let i = 0; i < n; i += det.period) push(i, Math.min(n, i + det.period), 'period')
      groups.forEach((g) => { g.rate = det.rate })
    } else {
      push(0, n, 'whole')
    }
  }

  // label：尽可能用「层」的说法
  groups.forEach((g, i) => {
    if (groups.length === 1) g.label = `全部 ${g.end - g.start} 个算子`
    else g.label = `第 ${i + 1} 组 · ${g.end - g.start} 个算子`
  })

  return groups
}

/**
 * 组内算子的数据流深度（列号）。
 * ops 本身是拓扑序，所以一遍扫就够。
 */
function computeDepths(graph, start, end) {
  const producer = new Map()
  for (let i = start; i < end; i += 1) {
    for (const oid of graph.ops[i].outs) producer.set(oid, i)
  }
  const depth = new Map()
  for (let i = start; i < end; i += 1) {
    let d = 0
    for (const oid of graph.ops[i].ins) {
      const p = producer.get(oid)
      if (p != null && p < i) d = Math.max(d, (depth.get(p) || 0) + 1)
    }
    depth.set(i, d)
  }
  return depth
}

/**
 * 算整张画的布局。
 * @param {object} graph
 * @param {Array} groups makeGroups 的结果
 * @param {Set<number>} expanded 展开的组 id 集合
 * @returns {{nodes:Array, edges:Array, nodeByOp:Map, width:number, height:number}}
 */
function buildLayout(graph, groups, expanded) {
  const nodes = []
  const nodeByOp = new Map()
  let y = 0
  let maxX = 0

  for (const g of groups) {
    const isOpen = expanded.has(g.id)
    if (!isOpen) {
      const node = {
        id: `g${g.id}`,
        kind: 'group',
        group: g,
        x: 0,
        y,
        w: GROUP_W,
        h: GROUP_H,
      }
      nodes.push(node)
      for (let i = g.start; i < g.end; i += 1) nodeByOp.set(i, node)
      y += GROUP_H + GROUP_GAP
      maxX = Math.max(maxX, GROUP_W)
      continue
    }

    const depth = computeDepths(graph, g.start, g.end)
    const byDepth = new Map()
    for (let i = g.start; i < g.end; i += 1) {
      const d = depth.get(i) || 0
      if (!byDepth.has(d)) byDepth.set(d, [])
      byDepth.get(d).push(i)
    }

    let maxRows = 0
    for (const arr of byDepth.values()) maxRows = Math.max(maxRows, arr.length)
    const innerH = maxRows * (NODE_H + ROW_GAP) - ROW_GAP
    const groupH = Math.max(NODE_H + GROUP_HEAD + 12, innerH + GROUP_HEAD + 16)

    // 组框节点（不参与边，只画背景和标题）
    const frame = {
      id: `gf${g.id}`,
      kind: 'frame',
      group: g,
      x: -18,
      y,
      w: 0,
      h: groupH,
    }
    nodes.push(frame)

    const top = y + GROUP_HEAD
    for (const [d, arr] of [...byDepth.entries()].sort((a, b) => a[0] - b[0])) {
      arr.forEach((opIndex, row) => {
        const node = {
          id: `o${opIndex}`,
          kind: 'op',
          opIndex,
          group: g,
          x: d * (NODE_W + COL_GAP),
          y: top + row * (NODE_H + ROW_GAP),
          w: NODE_W,
          h: NODE_H,
        }
        nodes.push(node)
        nodeByOp.set(opIndex, node)
        maxX = Math.max(maxX, node.x + NODE_W)
      })
    }
    frame.w = Math.max(GROUP_W, maxX + 18 + 18)
    y += groupH + GROUP_GAP
    maxX = 0
  }

  const edges = buildEdges(graph, nodeByOp)

  return {
    nodes,
    edges,
    nodeByOp,
    width: Math.max(400, ...nodes.map((n) => n.x + n.w)) + 80,
    height: Math.max(200, y + 40),
  }
}

/**
 * 连线。折叠时同一对组节点之间只留一条，附上条数（28 层折叠后不会变成毛线球）。
 */
function buildEdges(graph, nodeByOp) {
  const seen = new Map()
  const edges = []

  for (const op of graph.ops) {
    const dst = nodeByOp.get(op.index)
    if (!dst) continue
    for (const oid of op.ins) {
      const el = graph.operandById.get(oid)
      if (!el || el.opIndex < 0) continue          // 图输入 / 常量：没有生产节点
      const src = nodeByOp.get(el.opIndex)
      if (!src || src === dst) continue
      const key = `${src.id}>${dst.id}`
      const hit = seen.get(key)
      if (hit) { hit.count += 1; continue }
      const edge = { id: key, src, dst, count: 1 }
      seen.set(key, edge)
      edges.push(edge)
    }
  }
  return edges
}

/** 命中测试（逻辑坐标） */
function hitTest(layout, wx, wy) {
  for (let i = layout.nodes.length - 1; i >= 0; i -= 1) {
    const n = layout.nodes[i]
    if (n.kind === 'frame') continue
    if (wx >= n.x && wx <= n.x + n.w && wy >= n.y && wy <= n.y + n.h) return n
  }
  return null
}

/** 让某个节点进入视野 */
function centerOn(layout, node) {
  return {
    x: node.x + node.w / 2,
    y: node.y + node.h / 2,
  }
}


  // ===== ui/canvas.js =====
/**
 * 节点画布 —— 纯 Canvas 2D，不依赖任何图形库。
 *
 * 为什么不用 React Flow / litegraph：折叠后画布上只有几十个节点，
 * 展开一层也就几百个，Canvas 直接画足够快，而且省掉一层依赖和一套样式打架。
 * 需求 §7.2 要的「1 万节点」靠 折叠 + 视口裁剪 达成。
 */





/** 从宿主主题里取色，避免写死颜色 */
function readStyle(el) {
  const cs = getComputedStyle(el || document.body)
  const v = (name, fallback) => {
    const got = cs.getPropertyValue(name).trim()
    return got || fallback
  }
  return {
    bg: v('--dsw-alias-bg-base', '#1b1b1f'),
    layer1: v('--dsw-alias-bg-layer-1', '#242429'),
    layer2: v('--dsw-alias-bg-layer-2', '#2d2d33'),
    text: v('--dsw-alias-label-primary', '#e8e8ea'),
    text2: v('--dsw-alias-label-secondary', '#9a9aa2'),
    border: v('--dsw-alias-border-l1', '#3a3a42'),
    border2: v('--dsw-alias-border-l2', '#4d4d57'),
    brand: v('--dsw-alias-brand-primary', '#4c8dff'),
  }
}

const FONT_UI = 'system-ui, "Microsoft YaHei", sans-serif'
const FONT_MONO = 'ui-monospace, Consolas, monospace'

/**
 * 视口跨挂载保留。
 * main 是 keyed slot，切到「对话」再回来时这个组件会被重新挂载，
 * 不记住视口的话用户每次都要重新滚到刚才看的地方。
 */
const VIEW = { vp: null, fitKey: null }

function GraphCanvas({ layout, graph, selectedId, focus, onSelect, onToggleGroup, onContextMenu }) {
  const wrapRef = React.useRef(null)
  const canvasRef = React.useRef(null)
  const vpRef = React.useRef(VIEW.vp || { x: 0, y: 0, scale: 1 })
  const sizeRef = React.useRef({ w: 0, h: 0 })
  const styleRef = React.useRef(null)
  const hoverRef = React.useRef(null)
  const dragRef = React.useRef(null)
  const rafRef = React.useRef(0)
  const layoutRef = React.useRef(layout)
  const selectedRef = React.useRef(selectedId)
  const graphRef = React.useRef(graph)
  const [zoom, setZoom] = React.useState(VIEW.vp ? VIEW.vp.scale : 1)

  layoutRef.current = layout
  selectedRef.current = selectedId
  graphRef.current = graph

  // ---- 绘制 ---------------------------------------------------------------

  const paint = React.useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const { w, h } = sizeRef.current
    const lay = layoutRef.current
    if (!w || !h || !lay) return

    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (!styleRef.current) styleRef.current = readStyle(wrapRef.current)
    const S = styleRef.current
    const vp = vpRef.current
    const g = graphRef.current

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = S.bg
    ctx.fillRect(0, 0, w, h)

    // 世界坐标变换：之后的绘制一律用世界单位，线宽/字号也跟着缩放
    ctx.setTransform(dpr * vp.scale, 0, 0, dpr * vp.scale, -vp.x * dpr * vp.scale, -vp.y * dpr * vp.scale)

    const vx0 = vp.x
    const vy0 = vp.y
    const vx1 = vp.x + w / vp.scale
    const vy1 = vp.y + h / vp.scale
    const pad = 60
    const visible = (n) => n.x + n.w >= vx0 - pad && n.x <= vx1 + pad && n.y + n.h >= vy0 - pad && n.y <= vy1 + pad
    const selId = selectedRef.current
    ctx.textBaseline = 'middle'

    // 1. 组框背景
    for (const n of lay.nodes) {
      if (n.kind !== 'frame' || !visible(n)) continue
      ctx.fillStyle = S.layer1
      ctx.fillStyle = 'rgba(128,128,140,0.07)'
      roundRect(ctx, n.x, n.y, n.w, n.h, 12)
      ctx.fill()
      ctx.strokeStyle = S.border
      ctx.lineWidth = 1 / vp.scale
      ctx.stroke()
      ctx.fillStyle = S.text2
      ctx.font = `600 12px ${FONT_UI}`
      ctx.fillText(n.group.label, n.x + 14, n.y + 18, Math.max(60, n.w - 28))
    }

    // 2. 连线
    for (const e of lay.edges) {
      if (!visible(e.src) && !visible(e.dst)) continue
      const x1 = e.src.x + e.src.w
      const y1 = e.src.y + e.src.h / 2
      const x2 = e.dst.x
      const y2 = e.dst.y + e.dst.h / 2
      const dx = Math.max(24, Math.abs(x2 - x1) * 0.45)
      const hot = selId && (e.src.id === selId || e.dst.id === selId)
      ctx.strokeStyle = hot ? S.brand : S.border2
      ctx.lineWidth = (hot ? 1.8 : 1.2) / vp.scale
      ctx.globalAlpha = hot ? 0.95 : 0.45
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.bezierCurveTo(x1 + dx, y1, x2 - dx, y2, x2, y2)
      ctx.stroke()
      if (e.count > 1 && vp.scale > 0.35) {
        ctx.globalAlpha = 0.9
        ctx.fillStyle = S.text2
        ctx.font = `10px ${FONT_UI}`
        ctx.textAlign = 'center'
        ctx.fillText(`×${e.count}`, (x1 + x2) / 2, (y1 + y2) / 2 - 5)
        ctx.textAlign = 'left'
      }
    }
    ctx.globalAlpha = 1

    // 3. 节点
    for (const n of lay.nodes) {
      if (n.kind === 'frame' || !visible(n)) continue
      const isSel = n.id === selId
      const isHover = n.id === hoverRef.current

      ctx.fillStyle = n.kind === 'group' ? S.layer2 : S.layer1
      roundRect(ctx, n.x, n.y, n.w, n.h, 9)
      ctx.fill()

      ctx.strokeStyle = isSel ? S.brand : (isHover ? S.border2 : S.border)
      ctx.lineWidth = (isSel ? 2 : 1) / vp.scale
      ctx.stroke()

      if (n.kind === 'group') {
        ctx.fillStyle = S.text
        ctx.font = `600 13px ${FONT_UI}`
        ctx.fillText(n.group.label, n.x + 12, n.y + 21, n.w - 24)
        ctx.fillStyle = S.text2
        ctx.font = `11px ${FONT_UI}`
        ctx.fillText(`双击展开 · ${n.group.end - n.group.start} 个算子`, n.x + 12, n.y + 41, n.w - 24)
        continue
      }

      const op = g && g.ops[n.opIndex]
      if (!op) continue
      ctx.fillStyle = S.text
      ctx.font = `600 12.5px ${FONT_MONO}`
      ctx.fillText(opName(op.opcode), n.x + 10, n.y + 16, n.w - 20)

      ctx.fillStyle = S.text2
      ctx.font = `10.5px ${FONT_UI}`
      const outEl = g.operandById.get(op.outs[0])
      ctx.fillText(outEl ? `${outEl.dims.join('×')} · ${typeShort(outEl.type)}` : '—', n.x + 10, n.y + 33, n.w - 20)

      ctx.fillStyle = S.border2
      ctx.beginPath()
      ctx.arc(n.x, n.y + n.h / 2, 2.5 / vp.scale, 0, Math.PI * 2)
      ctx.fill()
      ctx.beginPath()
      ctx.arc(n.x + n.w, n.y + n.h / 2, 2.5 / vp.scale, 0, Math.PI * 2)
      ctx.fill()
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    VIEW.vp = vpRef.current   // 记住视口，切换面板回来时接着看
  }, [])

  const schedule = React.useCallback(() => {
    if (rafRef.current) return
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = 0
      paint()
    })
  }, [paint])

  // ---- 尺寸 / 主题 ---------------------------------------------------------

  React.useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return undefined
    const measure = () => {
      const r = wrap.getBoundingClientRect()
      sizeRef.current = { w: r.width, h: r.height }
      schedule()
    }
    const ro = new ResizeObserver(measure)
    ro.observe(wrap)
    measure()
    return () => ro.disconnect()
  }, [schedule])

  React.useEffect(() => {
    const mo = new MutationObserver(() => {
      styleRef.current = null
      schedule()
    })
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'style', 'data-theme'] })
    return () => mo.disconnect()
  }, [schedule])

  // 换图（或第一次渲染）时自动 fit。
  // fitKey 和视口本身都记在 VIEW 里，所以切走再回来不会重新 fit、也不会丢位置。
  const fitKey = `${graph ? graph.ops.length : 0}:${layout ? layout.nodes.length : 0}`
  React.useEffect(() => {
    const lay = layoutRef.current
    const { w, h } = sizeRef.current
    if (lay && w && h && VIEW.fitKey !== fitKey) {
      VIEW.fitKey = fitKey
      const next = fitViewport(lay, w, h)
      vpRef.current = next
      setZoom(next.scale)
    } else if (VIEW.vp) {
      vpRef.current = VIEW.vp
      setZoom(VIEW.vp.scale)
    }
    schedule()
  }, [layout, fitKey, schedule])

  // 外部定位请求（大纲里点一条）
  const focusNonce = focus ? focus.nonce : 0
  React.useEffect(() => {
    if (!focus || !layoutRef.current) return
    const n = layoutRef.current.nodes.find((x) => x.id === focus.id)
    if (!n) return
    const { w, h } = sizeRef.current
    const vp = vpRef.current
    vpRef.current = {
      scale: vp.scale,
      x: n.x + n.w / 2 - w / (2 * vp.scale),
      y: n.y + n.h / 2 - h / (2 * vp.scale),
    }
    schedule()
  }, [focusNonce, schedule])

  // ---- 交互 ---------------------------------------------------------------

  const toWorld = React.useCallback((clientX, clientY) => {
    const canvas = canvasRef.current
    const r = canvas.getBoundingClientRect()
    const vp = vpRef.current
    return {
      x: (clientX - r.left) / vp.scale + vp.x,
      y: (clientY - r.top) / vp.scale + vp.y,
    }
  }, [])

  const onWheel = React.useCallback((ev) => {
    ev.preventDefault()
    const canvas = canvasRef.current
    if (!canvas) return
    const r = canvas.getBoundingClientRect()
    const vp = vpRef.current
    const mx = ev.clientX - r.left
    const my = ev.clientY - r.top
    const beforeX = mx / vp.scale + vp.x
    const beforeY = my / vp.scale + vp.y
    const nextScale = Math.min(3.5, Math.max(0.06, vp.scale * Math.exp(-ev.deltaY * 0.0015)))
    vpRef.current = {
      scale: nextScale,
      x: beforeX - mx / nextScale,
      y: beforeY - my / nextScale,
    }
    setZoom(nextScale)
    schedule()
  }, [schedule])

  const onPointerDown = React.useCallback((ev) => {
    const canvas = canvasRef.current
    if (!canvas) return
    try { canvas.setPointerCapture(ev.pointerId) } catch { /* 有的宿主不支持 */ }
    const pt = toWorld(ev.clientX, ev.clientY)
    dragRef.current = {
      moved: false,
      pointerId: ev.pointerId,
      startX: ev.clientX,
      startY: ev.clientY,
      vpStart: { ...vpRef.current },
      hit: hitTest(layoutRef.current, pt.x, pt.y),
    }
    canvas.style.cursor = 'grabbing'
  }, [toWorld])

  const onPointerMove = React.useCallback((ev) => {
    const drag = dragRef.current
    const lay = layoutRef.current
    if (!lay) return
    if (drag) {
      const dx = ev.clientX - drag.startX
      const dy = ev.clientY - drag.startY
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true
      if (drag.moved) {
        const vp = drag.vpStart
        vpRef.current = { scale: vp.scale, x: vp.x - dx / vp.scale, y: vp.y - dy / vp.scale }
        schedule()
      }
      return
    }
    const pt = toWorld(ev.clientX, ev.clientY)
    const hit = hitTest(lay, pt.x, pt.y)
    const id = hit ? hit.id : null
    if (id !== hoverRef.current) {
      hoverRef.current = id
      if (canvasRef.current) canvasRef.current.style.cursor = hit ? 'pointer' : 'grab'
      schedule()
    }
  }, [schedule, toWorld])

  const onPointerUp = React.useCallback(() => {
    const drag = dragRef.current
    dragRef.current = null
    if (canvasRef.current) canvasRef.current.style.cursor = 'grab'
    if (!drag || drag.moved) return
    onSelect(drag.hit || null)
  }, [onSelect])

  const onDblClick = React.useCallback((ev) => {
    const pt = toWorld(ev.clientX, ev.clientY)
    const hit = hitTest(layoutRef.current, pt.x, pt.y)
    if (hit && hit.kind === 'group') onToggleGroup(hit.group.id)
  }, [onToggleGroup, toWorld])

  const onCtx = React.useCallback((ev) => {
    ev.preventDefault()
    const pt = toWorld(ev.clientX, ev.clientY)
    const hit = hitTest(layoutRef.current, pt.x, pt.y)
    onContextMenu(hit, ev.clientX, ev.clientY)
  }, [onContextMenu, toWorld])

  return React.createElement('div', {
    ref: wrapRef,
    className: 'kv-canvas-wrap',
    onWheel,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onDoubleClick: onDblClick,
    onContextMenu: onCtx,
  },
  React.createElement('canvas', { ref: canvasRef, className: 'kv-canvas', style: { cursor: 'grab' } }),
  React.createElement('div', { className: 'kv-zoom-chip' }, `${Math.round(zoom * 100)}%`),
  React.createElement('div', { className: 'kv-canvas-hint' }, '滚轮缩放 · 拖拽平移 · 双击组展开'))
}

// --------------------------------------------------------------------------- //

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2)
  ctx.beginPath()
  ctx.moveTo(x + rr, y)
  ctx.arcTo(x + w, y, x + w, y + h, rr)
  ctx.arcTo(x + w, y + h, x, y + h, rr)
  ctx.arcTo(x, y + h, x, y, rr)
  ctx.arcTo(x, y, x + w, y, rr)
  ctx.closePath()
}

function fitViewport(lay, w, h) {
  const cw = Math.max(220, lay.width)
  const ch = Math.max(160, lay.height)
  const scale = Math.min(1.05, Math.max(0.06, Math.min(w / cw, h / ch) * 0.92))
  return {
    scale,
    x: cw / 2 - w / (2 * scale),
    y: -24 / scale,
  }
}


  // ===== ui/panels.js =====
/**
 * 侧栏与面板组件：大纲 / 检查器 / 统计 / 问题列表。
 * 全都是纯展示 + 回调，状态在 page.js 里。
 */





const fmt = (n) => {
  if (n == null) return '—'
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${n} B`
}

// --------------------------------------------------------------------------- //
// 大纲：分组 + 组内算子
// --------------------------------------------------------------------------- //

function Outline({ graph, groups, expanded, activeGroupId, onToggle, onFocusOp, onFocusGroup }) {
  if (!graph) {
    return React.createElement('div', { className: 'kv-empty' }, '还没打开图')
  }
  return React.createElement('div', null,
    groups.map((g) => {
      const open = expanded.has(g.id)
      const isActive = activeGroupId === g.id
      const inner = []
      inner.push(React.createElement('div', {
        key: 'g',
        className: `kv-group-row${isActive ? ' active' : ''}`,
        onClick: () => onFocusGroup(g),
      },
      React.createElement('span', {
        className: 'kv-caret',
        onClick: (ev) => { ev.stopPropagation(); onToggle(g.id) },
      }, open ? '▾' : '▸'),
      React.createElement('span', { className: 'kv-group-name' }, g.label),
      React.createElement('span', { className: 'kv-group-meta' }, `${g.end - g.start}`)))

      if (open) {
        for (let i = g.start; i < g.end; i += 1) {
          const op = graph.ops[i]
          if (!op) continue
          const out = graph.operandById.get(op.outs[0])
          inner.push(React.createElement('div', {
            key: `o${i}`,
            className: 'kv-group-row',
            style: { paddingLeft: 22, fontSize: 11.5 },
            onClick: () => onFocusOp(i),
          },
          React.createElement('span', { className: 'kv-group-name' }, `#${i} ${opName(op.opcode)}`),
          React.createElement('span', { className: 'kv-group-meta' },
            out ? `${out.dims.join('×')}` : '')))
        }
      }
      return inner
    }))
}

// --------------------------------------------------------------------------- //
// 检查器
// --------------------------------------------------------------------------- //

function Inspector({ graph, sel, consts, dirtyConsts, onEditDims, onDeleteOp, onDuplicateOp, onEditConst, onFocusOperand }) {
  if (!graph || !sel) {
    return React.createElement('div', { className: 'kv-empty' },
      '点一个节点看它的输入输出、形状、常量路径。',
      React.createElement('br'),
      '双击折叠的组可以展开；滚轮缩放。')
  }

  if (sel.kind === 'group' || sel.kind === 'frame') {
    const g = sel.group
    let bytes = 0
    let ops = 0
    for (let i = g.start; i < g.end; i += 1) {
      const op = graph.ops[i]
      if (!op) continue
      ops += 1
      for (const oid of op.outs) {
        const el = graph.operandById.get(oid)
        if (el) bytes += operandBytes(el)
      }
    }
    return React.createElement('div', null,
      React.createElement('div', { className: 'kv-kv' },
        React.createElement('dt', null, '分组'), React.createElement('dd', null, g.label),
        React.createElement('dt', null, '算子'), React.createElement('dd', null, String(ops)),
        React.createElement('dt', null, '输出字节'), React.createElement('dd', null, fmt(bytes)),
        React.createElement('dt', null, '区间'), React.createElement('dd', null, `#${g.start} … #${g.end - 1}`)),
      React.createElement('div', { className: 'kv-note' }, '双击画布上的组节点可以展开/折叠。'))
  }

  const op = graph.ops[sel.opIndex]
  if (!op) return React.createElement('div', { className: 'kv-empty' }, '算子不存在')

  const warns = pitfalls(op, graph)
  const slots = paramSlots(op.opcode)
  const want = ARITY[op.opcode]

  const constBytes = () => {
    let sum = 0
    for (const oid of op.ins) {
      const el = graph.operandById.get(oid)
      if (!el) continue
      if (el.tailKind === 'w') sum += el.tailLen != null ? el.tailLen : operandBytes(el)
      else if (el.tailKind === 'a') sum += 4
    }
    for (const oid of op.outs) {
      const el = graph.operandById.get(oid)
      if (el) sum += 0
    }
    return sum
  }

  const tensorRow = (oid, role) => {
    const el = graph.operandById.get(oid)
    if (!el) {
      return React.createElement('div', { key: `${role}${oid}`, className: 'kv-tensor' },
        React.createElement('span', { className: 'kv-tid' }, String(oid)),
        React.createElement('span', { className: 'kv-tshape', style: { color: 'var(--dsw-alias-state-error-primary)' } }, '未声明'))
    }
    const isParam = role === 'in' && slots.includes(op.ins.indexOf(oid))
    return React.createElement('div', {
      key: `${role}${oid}`,
      className: 'kv-tensor',
      onClick: () => onFocusOperand && onFocusOperand(el),
      title: el.tailPath || '',
    },
    React.createElement('span', { className: 'kv-tid' }, String(oid)),
    React.createElement('span', { className: 'kv-tshape' }, `${el.dims.join('×') || 'scalar'}`),
    React.createElement('span', { className: 'kv-ttype' }, typeShort(el.type)),
    el.tailKind === 'w'
      ? React.createElement('span', { className: 'kv-tconst' }, isParam ? '参数' : '常量')
      : null,
    el.tailKind === 'a'
      ? React.createElement('span', { className: 'kv-tconst' }, `a ${el.tailValue}`)
      : null,
    isParam && dirtyConsts && dirtyConsts.has(oid)
      ? React.createElement('span', { className: 'kv-badge warn' }, '改')
      : null)
  }

  return React.createElement('div', null,
    React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 } },
      React.createElement('strong', { style: { fontSize: 13 } }, `#${op.index} ${opName(op.opcode)}`),
      React.createElement('span', { className: 'kv-badge' }, `op ${op.opcode}`),
      want != null && op.ins.length !== want
        ? React.createElement('span', { className: 'kv-badge warn' }, `输入 ${op.ins.length}/${want}`)
        : null),

    warns.length
      ? warns.map((w, i) => React.createElement('div', {
        key: `w${i}`,
        className: `kv-note ${w.level === 'error' ? 'err' : 'warn'}`,
      }, w.text))
      : null,

    React.createElement('div', { className: 'kv-sub' }, `输入 · ${op.ins.length}`),
    op.ins.map((oid) => tensorRow(oid, 'in')),

    React.createElement('div', { className: 'kv-sub' }, `输出 · ${op.outs.length}`),
    op.outs.map((oid) => tensorRow(oid, 'out')),

    React.createElement('div', { className: 'kv-sub' }, '常量占用'),
    React.createElement('div', { className: 'kv-mono kv-muted', style: { fontSize: 11.5 } }, fmt(constBytes())),

    op.outs.length === 1
      ? React.createElement(OutputShapeEditor, {
        graph,
        oid: op.outs[0],
        onEditDims,
      })
      : null,

    slots.length
      ? React.createElement(ParamEditor, {
        graph, op, slots, consts, onEditConst,
      })
      : null,

    React.createElement('div', { className: 'kv-sub' }, '操作'),
    React.createElement('div', { style: { display: 'flex', gap: 6, flexWrap: 'wrap' } },
      React.createElement('button', {
        className: 'kv-btn',
        onClick: () => onDuplicateOp(op.index),
      }, '复制算子'),
      React.createElement('button', {
        className: 'kv-btn kv-btn-danger',
        onClick: () => onDeleteOp(op.index),
      }, '删除算子')))
}

/** 输出形状：改了就触发下游重推 */
function OutputShapeEditor({ graph, oid, onEditDims }) {
  const el = graph.operandById.get(oid)
  const [text, setText] = React.useState(el ? el.dims.join(' ') : '')
  React.useEffect(() => { setText(el ? el.dims.join(' ') : '') }, [oid, el ? el.dims.join(' ') : ''])
  if (!el) return null
  const commit = () => {
    const dims = text.trim().split(/\s+/).filter(Boolean).map((v) => parseInt(v, 10))
    if (!dims.length || dims.some((v) => !Number.isFinite(v) || v < 0)) {
      setText(el.dims.join(' '))
      return
    }
    onEditDims(oid, dims)
  }
  return React.createElement('div', null,
    React.createElement('div', { className: 'kv-sub' }, '输出形状（改了下游自动重推）'),
    React.createElement('div', { className: 'kv-dims' },
      React.createElement('input', {
        className: 'kv-input',
        style: { width: '100%' },
        value: text,
        onChange: (ev) => setText(ev.target.value),
        onBlur: commit,
        onKeyDown: (ev) => { if (ev.key === 'Enter') ev.currentTarget.blur() },
        placeholder: '例如 256 2048',
      })))
}

/** 参数常量（RESHAPE 的 shape / TRANSPOSE 的 perm / REDUCE 的 axis …） */
function ParamEditor({ graph, op, slots, consts, onEditConst }) {
  const rows = slots.map((slotIndex) => {
    const oid = op.ins[slotIndex]
    const el = graph.operandById.get(oid)
    if (!el) return null
    const ints = constInts(el, consts ? consts.get(oid) : null)
    const editable = Boolean(ints) && (el.tailKind === 'a' || el.tailKind === 'w')
    return { slotIndex, oid, el, ints, editable }
  }).filter(Boolean)

  if (!rows.length) return null

  return React.createElement('div', null,
    React.createElement('div', { className: 'kv-sub' }, '参数'),
    rows.map((r) => React.createElement(ParamRow, {
      key: r.oid,
      row: r,
      onEditConst,
    })))
}

function ParamRow({ row, onEditConst }) {
  const { oid, el, ints, editable } = row
  const [text, setText] = React.useState(ints ? ints.join(' ') : '')
  const label = opNameForParam(el)
  React.useEffect(() => {
    setText(ints ? ints.join(' ') : '')
  }, [oid, ints ? ints.join(' ') : ''])

  if (!ints) {
    return React.createElement('div', { className: 'kv-note' },
      `operand ${oid} 是常量（${el.tailKind === 'w' ? el.tailPath : `a ${el.tailValue}`}），但读不到内容 —— 形状推导会跳过这个算子。`)
  }

  const commit = () => {
    const next = text.trim().split(/[\s,]+/).filter(Boolean).map((v) => parseInt(v, 10))
    if (!next.length || next.some((v) => !Number.isFinite(v))) {
      setText(ints.join(' '))
      return
    }
    onEditConst(oid, next)
  }

  return React.createElement('div', { style: { marginBottom: 6 } },
    React.createElement('div', { className: 'kv-muted', style: { fontSize: 10.5, marginBottom: 2 } },
      `${label} · operand ${oid}${editable ? '' : '（只读）'}`),
    React.createElement('input', {
      className: 'kv-input',
      value: text,
      readOnly: !editable,
      onChange: (ev) => setText(ev.target.value),
      onBlur: commit,
      onKeyDown: (ev) => { if (ev.key === 'Enter') ev.currentTarget.blur() },
    }),
    el.tailKind === 'w'
      ? React.createElement('div', { className: 'kv-muted', style: { fontSize: 10, marginTop: 2, wordBreak: 'break-all' } },
        el.tailPath, el.tailOff != null ? ` @${el.tailOff}+${el.tailLen}` : '')
      : null)
}

function opNameForParam(el) {
  if (!el) return '参数'
  return el.dims && el.dims.length ? `int32[${el.dims.join('×')}]` : 'int32'
}

// --------------------------------------------------------------------------- //
// 统计
// --------------------------------------------------------------------------- //

function StatsView({ graph, st, groups }) {
  if (!graph || !st) return React.createElement('div', { className: 'kv-empty' }, '还没打开图')
  const byGroup = React.useMemo(() => statsByGroups(graph, groups), [graph, groups])
  const maxCount = st.byOpcode.length ? st.byOpcode[0].count : 1

  return React.createElement('div', null,
    React.createElement('div', { className: 'kv-kv' },
      React.createElement('dt', null, '算子'), React.createElement('dd', null, String(st.opCount)),
      React.createElement('dt', null, 'operand'), React.createElement('dd', null, String(st.operandCount)),
      React.createElement('dt', null, '常量个数'), React.createElement('dd', null, String(st.constOperands)),
      React.createElement('dt', null, '常量字节'), React.createElement('dd', null, fmt(st.constBytes)),
      React.createElement('dt', null, '张量字节'), React.createElement('dd', null, fmt(st.operandBytes)),
      React.createElement('dt', null, '常量文件'), React.createElement('dd', null, String(st.uniqueConstPaths))),

    React.createElement('div', { className: 'kv-sub' }, 'opcode 分布'),
    st.byOpcode.map((row) => React.createElement('div', { key: row.code, className: 'kv-bar' },
      React.createElement('span', { className: 'kv-bar-name', title: `op ${row.code}` }, row.name),
      React.createElement('span', { className: 'kv-bar-track' },
        React.createElement('span', {
          className: 'kv-bar-fill',
          style: { width: `${Math.max(2, (row.count / maxCount) * 100)}%` },
        })),
      React.createElement('span', { className: 'kv-bar-num' }, String(row.count)))),

    React.createElement('div', { className: 'kv-sub' }, '按分组'),
    React.createElement('table', { style: { width: '100%', fontSize: 11.5, borderCollapse: 'collapse' } },
      React.createElement('thead', null,
        React.createElement('tr', { style: { color: 'var(--dsw-alias-label-secondary)' } },
          React.createElement('th', { style: { textAlign: 'left', padding: '3px 4px' } }, '组'),
          React.createElement('th', { style: { textAlign: 'right', padding: '3px 4px' } }, '算子'),
          React.createElement('th', { style: { textAlign: 'right', padding: '3px 4px' } }, '字节'))),
      React.createElement('tbody', null,
        byGroup.map((g) => React.createElement('tr', { key: g.id },
          React.createElement('td', { style: { padding: '3px 4px' } },
            `${g.id + 1}. #${g.start}–${g.end - 1}`),
          React.createElement('td', { style: { textAlign: 'right', padding: '3px 4px' } }, String(g.opCount)),
          React.createElement('td', { style: { textAlign: 'right', padding: '3px 4px' } }, fmt(g.bytes)))))))
}

// --------------------------------------------------------------------------- //
// 问题列表
// --------------------------------------------------------------------------- //

function ProblemsView({ problems, onFocusProblem }) {
  if (!problems || !problems.length) {
    return React.createElement('div', { className: 'kv-empty' },
      React.createElement('span', { className: 'kv-badge ok' }, '自检通过'),
      ' 结构、id 连续性、形状推导都没问题。')
  }
  return React.createElement('div', null,
    problems.map((p, i) => React.createElement('div', {
      key: `${i}`,
      className: `kv-problem ${p.level === 'error' ? 'err' : (p.level === 'warn' ? 'warn' : 'info')}`,
      onClick: () => onFocusProblem && onFocusProblem(p),
    },
    React.createElement('span', { className: 'lv' }, p.level === 'error' ? '✕' : (p.level === 'warn' ? '!' : 'i')),
    React.createElement('span', null, p.text))))
}


  // ===== ui/page.js =====
/**
 * 主页面 —— 把 core（数据）+ ui（画布/面板）接起来。
 *
 * 数据流向：
 *   Host /katorch-viz/open  →  原文 + 形状参数常量
 *        ↓ parseGtxt（core）
 *   graph（唯一真相，编辑都在它上面改）
 *        ↓ buildLayout（ui）
 *   canvas 绘制
 *        ↓ serialize（core）
 *   Host /katorch-viz/save  →  g.txt
 *
 * 校验（结构 / id 连续性 / 形状推导）在浏览器里跑，不用往返服务器。
 */












const API = '/katorch-viz'
const NS = 'katorch-viz'

// --------------------------------------------------------------------------- //
// 常量字节 <-> int32 数组
// --------------------------------------------------------------------------- //

function constMapFrom(json) {
  const m = new Map()
  for (const [oid, ints] of Object.entries(json || {})) {
    m.set(Number(oid), uint8FromInts(ints))
  }
  return m
}

function uint8FromInts(ints) {
  const buf = new Uint8Array(ints.length * 4)
  const dv = new DataView(buf.buffer)
  ints.forEach((v, i) => dv.setInt32(i * 4, v | 0, true))
  return buf
}

function intsFromU8(u8) {
  if (!u8) return []
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength)
  const n = Math.floor(u8.byteLength / 4)
  const out = new Array(n)
  for (let i = 0; i < n; i += 1) out[i] = dv.getInt32(i * 4, true)
  return out
}

async function postJson(route, body) {
  const res = await fetch(API + route, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body || {}),
  })
  const data = await res.json().catch(() => ({ ok: false, error: '返回不是 JSON' }))
  if (!res.ok || data.ok === false) throw new Error(data.error || `HTTP ${res.status}`)
  return data
}

// --------------------------------------------------------------------------- //
// 面板状态为什么放在模块级
//
// main 是 keyed slot：用户在侧栏点「对话」，这个 key 一变，React 就把整个面板
// unmount 掉，组件里的 useState 全部销毁 —— 切回来时图、展开状态、选中全没了。
//
// 所以状态放模块级对象 PANEL 里：组件挂载时从它取初值，每次变化写回。
// 轻量偏好（路径、分组方式、展开的组）再落一层 localStorage，刷新页面也能接上。
// 图对象本身不落 localStorage（几百 KB~，且可从路径重读），只活在内存里。
// --------------------------------------------------------------------------- //

const LS_KEY = 'katorch-viz:prefs'

function loadPrefs() {
  try {
    const raw = window.localStorage.getItem(LS_KEY)
    return raw ? (JSON.parse(raw) || {}) : {}
  } catch {
    return {}   // 隐私模式 / 禁了 localStorage
  }
}

function savePrefs(patch) {
  try {
    window.localStorage.setItem(LS_KEY, JSON.stringify({ ...loadPrefs(), ...patch }))
  } catch { /* 存不上就算了，不影响主流程 */ }
}

const PREFS = loadPrefs()

const PANEL = {
  graph: null,
  openedPath: PREFS.openedPath || '',
  pathInput: PREFS.pathInput || '',
  origText: '',
  constStore: null,
  constMeta: {},
  constDirty: null,
  rev: 0,
  sel: null,
  expanded: new Set(Array.isArray(PREFS.expanded) ? PREFS.expanded : []),
  groupMode: PREFS.groupMode || 'period',
  groupSize: PREFS.groupSize || 284,
  rightTab: PREFS.rightTab || 'inspect',
  footOpen: PREFS.footOpen !== false,
  dirty: false,
  /** canvas 的缩放/平移，由 canvas.js 写 */
  view: null,
  fitKey: null,
}

/**
 * 和 useState 一样用，但值放在 PANEL 里，组件重新挂载时能接上。
 * @param {string} key PANEL 里的键
 * @param {any|Function} initial 没有存过值时的初值
 */
function usePanelState(key, initial) {
  const [value, setValue] = React.useState(() => {
    if (PANEL[key] != null) return PANEL[key]
    return typeof initial === 'function' ? initial() : initial
  })
  React.useEffect(() => { PANEL[key] = value }, [key, value])
  const set = React.useCallback((next) => {
    setValue((prev) => {
      const v = typeof next === 'function' ? next(prev) : next
      PANEL[key] = v
      return v
    })
  }, [key])
  return [value, set]
}

// --------------------------------------------------------------------------- //
// 页面
// --------------------------------------------------------------------------- //

function KatorchPage() {
  // 会跨挂载保留的
  const [graph, setGraph] = usePanelState('graph', null)
  const [openedPath, setOpenedPath] = usePanelState('openedPath', '')
  const [pathInput, setPathInput] = usePanelState('pathInput', '')
  const [origText, setOrigText] = usePanelState('origText', '')
  const [constStore, setConstStore] = usePanelState('constStore', () => new Map())
  const [constMeta, setConstMeta] = usePanelState('constMeta', {})
  const [constDirty, setConstDirty] = usePanelState('constDirty', () => new Set())
  const [rev, setRev] = usePanelState('rev', 0)
  const [sel, setSel] = usePanelState('sel', null)
  const [expanded, setExpanded] = usePanelState('expanded', () => new Set())
  const [groupMode, setGroupMode] = usePanelState('groupMode', 'period')
  const [groupSize, setGroupSize] = usePanelState('groupSize', 284)
  const [rightTab, setRightTab] = usePanelState('rightTab', 'inspect')
  const [footOpen, setFootOpen] = usePanelState('footOpen', true)
  const [dirty, setDirty] = usePanelState('dirty', false)

  // 纯临时 UI，切走就该消失
  const [busy, setBusy] = React.useState(false)
  const [toast, setToast] = React.useState(null)
  const [menu, setMenu] = React.useState(null)
  const [focusNode, setFocusNode] = React.useState(null)
  const [browser, setBrowser] = React.useState(null)
  const toastTimer = React.useRef(0)

  const bump = React.useCallback(() => setRev((r) => r + 1), [])

  const say = React.useCallback((kind, text) => {
    setToast({ kind, text })
    clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), kind === 'err' ? 6500 : 3200)
  }, [])

  // ---- 初始化：问 Host 要默认路径 -----------------------------------------
  // 只在「这次真的没有图」时才自动打开。面板被切走再切回来时 PANEL 里还有图，
  // 这里必须什么都不做，否则会重新读文件、把用户的展开/选中状态冲掉。
  React.useEffect(() => {
    if (PANEL.graph) return undefined
    let alive = true
    fetch(`${API}/state`)
      .then((r) => r.json())
      .then((d) => {
        if (!alive || !d || !d.ok) return
        const p = d.lastPath || (d.config && d.config.defaultPath) || PANEL.pathInput || ''
        if (p) {
          if (!PANEL.pathInput) setPathInput(p)
          if (!PANEL.graph) doOpen(p).catch(() => { /* 默认路径打不开就算了 */ })
        }
      })
      .catch(() => { /* Host 不在就等用户手输 */ })
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---- 轻量偏好落 localStorage（刷新页面也能接上）--------------------------
  React.useEffect(() => {
    savePrefs({
      openedPath,
      pathInput,
      groupMode,
      groupSize,
      rightTab,
      footOpen,
      expanded: [...expanded],
    })
  }, [openedPath, pathInput, groupMode, groupSize, rightTab, footOpen, expanded])

  // ---- 打开 ---------------------------------------------------------------

  const doOpen = React.useCallback(async (p) => {
    const target = String(p || '').trim()
    if (!target) { say('err', t('needOpen')); return }
    setBusy(true)
    try {
      const data = await postJson('/open', { path: target })
      const parsed = parseGtxt(data.text)
      setGraph(parsed.graph)
      setOpenedPath(data.path)
      setPathInput(data.path)
      setOrigText(data.text)
      setConstStore(constMapFrom(data.consts))
      setConstMeta(data.constMeta || {})
      setConstDirty(new Set())
      setDirty(false)
      setSel(null)
      setExpanded(new Set())
      setMenu(null)
      bump()
      const warns = parsed.warnings.length
      say('ok', `已打开 ${parsed.graph.ops.length} 个算子 / ${parsed.graph.operandById.size} 个 operand`
        + (warns ? `（${warns} 条解析警告）` : ''))
    } catch (err) {
      say('err', String(err.message || err))
    } finally {
      setBusy(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bump, say])

  // ---- 派生数据 -----------------------------------------------------------

  const groups = React.useMemo(
    () => (graph ? makeGroups(graph, { mode: groupMode, size: groupSize }) : []),
    [graph, groupMode, groupSize, rev],
  )

  const layout = React.useMemo(
    () => (graph ? buildLayout(graph, groups, expanded) : null),
    [graph, groups, expanded, rev],
  )

  const curStats = React.useMemo(() => (graph ? stats(graph) : null), [graph, rev])

  const problems = React.useMemo(() => {
    if (!graph) return []
    const structural = checkStructure(graph)
    const shape = validateGraph(graph, constStore)
    return [...structural, ...shape]
  }, [graph, constStore, rev])

  const errorCount = React.useMemo(() => problems.filter((p) => p.level === 'error').length, [problems])

  // ---- 编辑 ---------------------------------------------------------------

  const markDirty = React.useCallback(() => setDirty(true), [])

  const onEditDims = React.useCallback((oid, dims) => {
    if (!graph) return
    setOperandDims(graph, oid, dims)
    const res = propagateShapes(graph, constStore)
    markDirty()
    bump()
    if (res.changed.length) say('ok', `下游 ${res.changed.length} 个张量跟着重推`)
    const errs = res.problems.filter((p) => p.level === 'error')
    if (errs.length) say('err', errs[0].text)
  }, [graph, constStore, bump, say, markDirty])

  const onEditConst = React.useCallback((oid, ints) => {
    if (!graph) return
    const el = graph.operandById.get(oid)
    if (!el) return
    if (el.tailKind === 'a') {
      el.tailValue = ints[0] | 0
      el.dirty = true
      const res = propagateShapes(graph, constStore)
      markDirty()
      bump()
      if (res.problems.some((p) => p.level === 'error')) say('err', '这个改动让下游不兼容了，看问题面板')
      else say('ok', `operand ${oid} 的常量已改成 ${el.tailValue}`)
      return
    }
    const next = new Map(constStore)
    next.set(oid, uint8FromInts(ints))
    setConstStore(next)
    const d = new Set(constDirty)
    d.add(oid)
    setConstDirty(d)
    const res = propagateShapes(graph, next)
    markDirty()
    bump()
    const errs = res.problems.filter((p) => p.level === 'error')
    if (errs.length) say('err', errs[0].text)
    else say('ok', `operand ${oid} 的参数改成 [${ints.join(', ')}]（导出时会写回 .bin）`)
  }, [graph, constStore, constDirty, bump, say, markDirty])

  const onDeleteOp = React.useCallback((opIndex) => {
    if (!graph) return
    const op = graph.ops[opIndex]
    if (!op) return
    const label = `#${opIndex} ${opName(op.opcode)}`
    if (!window.confirm(`删除 ${label}？\n它的独占输出张量也会一起删掉，后面的 operand id 会重新连续编号。`)) return
    try {
      deleteOp(graph, opIndex)
      setSel(null)
      markDirty()
      bump()
      say('ok', `已删除 ${label}`)
    } catch (err) {
      say('err', String(err.message || err))
    }
  }, [graph, bump, say, markDirty])

  const onDuplicateOp = React.useCallback((opIndex) => {
    if (!graph) return
    try {
      const res = duplicateOp(graph, opIndex)
      markDirty()
      bump()
      setSel({ kind: 'op', opIndex: res.opIndex, id: `o${res.opIndex}` })
      say('ok', `已复制成 #${res.opIndex}`)
    } catch (err) {
      say('err', String(err.message || err))
    }
  }, [graph, bump, say, markDirty])

  // ---- 校验 / 导出 --------------------------------------------------------

  const doCheck = React.useCallback(() => {
    if (!graph) { say('err', t('needOpen')); return }
    bump()
    if (errorCount) say('err', `自检发现 ${errorCount} 个错误`)
    else say('ok', `${t('checked')}：结构 + id 连续性 + 形状推导都通过`)
  }, [graph, bump, errorCount, say])

  const doExport = React.useCallback(async (targetPath) => {
    if (!graph) { say('err', t('needOpen')); return }
    const target = String(targetPath || openedPath || '').trim()
    if (!target) { say('err', '没有目标路径'); return }
    setBusy(true)
    try {
      // 1. 先把改过的常量写回 .bin（不然图里写的参数和文件里的对不上）
      let written = 0
      for (const oid of constDirty) {
        const el = graph.operandById.get(oid)
        if (!el || el.tailKind !== 'w' || !el.tailPath) continue
        const ints = intsFromU8(constStore.get(oid))
        await postJson('/write-const', {
          path: el.tailPath,
          ints,
          off: el.tailOff,
          len: el.tailLen,
        })
        written += 1
      }
      if (written) say('ok', `${t('constsWritten')} ${written} 个`)

      // 2. 强制重编号后序列化
      normalizeIds(graph)
      const text = serialize(graph)

      // 3. 写盘前最后一次自检（错了就不写，避免把坏图交给 host.c）
      const check = parseGtxt(text)
      const structural = checkStructure(check.graph).filter((p) => p.level === 'error')
      const shape = validateGraph(check.graph, constStore).filter((p) => p.level === 'error')
      if (structural.length || shape.length) {
        say('err', `自检有 ${structural.length + shape.length} 个错误，没有写盘。看问题面板。`)
        bump()
        setFootOpen(true)
        return
      }

      const saved = await postJson('/save', { path: target, text })
      setOrigText(text)
      setOpenedPath(saved.path)
      setPathInput(saved.path)
      setConstDirty(new Set())
      setDirty(false)
      bump()
      say('ok', `${t('exported')} ${saved.path}（${saved.bytes} 字节${saved.backedUpTo ? '，原文件已备份' : ''}）`)
    } catch (err) {
      say('err', String(err.message || err))
    } finally {
      setBusy(false)
    }
  }, [graph, openedPath, constDirty, constStore, bump, say])

  // ---- 交互辅助 -----------------------------------------------------------

  const groupOfOp = React.useCallback((opIndex) => {
    for (const g of groups) if (opIndex >= g.start && opIndex < g.end) return g
    return null
  }, [groups])

  const focusOp = React.useCallback((opIndex) => {
    const g = groupOfOp(opIndex)
    if (g && !expanded.has(g.id)) {
      const next = new Set(expanded)
      next.add(g.id)
      setExpanded(next)
    }
    setSel({ kind: 'op', opIndex, id: `o${opIndex}` })
    setFocusNode({ id: `o${opIndex}`, nonce: Date.now() })
  }, [groupOfOp, expanded])

  const focusGroup = React.useCallback((g) => {
    setSel({ kind: 'group', group: g, id: `g${g.id}` })
    setFocusNode({ id: `g${g.id}`, nonce: Date.now() })
  }, [])

  const toggleGroup = React.useCallback((groupId) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(groupId)) next.delete(groupId)
      else next.add(groupId)
      return next
    })
  }, [])

  const onFocusProblem = React.useCallback((p) => {
    if (p.opIndex != null) focusOp(p.opIndex)
    else if (p.oid != null && graph) {
      const el = graph.operandById.get(p.oid)
      if (el && el.opIndex >= 0) focusOp(el.opIndex)
    }
  }, [focusOp, graph])

  const onCanvasSelect = React.useCallback((node) => { setSel(node) }, [])

  const onContextMenu = React.useCallback((node, x, y) => {
    if (!node) { setMenu(null); return }
    setMenu({ x, y, node })
  }, [])

  // 右键菜单关掉
  React.useEffect(() => {
    if (!menu) return undefined
    const close = () => setMenu(null)
    const onKey = (ev) => { if (ev.key === 'Escape') setMenu(null) }
    window.addEventListener('pointerdown', close)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('pointerdown', close)
      window.removeEventListener('keydown', onKey)
    }
  }, [menu])

  // ---- 浏览目录（简单的弹层）---------------------------------------------
  const doBrowse = React.useCallback(async (dir) => {
    try {
      const res = await fetch(`${API}/browse?path=${encodeURIComponent(dir || pathInput || 'F:\\\\')}`)
      const data = await res.json()
      if (!data.ok) throw new Error(data.error)
      setBrowser(data)
    } catch (err) {
      say('err', String(err.message || err))
    }
  }, [pathInput, say])

  const fileInputRef = React.useRef(null)

  /** 本地文件兜底：Host 路由不可用时，直接读用户拖进来的文件 */
  const onDropFile = React.useCallback(async (file) => {
    if (!file) return
    const text = await file.text()
    const parsed = parseGtxt(text)
    setGraph(parsed.graph)
    setOpenedPath('')
    setPathInput(file.name)
    setOrigText(text)
    setConstStore(new Map())
    setConstMeta({})
    setConstDirty(new Set())
    setDirty(false)
    setSel(null)
    setExpanded(new Set())
    bump()
    say('ok', `已从本地文件载入 ${parsed.graph.ops.length} 个算子（没有磁盘路径，导出会走浏览器下载）`)
  }, [bump, say])

  const doDownload = React.useCallback(() => {
    if (!graph) return
    normalizeIds(graph)
    const text = serialize(graph)
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'g.txt'
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 4000)
    say('ok', '已下载 g.txt')
  }, [graph, say])

  // ---- 渲染 ---------------------------------------------------------------

  const activeGroupId = sel && sel.kind === 'group' ? sel.group.id
    : (sel && sel.kind === 'op' ? (groupOfOp(sel.opIndex) || {}).id : undefined)

  return React.createElement('div', { className: 'kv-root' },
    // 工具栏
    React.createElement('div', { className: 'kv-toolbar' },
      React.createElement('span', { className: 'kv-title' }, t('title')),
      React.createElement('input', {
        className: 'kv-path',
        value: pathInput,
        placeholder: t('path.ph'),
        onChange: (ev) => setPathInput(ev.target.value),
        onKeyDown: (ev) => { if (ev.key === 'Enter') doOpen(pathInput) },
      }),
      React.createElement('button', { className: 'kv-btn kv-btn-primary', disabled: busy, onClick: () => doOpen(pathInput) }, t('open')),
      React.createElement('button', { className: 'kv-btn', onClick: () => doBrowse(pathInput) }, t('browse')),
      React.createElement('button', {
        className: 'kv-btn', disabled: !graph || busy, onClick: () => doCheck(),
      }, t('check')),
      React.createElement('button', {
        className: 'kv-btn', disabled: !graph || busy, onClick: () => doExport(openedPath || pathInput),
      }, t('export')),
      React.createElement('button', {
        className: 'kv-btn', disabled: !openedPath || busy, onClick: () => doExport(pathInput),
      }, '另存为'),
      React.createElement('button', {
        className: 'kv-btn', disabled: !graph, onClick: doDownload, title: '不经过 Host，直接浏览器下载',
      }, '下载'),
      React.createElement('span', { className: 'kv-spacer' }),
      dirty ? React.createElement('span', { className: 'kv-dirty' }, t('dirty')) : null,
      React.createElement('select', {
        className: 'kv-select',
        value: groupMode,
        onChange: (ev) => { setGroupMode(ev.target.value); setExpanded(new Set()) },
        title: t('group'),
      },
      React.createElement('option', { value: 'period' }, t('group.auto')),
      React.createElement('option', { value: 'size' }, t('group.size')),
      React.createElement('option', { value: 'none' }, t('group.none'))),
      groupMode === 'size'
        ? React.createElement('input', {
          className: 'kv-input',
          style: { width: 78 },
          value: String(groupSize),
          onChange: (ev) => {
            const v = parseInt(ev.target.value, 10)
            if (Number.isFinite(v) && v > 0) setGroupSize(v)
          },
          title: t('group.size.ph'),
        })
        : null,
      React.createElement('button', {
        className: 'kv-btn', disabled: !groups.length,
        onClick: () => setExpanded(new Set(groups.map((g) => g.id))),
      }, t('expandAll')),
      React.createElement('button', {
        className: 'kv-btn', disabled: !expanded.size,
        onClick: () => setExpanded(new Set()),
      }, t('collapseAll'))),

    // 主体
    React.createElement('div', { className: 'kv-body' },
      React.createElement('div', { className: 'kv-outline' },
        React.createElement('div', { className: 'kv-panel-head' },
          t('outline'),
          React.createElement('span', { className: 'kv-spacer' }),
          React.createElement('span', { className: 'kv-muted', style: { fontSize: 10.5 } },
            graph ? `${groups.length} 组` : '')),
        React.createElement('div', { className: 'kv-panel-body' },
          React.createElement(Outline, {
            graph,
            groups,
            expanded,
            activeGroupId,
            onToggle: toggleGroup,
            onFocusOp: focusOp,
            onFocusGroup: focusGroup,
          }))),

      React.createElement('div', { className: 'kv-center' },
        layout
          ? React.createElement(GraphCanvas, {
            layout,
            graph,
            selectedId: sel ? sel.id : null,
            focus: focusNode,
            onSelect: onCanvasSelect,
            onToggleGroup: toggleGroup,
            onContextMenu,
          })
          : React.createElement('div', { className: 'kv-empty', style: { margin: 'auto' } },
            React.createElement('div', { style: { fontWeight: 600, marginBottom: 6 } }, t('empty.title')),
            t('empty.hint'),
            React.createElement('div', { style: { marginTop: 12 } },
              React.createElement('button', { className: 'kv-btn', onClick: () => fileInputRef.current && fileInputRef.current.click() },
                '从本地文件载入（拖进来也行）'),
              React.createElement('input', {
                ref: fileInputRef,
                type: 'file',
                accept: '.txt',
                style: { display: 'none' },
                onChange: (ev) => { onDropFile(ev.target.files && ev.target.files[0]); ev.target.value = '' },
              })))),

        // 底部问题区
        React.createElement('div', { className: `kv-foot${footOpen ? '' : ' collapsed'}` },
          React.createElement('div', {
            className: 'kv-panel-head',
            style: { cursor: 'pointer' },
            onClick: () => setFootOpen((v) => !v),
          },
          React.createElement('span', null, footOpen ? '▾' : '▸'),
          React.createElement('span', null, t('problems')),
          errorCount
            ? React.createElement('span', { className: 'kv-badge err' }, `${errorCount} 个错误`)
            : React.createElement('span', { className: 'kv-badge ok' }, t('noProblems')),
          React.createElement('span', { className: 'kv-spacer' }),
          problems.length
            ? React.createElement('span', { className: 'kv-muted', style: { fontSize: 10.5 } }, `${problems.length} 条`)
            : null),
          footOpen
            ? React.createElement('div', { className: 'kv-panel-body', style: { padding: '4px 6px' } },
              React.createElement(ProblemsView, { problems, onFocusProblem }))
            : null),

      // 右栏
      React.createElement('div', { className: 'kv-side' },
        React.createElement('div', { className: 'kv-tabs' },
          React.createElement('button', {
            className: `kv-tab${rightTab === 'inspect' ? ' active' : ''}`,
            onClick: () => setRightTab('inspect'),
          }, t('inspect')),
          React.createElement('button', {
            className: `kv-tab${rightTab === 'stats' ? ' active' : ''}`,
            onClick: () => setRightTab('stats'),
          }, t('stats'))),
        React.createElement('div', { className: 'kv-panel-body' },
          rightTab === 'inspect'
            ? React.createElement(Inspector, {
              graph,
              sel,
              consts: constStore,
              dirtyConsts: constDirty,
              onEditDims,
              onDeleteOp,
              onDuplicateOp,
              onEditConst,
              onFocusOperand: (el) => { if (el.opIndex >= 0) focusOp(el.opIndex) },
            })
            : React.createElement(StatsView, { graph, st: curStats, groups })))),

    toast
      ? React.createElement('div', { className: `kv-toast ${toast.kind}` }, toast.text)
      : null,

    menu
      ? React.createElement('div', {
        className: 'kv-menu',
        style: { left: menu.x, top: menu.y },
        onPointerDown: (ev) => ev.stopPropagation(),
      },
      menu.node.kind === 'group'
        ? React.createElement('button', {
          onClick: () => {
            toggleGroup(menu.node.group.id)
            setMenu(null)
          },
        }, expanded.has(menu.node.group.id) ? t('menu.collapse') : t('menu.expand'))
        : null,
      menu.node.kind === 'group'
        ? React.createElement('button', {
          onClick: () => { focusGroup(menu.node.group); setMenu(null) },
        }, t('menu.focus'))
        : null,
      menu.node.kind === 'op'
        ? React.createElement('button', {
          onClick: () => { focusOp(menu.node.opIndex); setMenu(null) },
        }, t('menu.focus'))
        : null,
      menu.node.kind === 'op' ? React.createElement('div', { className: 'sep' }) : null,
      menu.node.kind === 'op'
        ? React.createElement('button', {
          onClick: () => { onDuplicateOp(menu.node.opIndex); setMenu(null) },
        }, t('menu.duplicate'))
        : null,
      menu.node.kind === 'op'
        ? React.createElement('button', {
          onClick: () => { onDeleteOp(menu.node.opIndex); setMenu(null) },
        }, t('menu.delete'))
        : null)
      : null,

    // 目录浏览器
    browser
      ? React.createElement('div', {
        className: 'kv-menu',
        style: { left: 24, top: 96, width: 520, maxHeight: '70vh', overflow: 'auto', position: 'fixed' },
      },
      React.createElement('div', { style: { display: 'flex', gap: 6, padding: '4px 4px 8px' } },
        React.createElement('button', { className: 'kv-btn', onClick: () => doBrowse(browser.parent) }, '↑ 上一级'),
        React.createElement('span', { className: 'kv-muted', style: { fontSize: 11, alignSelf: 'center', wordBreak: 'break-all' } }, browser.path),
        React.createElement('span', { className: 'kv-spacer' }),
        React.createElement('button', { className: 'kv-btn', onClick: () => setBrowser(null) }, '关')),
      (browser.dirs || []).map((d) => React.createElement('button', {
        key: d.path,
        onClick: () => doBrowse(d.path),
      }, `📁 ${d.name}`)),
      (browser.files || []).length
        ? React.createElement('div', { className: 'sep' })
        : null,
      (browser.files || []).map((f) => React.createElement('button', {
        key: f.path,
        onClick: () => { setBrowser(null); setPathInput(f.path); doOpen(f.path) },
      }, `📄 ${f.name}${f.size != null ? `  (${f.size} B)` : ''}`)))
      : null)
}

// --------------------------------------------------------------------------- //
// 插件入口
// --------------------------------------------------------------------------- //

function injectStyle(css) {
  const el = document.createElement('style')
  el.setAttribute('data-katorch-viz', '1')
  el.textContent = css
  document.head.appendChild(el)
  return () => { try { el.remove() } catch { /* 已经不在 */ } }
}

function NavIcon({ size }) {
  const s = size || 20
  return React.createElement('svg', {
    width: s, height: s, viewBox: '0 0 24 24', fill: 'none',
    stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round', strokeLinejoin: 'round',
    'aria-hidden': true, style: { display: 'block' },
  },
  React.createElement('rect', { x: 2.5, y: 4, width: 7, height: 6, rx: 1.6 }),
  React.createElement('rect', { x: 14.5, y: 3, width: 7, height: 5, rx: 1.6 }),
  React.createElement('rect', { x: 14.5, y: 15, width: 7, height: 6, rx: 1.6 }),
  React.createElement('path', { d: 'M9.5 7h5M18 8v7M9.5 7v10h5' }))
}

function apply(ctx) {
  ctx.effect(() => ctx.locale.register(NS, DICTS), 'katorch-viz: dictionaries')
  try { setTranslator(ctx.locale.bind(NS)) } catch { /* 没有 locale 服务就用中文兜底 */ }
  ctx.effect(() => injectStyle(CSS), 'katorch-viz: styles')

  ctx.slots.inject('sidebar.panellist', () => ctx.slots.register({
    name: 'sidebar.panellist',
    id: 'katorch-viz',
    order: 21,
    label: () => t('nav'),
  }, NavIcon))

  ctx.slots.inject('main', () => ctx.slots.register({
    name: 'main',
    key: 'katorch-viz',
  }, KatorchPage))
}

    return { inject: ['slots', 'locale'], apply }
  },
})
