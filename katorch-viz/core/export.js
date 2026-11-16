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

import { opName, operandBytes, typeShort } from './spec.js'

/**
 * 重新连续编号（幂等）。
 * 已经连续时什么都不改，因此不影响逐字节往返。
 * @returns {{changed:boolean, map:Map<number,number>}}
 */
export function normalizeIds(graph) {
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

export function formatElement(el) {
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
export function serialize(graph, opts = {}) {
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
export function preview(graph) {
  return graph.elements.map((el) => formatElement(el)).join('\n') + '\n'
}

// --------------------------------------------------------------------------- //
// 图的编辑操作
// --------------------------------------------------------------------------- //

/** 删除一个算子及其独占的输出 operand */
export function deleteOp(graph, opIndex) {
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
export function duplicateOp(graph, opIndex) {
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
export function setOperandDims(graph, oid, dims) {
  const el = graph.operandById.get(oid)
  if (!el) throw new Error(`operand ${oid} 不存在`)
  el.dims = dims.map((d) => Math.max(0, Math.round(Number(d)) || 0))
  el.nd = el.dims.length
  el.dirty = true
  return el
}

/** 改一个算子的输入（连接编辑 P0 不做，但底层留着） */
export function setOpInputs(graph, opIndex, ins) {
  const op = graph.ops[opIndex]
  if (!op) throw new Error(`没有第 ${opIndex} 个算子`)
  op.ins = ins.map((v) => Math.round(Number(v)))
  op.dirty = true
  relink(graph)
  return op
}

/** 重建 op.index 与 operand 双向索引 */
export function relink(graph) {
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
export function stats(graph) {
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
export function statsByGroups(graph, groups) {
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
export function shapeLabel(el) {
  if (!el) return '?'
  const dims = el.dims.length ? el.dims.join('×') : 'scalar'
  return `${dims} · ${typeShort(el.type)}`
}
