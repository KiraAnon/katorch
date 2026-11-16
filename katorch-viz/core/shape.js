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

import {
  opName,
  UNARY, BINARY, REDUCE,
  FC, CONCAT, CONV, RESHAPE, TRANSPOSE, BATCH_MATMUL,
  constInts,
} from './spec.js'

export class ShapeError extends Error {}

/**
 * 单个算子的形状推导。
 * @param {number} opcode
 * @param {number[][]} ins 输入张量形状
 * @param {{axis?:number, perm?:number[], shape?:number[], adj_x?:number, adj_y?:number, keepdim?:boolean, pad?:number[], stride?:number[]}} kw
 * @returns {number[][]} 输出形状列表
 */
export function infer(opcode, ins, kw = {}) {
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
export function broadcast(a, b) {
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
export function opKw(opEl, graph, constData) {
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
export function inferOp(opEl, graph, constData) {
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
export function validateGraph(graph, constData) {
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
export function propagateShapes(graph, constData, maxRounds = 12) {
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
