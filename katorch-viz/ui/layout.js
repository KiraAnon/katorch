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

export const NODE_W = 150
export const NODE_H = 46
export const GROUP_W = 236
export const GROUP_H = 58
export const COL_GAP = 60
export const ROW_GAP = 14
export const GROUP_GAP = 26
export const GROUP_HEAD = 34

/**
 * 自动检测重复周期（DiT 每层的算子序列是一样的）。
 * @param {number[]} codes opcode 序列
 * @param {number} minRate 匹配率下限
 * @returns {{period:number, rate:number}|null}
 */
export function detectPeriod(codes, minRate = 0.9) {
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
export function makeGroups(graph, opts = {}) {
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
export function computeDepths(graph, start, end) {
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
export function buildLayout(graph, groups, expanded) {
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
export function buildEdges(graph, nodeByOp) {
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
export function hitTest(layout, wx, wy) {
  for (let i = layout.nodes.length - 1; i >= 0; i -= 1) {
    const n = layout.nodes[i]
    if (n.kind === 'frame') continue
    if (wx >= n.x && wx <= n.x + n.w && wy >= n.y && wy <= n.y + n.h) return n
  }
  return null
}

/** 让某个节点进入视野 */
export function centerOn(layout, node) {
  return {
    x: node.x + node.w / 2,
    y: node.y + node.h / 2,
  }
}
