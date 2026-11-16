/**
 * 节点画布 —— 纯 Canvas 2D，不依赖任何图形库。
 *
 * 为什么不用 React Flow / litegraph：折叠后画布上只有几十个节点，
 * 展开一层也就几百个，Canvas 直接画足够快，而且省掉一层依赖和一套样式打架。
 * 需求 §7.2 要的「1 万节点」靠 折叠 + 视口裁剪 达成。
 */
import React from 'react'
import { hitTest } from './layout.js'
import { opName, typeShort } from '../core/spec.js'


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

export function GraphCanvas({ layout, graph, selectedId, focus, onSelect, onToggleGroup, onContextMenu }) {
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
