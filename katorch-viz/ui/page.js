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
import React from 'react'
import { CSS } from './theme.js'
import { t, DICTS, setTranslator } from './i18n.js'
import { makeGroups, buildLayout } from './layout.js'
import { GraphCanvas } from './canvas.js'
import { Outline, Inspector, StatsView, ProblemsView } from './panels.js'
import { parseGtxt, checkStructure } from '../core/parser.js'
import {
  serialize, normalizeIds, stats, deleteOp, duplicateOp, setOperandDims,
} from '../core/export.js'
import { validateGraph, propagateShapes } from '../core/shape.js'
import { opName, operandBytes } from '../core/spec.js'


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

export function KatorchPage() {
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

export function apply(ctx) {
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
