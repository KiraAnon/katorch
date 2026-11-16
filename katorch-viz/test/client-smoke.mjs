/**
 * Client 冒烟测试 —— 在 Node 里加载 client.js，用假 ctx 跑一遍 apply，
 * 再把主面板组件函数真跑一次，抓「一装上就崩」的问题。
 *
 * 拿不到真 React 时用一个极简替身：createElement 造普通对象、hooks 返回固定值。
 * 这样组件函数体里的 TDZ / undefined 引用会立刻暴露，不需要浏览器。
 * 真正的视觉渲染还是要在浏览器里看，见 README。
 */
import fs from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(here, '..')

let pass = 0
const failures = []
const ok = (tag, cond, detail) => {
  if (cond) { pass += 1; console.log(`  PASS ${tag}`) }
  else { failures.push(`${tag}${detail ? `: ${detail}` : ''}`); console.log(`  FAIL ${tag}${detail ? `: ${detail}` : ''}`) }
}

// ---- React：有真的就用真的，没有就用替身 ---------------------------------
function realReact() {
  const dir = process.env.DSH_PROFILE_DIR
  if (!dir) return null
  try {
    const req = createRequire(path.join(dir, 'node_modules') + path.sep)
    return req('react')
  } catch { return null }
}

function fakeReact() {
  const hooks = { state: 0, ref: 0 }
  return {
    version: 'fake',
    createElement(type, props, ...children) {
      return { type, props: { ...(props || {}), children: children.length ? children : undefined } }
    },
    useState(init) {
      const v = typeof init === 'function' ? init() : init
      return [v, () => {}]
    },
    useRef(v) { hooks.ref += 1; return { current: v } },
    useMemo(fn) { return fn() },
    useCallback(fn) { return fn },
    useEffect() {},
    useLayoutEffect() {},
    Fragment: 'Fragment',
  }
}

const React = realReact() || fakeReact()
console.log(`React：${React.version === 'fake' ? '极简替身（渲染由本文件的迷你渲染器完成）' : React.version}`)

// ---- 迷你渲染器（只对替身 React 生效）------------------------------------
function renderNode(vnode, depth = 0, out = []) {
  if (depth > 60) return out
  if (vnode == null || vnode === false || vnode === true) return out
  if (Array.isArray(vnode)) {
    for (const v of vnode) renderNode(v, depth, out)
    return out
  }
  if (typeof vnode === 'string' || typeof vnode === 'number') {
    out.push(String(vnode))
    return out
  }
  if (typeof vnode.type === 'function') {
    const el = vnode.type({ ...(vnode.props || {}) })
    renderNode(el, depth + 1, out)
    return out
  }
  const p = vnode.props || {}
  out.push(`<${vnode.type}${p.className ? ` class="${p.className}"` : ''}>`)
  renderNode(p.children, depth + 1, out)
  out.push(`</${vnode.type}>`)
  return out
}

// ---- DOM 替身 -------------------------------------------------------------
const created = []
class FakeEl {
  constructor(tag) {
    this.tagName = tag
    this.children = []
    this.attributes = {}
    this.style = {}
    this.textContent = ''
  }
  setAttribute(k, v) { this.attributes[k] = v }
  appendChild(c) { this.children.push(c); return c }
  remove() { this.removed = true }
  addEventListener() {}
  removeEventListener() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: 800, height: 600 } }
  getContext() { return null }
  setPointerCapture() {}
}
const documentMock = {
  head: new FakeEl('head'),
  body: new FakeEl('body'),
  documentElement: new FakeEl('html'),
  createElement: (tag) => { const el = new FakeEl(tag); created.push(el); return el },
}

let definition = null
const windowMock = {
  __ModuleLoader__: { load: (def) => { definition = def } },
  devicePixelRatio: 1,
  addEventListener() {},
  removeEventListener() {},
  confirm: () => false,
}

globalThis.window = windowMock
globalThis.document = documentMock
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' })
globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0)
globalThis.cancelAnimationFrame = (id) => clearTimeout(id)
globalThis.ResizeObserver = class { observe() {} disconnect() {} }
globalThis.MutationObserver = class { observe() {} disconnect() {} }
globalThis.Blob = class { constructor() {} }
globalThis.URL = { createObjectURL: () => 'blob:x', revokeObjectURL: () => {} }
globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({ ok: false, error: 'stub' }) })

// ---- 加载 client.js -------------------------------------------------------
const code = fs.readFileSync(path.join(root, 'client.js'), 'utf8')
console.log('\n=== 加载 client.js ===')
try {
  // eslint-disable-next-line no-new-func
  const run = new Function('window', 'document', 'getComputedStyle', code)
  run(windowMock, documentMock, globalThis.getComputedStyle)
} catch (err) {
  console.log(`  FAIL 顶层执行：${err.stack || err.message}`)
  process.exit(1)
}
ok('顶层执行成功', true)
ok('注册了一个模块定义', definition != null)
ok('模块 id 是包名', definition && definition.id === '@local/katorch-viz', definition && definition.id)

// ---- factory --------------------------------------------------------------
console.log('\n=== 跑 factory ===')
let api = null
try {
  api = definition.factory((name) => {
    if (name === 'react') return React
    throw new Error(`不认识的模块：${name}`)
  })
} catch (err) {
  console.log(`  FAIL factory：${err.stack || err.message}`)
  process.exit(1)
}
ok('factory 返回 apply', typeof api.apply === 'function')
ok('声明了 inject', Array.isArray(api.inject) && api.inject.includes('slots'),
  JSON.stringify(api.inject))

// ---- apply ---------------------------------------------------------------
console.log('\n=== 跑 apply ===')
const registrations = []
const ctx = {
  effect(fn) { const dispose = fn(); return () => { if (typeof dispose === 'function') dispose() } },
  locale: {
    register(ns, dicts) { registrations.push({ kind: 'locale', ns, dicts }); return () => {} },
    bind() { return (k) => k },
  },
  slots: {
    inject(name, cb) { registrations.push({ kind: 'inject', name }); cb(); return () => {} },
    register(options, Component) {
      registrations.push({ kind: 'register', options, Component })
      return () => {}
    },
  },
}

try {
  api.apply(ctx)
} catch (err) {
  console.log(`  FAIL apply：${err.stack || err.message}`)
  process.exit(1)
}
ok('apply 没抛异常', true)

const slotRegs = registrations.filter((r) => r.kind === 'register')
console.log(`  注册：${registrations.map((r) => (r.kind === 'register' ? r.options.name : r.kind)).join(', ')}`)
ok('注册了 sidebar.panellist', slotRegs.some((r) => r.options.name === 'sidebar.panellist'))
ok('注册了 main 面板', slotRegs.some((r) => r.options.name === 'main' && r.options.key === 'katorch-viz'))
ok('注册了 locale 字典', registrations.some((r) => r.kind === 'locale'))
ok('注入了样式', created.some((el) => el.tagName === 'style' && el.textContent.includes('.kv-root')))

const comps = slotRegs.map((r) => r.Component)
ok('所有注册都拿到了组件函数', comps.every((c) => typeof c === 'function'), `${comps.length} 个`)

// ---- 真跑一次主面板组件函数 ----------------------------------------------
console.log('\n=== 渲染主面板 ===')
const Page = slotRegs.find((r) => r.options.name === 'main').Component
const Icon = slotRegs.find((r) => r.options.name === 'sidebar.panellist').Component
let html = ''
try {
  if (React.version === 'fake') {
    html = renderNode(React.createElement(Page)).join('')
  } else {
    const serverRender = (() => {
      try {
        const req = createRequire(path.join(process.env.DSH_PROFILE_DIR, 'node_modules') + path.sep)
        return req('react-dom/server')
      } catch { return null }
    })()
    if (!serverRender) throw new Error('没有 react-dom/server')
    html = serverRender.renderToString(React.createElement(Page))
  }
} catch (err) {
  console.log(`  FAIL 渲染：${err.stack || err.message}`)
  process.exit(1)
}

ok('渲染出内容', typeof html === 'string' && html.length > 200, `${html.length} 字节`)
ok('带根类名 kv-root', html.includes('kv-root'))
ok('带工具栏', html.includes('kv-toolbar'))
ok('带三栏', html.includes('kv-outline') && html.includes('kv-body') && html.includes('kv-side'))
ok('空态提示在', html.includes('还没有打开图') || html.includes('No graph loaded'))
ok('问题面板在', html.includes('kv-foot'))

let iconHtml = ''
try {
  iconHtml = React.version === 'fake'
    ? renderNode(React.createElement(Icon, { size: 20 })).join('')
    : html
} catch (err) {
  failures.push(`图标组件：${err.message}`)
}
ok('侧栏图标能渲染', iconHtml.includes('svg') || iconHtml.includes('path'), `${iconHtml.length} 字节`)

// ---- 状态跨挂载保留（源码级断言）----------------------------------------
// main 是 keyed slot，切到别的面板会把组件 unmount。业务状态必须走 usePanelState，
// 否则用户切走再切回来，图、展开状态、选中全丢（这是实际踩过的 bug）。
console.log('\n=== 状态跨挂载保留 ===')
{
  const pageSrc = fs.readFileSync(path.join(root, 'ui', 'page.js'), 'utf8')
  const canvasSrc = fs.readFileSync(path.join(root, 'ui', 'canvas.js'), 'utf8')

  ok('定义模块级 PANEL', /const PANEL = \{/.test(pageSrc))
  ok('定义 usePanelState', /function usePanelState\(/.test(pageSrc))

  const mustPersist = [
    'graph', 'openedPath', 'pathInput', 'constStore', 'constDirty',
    'rev', 'sel', 'expanded', 'groupMode', 'groupSize', 'rightTab', 'footOpen', 'dirty',
  ]
  for (const key of mustPersist) {
    const re = new RegExp(`= usePanelState\\('${key}'`)
    ok(`状态 ${key} 走 usePanelState`, re.test(pageSrc))
  }

  // 这些是纯临时 UI，应该继续用普通 useState（切走就该消失）
  for (const key of ['busy', 'toast', 'menu', 'focusNode', 'browser']) {
    const re = new RegExp(`const \\[${key}, set${key[0].toUpperCase()}${key.slice(1)}\\] = React\\.useState`)
    ok(`临时状态 ${key} 仍是 React.useState`, re.test(pageSrc))
  }

  ok('切回来时不重复打开文件', /if \(PANEL\.graph\) return undefined/.test(pageSrc))
  ok('轻量偏好落 localStorage',
    /function savePrefs/.test(pageSrc) && /savePrefs\(\{/.test(pageSrc))

  ok('canvas 视口模块级保留',
    /const VIEW = \{/.test(canvasSrc) && /VIEW\.vp = vpRef\.current/.test(canvasSrc))
  ok('切回来不重新 fit', /VIEW\.fitKey !== fitKey/.test(canvasSrc))
}

console.log(`\n=== 结果：${pass} 通过 / ${failures.length} 失败 ===`)
if (failures.length) process.exit(1)
