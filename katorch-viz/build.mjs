/**
 * 把 core/（纯数据层）和 ui/（React 组件）拼成一个浏览器可加载的 client.js。
 *
 * 为什么自己拼而不是上打包器：
 *   - 这几份文件只用 ESM 的顶层 import/export，规则简单到一次正则就够
 *   - 少一个构建依赖，改一行就能重编
 *   - core 同时被 Host（index.js，原生 ESM）和 Client（这一份）使用，
 *     保证「界面看到的」和「导出的」是同一套逻辑
 *
 * 顺序是拓扑序：依赖方必须排在被依赖方之后。
 * 拼完做两层校验：语法检查 + 关键符号存在性检查。
 */
import fs from 'node:fs'
import path from 'node:path'
import vm from 'node:vm'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))

const MODULES = [
  // core：不依赖 React
  'core/spec.js',
  'core/parser.js',
  'core/shape.js',
  'core/export.js',
  // ui：i18n / theme / layout 是纯的，canvas / panels / page 用 React
  'ui/i18n.js',
  'ui/theme.js',
  'ui/layout.js',
  'ui/canvas.js',
  'ui/panels.js',
  'ui/page.js',
]

/** 去掉 ESM 语法，留下可拼进同一个作用域的普通代码 */
function stripModule(src, file) {
  let out = src
  // import ... from '...'（含跨行写法）
  out = out.replace(/^import\s[\s\S]*?from\s+['"][^'"]+['"];?[ \t]*$/gm, '')
  // 裸 import '...'
  out = out.replace(/^import\s+['"][^'"]+['"];?[ \t]*$/gm, '')
  // export function / const / class ...
  out = out.replace(/^export\s+(async\s+function|function|const|let|var|class)/gm, '$1')
  if (/^\s*(import|export)\s/m.test(out)) {
    const line = out.split('\n').find((l) => /^\s*(import|export)\s/.test(l))
    throw new Error(`${file}: 还有没处理掉的 ESM 语法 → ${line.trim()}`)
  }
  return out.trimEnd()
}

/** 找出模块顶层的声明名，用来查重 */
function topLevelNames(code) {
  const names = []
  const re = /^(?:const|let|var|function|class|async function)\s+([A-Za-z_$][\w$]*)/gm
  let m
  while ((m = re.exec(code)) !== null) names.push(m[1])
  return names
}

function build() {
  const parts = []
  const seen = new Map()
  const warnings = []

  for (const rel of MODULES) {
    const abs = path.join(here, rel)
    if (!fs.existsSync(abs)) throw new Error(`缺文件：${abs}`)
    const code = stripModule(fs.readFileSync(abs, 'utf8'), rel)
    for (const name of topLevelNames(code)) {
      if (seen.has(name)) {
        warnings.push(`顶层声明重名：${name}（${seen.get(name)} 与 ${rel}）`)
      } else {
        seen.set(name, rel)
      }
    }
    parts.push(`\n  // ===== ${rel} =====\n${code}\n`)
  }

  const body = parts.join('\n')

  const out = `/**
 * KAtorch 可视化编译器 —— Client 半（自动生成，别手改）。
 * 源码在 core/ 与 ui/，用 node build.mjs 重新生成。
 * 生成时间：${new Date().toISOString()}
 */
window.__ModuleLoader__.load({
  id: '@local/katorch-viz',
  factory(require) {
    const React = require('react')
${body}
    return { inject: ['slots', 'locale'], apply }
  },
})
`

  fs.writeFileSync(path.join(here, 'client.js'), out, 'utf8')

  // 语法检查：包一层函数体当脚本解析（只解析不执行，ESM 语法已经剥掉）
  try {
    new vm.Script(`(function(){\n${out}\n})`)
  } catch (err) {
    console.error('\n生成的文件有语法错误：')
    console.error(err.message)
    console.error(err.stack ? String(err.stack).split('\n').slice(0, 6).join('\n') : '')
    process.exitCode = 1
  }

  // 关键符号在不在
  const need = ['parseGtxt', 'serialize', 'buildLayout', 'GraphCanvas', 'KatorchPage', 'apply', 'CSS', 't']
  const missing = need.filter((n) => !new RegExp(`(?:^|\\s)(?:function|const|let|class)\\s+${n}\\b`).test(out))
  if (missing.length) {
    console.error(`\n缺关键符号：${missing.join(', ')}`)
    process.exitCode = 1
  }

  if (warnings.length) {
    console.warn('\n⚠ 顶层声明重名（拼进同一作用域会报错）：')
    for (const w of warnings) console.warn(`  ${w}`)
    process.exitCode = 1
  }

  const kb = (out.length / 1024).toFixed(1)
  console.log(`client.js 已生成：${kb} KB / ${out.split('\n').length} 行`)
  console.log(`模块 ${MODULES.length} 个，顶层声明 ${seen.size} 个`)
  if (process.exitCode) console.log('但有错误，见上。')
}

build()
