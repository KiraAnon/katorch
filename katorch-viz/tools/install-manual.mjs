/**
 * 手工安装兜底 —— 当 install_bundle 被无关的目录锁（EPERM rename dsh-meme）挡住时用。
 *
 * 做的事与 install_bundle 等价的两步：
 *   1. profile/package.json 的 dependencies 加一条 link: 依赖
 *   2. profile/cordis.patch.yml 追加一行插件注册
 * node_modules 的 junction 由调用方先建好。
 *
 * 用 node 写文件而不是 PowerShell，避免 UTF-8 BOM（Node 的 readFileSync 不剥 BOM，
 * 下游 JSON.parse 会直接炸）。
 */
import fs from 'node:fs'
import path from 'node:path'

const profile = process.env.DSH_PROFILE_DIR
if (!profile) {
  console.error('缺 DSH_PROFILE_DIR')
  process.exit(1)
}

const pkgPath = path.join(profile, 'package.json')
const patchPath = path.join(profile, 'cordis.patch.yml')
const LINK = 'link:F:/KAtorch/katorch-viz'

// ---- package.json ---------------------------------------------------------
const pkgText = fs.readFileSync(pkgPath, 'utf8')
const pkg = JSON.parse(pkgText)
if (pkg.dependencies && pkg.dependencies['@local/katorch-viz']) {
  console.log('package.json 已有依赖，跳过')
} else {
  // 文本插入，保住原有格式（整对象重写会把整个文件重排，之后 diff 全是噪声）
  const anchor = '    "@local/dsh-modeling-pipeline"'
  const idx = pkgText.indexOf(anchor)
  if (idx < 0) throw new Error('找不到插入锚点')
  const eol = pkgText.indexOf('\n', idx) + 1
  const next = `${pkgText.slice(0, eol)}    "@local/katorch-viz": "${LINK}",\n${pkgText.slice(eol)}`
  JSON.parse(next)  // 写之前先验一遍
  fs.writeFileSync(pkgPath, next, 'utf8')
  console.log('package.json：已加依赖')
}

// ---- cordis.patch.yml -----------------------------------------------------
let patchText = fs.readFileSync(patchPath, 'utf8')

// profile 的 cordis.patch.yml 顶层行是「按 id 覆盖已有行」，不存在的 id 会被静默忽略。
// 新增插件必须用 - insert: 包一层（这文件里 file-changes / compact 等都是这么进来的）。
// 第一版直接写成 `- id: katorch-viz` → 重启后什么都没发生，就是这个原因。
const WRONG_BLOCK = [
  '# KAtorch 可视化编译器（本地 link 安装）',
  '- id: katorch-viz',
  '  name: "@local/katorch-viz"',
  '  config:',
  "    defaultPath: 'F:\\KAtorch\\PC编译\\wsl-sysroot\\n1\\g.txt'",
  '    constRewrite:',
  "      - [ '/data/data/com.termux/files/home/npt/', 'F:/KAtorch/artifacts/npt/' ]",
].join('\n')

const RIGHT_BLOCK = [
  '# KAtorch 可视化编译器（本地 link 安装）',
  '- insert:',
  '    - id: katorch-viz',
  "      name: '@local/katorch-viz'",
  '      config:',
  "        defaultPath: 'F:\\KAtorch\\PC编译\\wsl-sysroot\\n1\\g.txt'",
  '        constRewrite:',
  "          - ['/data/data/com.termux/files/home/npt/', 'F:/KAtorch/artifacts/npt/']",
].join('\n')

const cleaned = patchText.replace(WRONG_BLOCK, '')
if (cleaned !== patchText) {
  console.log('cordis.patch.yml：已移除写错格式的旧条目')
  patchText = cleaned
}

if (patchText.includes('@local/katorch-viz')) {
  console.log('cordis.patch.yml 已有正确的 insert 条目，跳过')
} else {
  fs.writeFileSync(patchPath, patchText.replace(/\s*$/, '\n') + RIGHT_BLOCK + '\n', 'utf8')
  console.log('cordis.patch.yml：已用 insert 语法追加条目')
}

// ---- 回读验证 -------------------------------------------------------------
const after = JSON.parse(fs.readFileSync(pkgPath, 'utf8'))
console.log('\n回读：')
console.log(`  dependencies.@local/katorch-viz = ${after.dependencies['@local/katorch-viz']}`)
console.log(`  patch 含 katorch-viz = ${fs.readFileSync(patchPath, 'utf8').includes('katorch-viz')}`)
const head = fs.readFileSync(pkgPath, 'utf8').slice(0, 3)
console.log(`  package.json 首字节 = ${JSON.stringify(head)}（应为 '{"\\n'，无 BOM）`)
