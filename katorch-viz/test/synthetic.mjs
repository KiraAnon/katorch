/**
 * 形状推导的合成图测试 + 与 Python 权威实现（shape_infer.validate_txt）对拍。
 *
 * 跑法：node test/synthetic.mjs
 */
import fs from 'node:fs'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

import { parseGtxt, checkStructure } from '../core/parser.js'
import { serialize, setOperandDims } from '../core/export.js'
import { validateGraph, propagateShapes, infer } from '../core/shape.js'
import { pitfalls, FC, RESHAPE, TRANSPOSE } from '../core/spec.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const dir = path.join(here, 'fixtures', 'synthetic')
const gPath = path.join(dir, 'g.txt')

let pass = 0
const failures = []
const ok = (tag, cond, detail) => {
  if (cond) { pass += 1; console.log(`  PASS ${tag}`) }
  else { failures.push(`${tag}${detail ? `: ${detail}` : ''}`); console.log(`  FAIL ${tag}${detail ? `: ${detail}` : ''}`) }
}

const text = fs.readFileSync(gPath, 'utf8')

/** 把所有 w 常量的 .bin 读进来 */
function loadConsts(graph) {
  const data = new Map()
  const missing = []
  for (const el of graph.operandById.values()) {
    if (el.tailKind !== 'w' || !el.tailPath) continue
    try {
      const buf = fs.readFileSync(el.tailPath)
      if (el.tailOff != null && el.tailLen != null) {
        data.set(el.id, buf.subarray(el.tailOff, el.tailOff + el.tailLen))
      } else {
        data.set(el.id, buf)
      }
    } catch {
      missing.push(el.tailPath)
    }
  }
  return { data, missing }
}

// --------------------------------------------------------------------------- //
console.log('\n=== 合成图：有常量时的形状推导 ===')
const { graph, warnings } = parseGtxt(text)
ok('解析无警告', warnings.length === 0, warnings.join(' | '))
ok('往返一致', serialize(parseGtxt(text).graph) === text)

const { data, missing } = loadConsts(graph)
ok('常量文件全部读到', missing.length === 0, missing.join(', '))
ok('读到 4 个常量', data.size === 4, `实际 ${data.size}`)

const problems = validateGraph(graph, data)
const errors = problems.filter((p) => p.level === 'error')
const infos = problems.filter((p) => p.level === 'info')
ok('形状校验无 error', errors.length === 0, errors.map((p) => p.text).join(' | '))
ok('没有因缺常量而跳过', infos.length === 0, infos.map((p) => p.text).join(' | '))

// 逐个算子的推导结果
console.log('  推导出的形状：')
const expectShapes = { 2: [1, 8, 4], 4: [8, 4], 7: [8, 1], 11: [8, 16] }
for (const op of graph.ops) {
  const outId = op.outs[0]
  const el = graph.operandById.get(outId)
  console.log(`    #${op.index} op${op.opcode} -> operand ${outId} [${el.dims}]`)
  const want = expectShapes[outId]
  if (want) ok(`operand ${outId} 推导 = [${want}]`, el.dims.join(',') === want.join(','), `得到 [${el.dims}]`)
}

// --------------------------------------------------------------------------- //
console.log('\n=== 故意写错一个声明，必须被抓到 ===')
{
  const g2 = parseGtxt(text).graph
  const { data: d2 } = loadConsts(g2)
  const el = g2.operandById.get(7)          // REDUCE_SUM 的输出，应为 [8,1]
  el.dims = [8, 4]
  el.nd = 2
  el.dirty = true
  const errs = validateGraph(g2, d2).filter((p) => p.level === 'error')
  ok('抓到 REDUCE_SUM 声明错', errs.some((p) => p.text.includes('operand 7') || p.text.includes('输出 7')),
    errs.map((p) => p.text).join(' | '))
  ok('错误文案指出了 adapter 会静默出错',
    errs.some((p) => p.text.includes('静默')))
}

// --------------------------------------------------------------------------- //
console.log('\n=== 改上游形状 → 下游自动重算 ===')
{
  const g3 = parseGtxt(text).graph
  const { data: d3 } = loadConsts(g3)
  // 把 RESHAPE 的目标形状改成 [4,8]（元素数仍是 32，合法）
  const shapeOperand = g3.operandById.get(3)
  ok('shape 常量就是 perm/axis/shape 里的那个 w', shapeOperand && shapeOperand.tailPath.endsWith('shape.bin'))
  const shapeBytes = Buffer.alloc(8)
  shapeBytes.writeInt32LE(4, 0)
  shapeBytes.writeInt32LE(8, 4)
  d3.set(3, shapeBytes)

  const res = propagateShapes(g3, d3)
  const o4 = g3.operandById.get(4)
  const o7 = g3.operandById.get(7)
  ok('RESHAPE 输出跟着变成 [4,8]', o4.dims.join(',') === '4,8', `得到 [${o4.dims}]`)
  ok('REDUCE_SUM 输出跟着变成 [4,1]', o7.dims.join(',') === '4,1', `得到 [${o7.dims}]`)
  // 这个改动把下游 FC 弄坏了（x[-1]=1 != w[-1]=4）——这正是编辑器必须报出来的
  const still = res.problems.filter((p) => p.level === 'error')
  ok('下游 FC 不兼容时被报出来', still.some((p) => p.text.includes('FC')),
    still.map((p) => p.text).join(' | '))

  const out = serialize(g3)
  const rp = parseGtxt(out)
  ok('改完仍能往返', serialize(rp.graph) === out)
  ok('改完结构仍合法', checkStructure(rp.graph).filter((p) => p.level === 'error').length === 0)
  const changed = out.split('\n').filter((l, i) => l !== text.split('\n')[i])
  console.log(`  改动的行：${changed.join(' / ')}`)
  // 注意：REDUCE 的输出形状在 OPD 行里，所以改了 4 和 7 两行
  ok('只改了受影响的行', changed.length <= 3, `改了 ${changed.length} 行`)
}

// --------------------------------------------------------------------------- //
console.log('\n=== 坑检查（需求 §10）===')
{
  const g4 = parseGtxt(text).graph
  const fcOp = g4.ops.find((op) => op.opcode === FC)
  ok('FC 的 act 是 type 1 时不报警', pitfalls(fcOp, g4).filter((p) => p.level === 'error').length === 0)

  // 把 act 改成 type 1 —— 需求 §10 说的「静默通过、结果全错」
  const actEl = g4.operandById.get(fcOp.ins[3])
  ok('FC 的 act 原本是 type 4', actEl.type === 4, `实际 type ${actEl.type}`)
  actEl.type = 1
  const warns = pitfalls(fcOp, g4)
  ok('FC act 变成 type 1 时报警', warns.some((p) => p.level === 'error' && p.text.includes('act')))
}

// --------------------------------------------------------------------------- //
console.log('\n=== 与 Python 权威实现对拍 ===')
{
  const py = findPython()
  if (!py) {
    console.log('  跳过：找不到 python（可以装一个再跑，或用 --python 指定）')
  } else {
    const script = String.raw`
import sys, struct, json
sys.path.insert(0, r'F:\KAtorch\编译器')
import shape_infer as si

def rdf(p):
    try: return open(p, 'rb').read()
    except Exception: return None

with open(r'${gPath.replace(/\\/g, '\\\\')}', 'r', encoding='utf-8') as f:
    text = f.read()
errs = si.validate_txt(text, rdf)
print(json.dumps(errs, ensure_ascii=False))
`
    try {
      const out = execFileSync(py, ['-c', script], { encoding: 'utf8', timeout: 30000 })
      const pyErrs = JSON.parse(out.trim().split('\n').pop())
      console.log(`  Python validate_txt 报 ${pyErrs.length} 条：`)
      for (const e of pyErrs) console.log(`    ${e}`)
      ok('Python 也认为这份图没有形状错误', pyErrs.length === 0, pyErrs.join(' | '))

      // 对错误的图，两边都应该报错
      const badText = text.replace('OPD 7 8 2 8 1 0 0 o', 'OPD 7 8 2 8 4 0 0 o')
      const badPath = gPath.replace('g.txt', '_bad.txt')
      fs.writeFileSync(badPath, badText, 'utf8')
      const script2 = script.replace(gPath.replace(/\\/g, '\\\\'), badPath.replace(/\\/g, '\\\\'))
      const out2 = execFileSync(py, ['-c', script2], { encoding: 'utf8', timeout: 30000 })
      const pyErrs2 = JSON.parse(out2.trim().split('\n').pop())
      ok('Python 也认为错误的图有错', pyErrs2.length > 0)

      const gBad = parseGtxt(badText).graph
      const jsErrs2 = validateGraph(gBad, loadConsts(gBad).data).filter((p) => p.level === 'error')
      ok('JS 与 Python 在错误图上结论一致', jsErrs2.length > 0)
      fs.unlinkSync(badPath)
    } catch (err) {
      console.log(`  对拍失败：${err.message.split('\n')[0]}`)
    }
  }
}

// --------------------------------------------------------------------------- //
function findPython() {
  const candidates = [
    process.env.KATORCH_PYTHON,
    'python',
    'py',
    'C:\\Users\\Administrator\\miniconda3\\python.exe',
  ].filter(Boolean)
  for (const c of candidates) {
    try {
      execFileSync(c, ['-c', 'import sys; print(sys.version_info[0])'], { encoding: 'utf8', timeout: 8000 })
      return c
    } catch { /* 试下一个 */ }
  }
  return null
}

// --------------------------------------------------------------------------- //
console.log(`\n=== 结果：${pass} 通过 / ${failures.length} 失败 ===`)
if (failures.length) {
  for (const f of failures) console.log(`  !! ${f}`)
  process.exit(1)
}
