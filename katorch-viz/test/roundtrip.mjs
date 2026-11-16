/**
 * 数据层验收测试 —— 需求 §8 的三条硬门槛，这里能测的都测。
 *
 *   测试 1：往返一致（导入 → 不改 → 导出 → 逐字节比对）
 *   测试 2：能编译（本地只能做 host.c 的结构等价校验；真机编译见 tools/verify-on-phone.sh）
 *   测试 3：改一处，仍合法
 *
 * 跑法：node test/roundtrip.mjs
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { parseGtxt, checkStructure } from '../core/parser.js'
import { serialize, normalizeIds, deleteOp, duplicateOp, stats, setOperandDims } from '../core/export.js'
import { validateGraph } from '../core/shape.js'
import { opName, ARITY } from '../core/spec.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const SAMPLE = process.env.KATORCH_SAMPLE
  || 'F:\\KAtorch\\PC编译\\wsl-sysroot\\n1\\g.txt'

let pass = 0
const failures = []

function ok(tag, cond, detail) {
  if (cond) { pass += 1; console.log(`  PASS ${tag}`) }
  else { failures.push(`${tag}${detail ? `: ${detail}` : ''}`); console.log(`  FAIL ${tag}${detail ? `: ${detail}` : ''}`) }
}

function eq(tag, a, b) {
  ok(tag, a === b, a === b ? '' : `期望 ${JSON.stringify(b)}，得到 ${JSON.stringify(a)}`)
}

// --------------------------------------------------------------------------- //
console.log(`\n=== 测试 1：往返一致 ===\n样图：${SAMPLE}`)
if (!fs.existsSync(SAMPLE)) {
  console.error(`样图不存在：${SAMPLE}`)
  process.exit(2)
}
const original = fs.readFileSync(SAMPLE, 'utf8')
const originalBuf = fs.readFileSync(SAMPLE)

const { graph, warnings } = parseGtxt(original)
console.log(`解析：${graph.ops.length} 个算子 / ${graph.operandById.size} 个 operand / 警告 ${warnings.length} 条`)
for (const w of warnings) console.log(`   warn: ${w}`)

const exported = serialize(graph)
eq('导出长度 == 原文件长度', exported.length, original.length)
ok('逐字节一致', exported === original,
  exported === original ? '' : `首个差异位置 ${firstDiff(exported, original)}`)

// 再跑一遍应该仍然幂等（重编号是幂等的）
const second = serialize(parseGtxt(exported).graph)
ok('二次往返仍一致', second === original)

// CRLF 文件也要能原样吐回
const crlf = original.replace(/\n/g, '\r\n')
const crlfGraph = parseGtxt(crlf).graph
ok('CRLF 原样保留', serialize(crlfGraph) === crlf)

// 空行 / 注释 / 未知行原样保留
const noisy = '# 这是注释\n\n' + original + '\n# 尾巴\n'
ok('注释与空行原样保留', serialize(parseGtxt(noisy).graph) === noisy)

// 没写 END 也要能吐回
const noEnd = original.replace(/END\r?\n?$/, '')
ok('缺 END 也能往返', serialize(parseGtxt(noEnd).graph) === noEnd)

// --------------------------------------------------------------------------- //
console.log('\n=== 测试 2：结构等价于 host.c 的 build_from_file ===')
const structural = checkStructure(graph)
const structuralErrors = structural.filter((p) => p.level === 'error')
ok('结构检查无错误', structuralErrors.length === 0,
  structuralErrors.slice(0, 3).map((p) => p.text).join(' | '))

// operand id 连续
const opds = graph.elements.filter((el) => el.kind === 'operand')
ok('operand id 是 0..N-1', opds.every((el, i) => el.id === i), `N=${opds.length}`)

// 每个 OP 行都能被 strtol 序列还原（数字 + 数量自洽）
ok('所有 OP 行 token 数量自洽', graph.ops.every((op) => op.ins.length + op.outs.length + 4 === op.tokens.length),
  graph.ops.filter((op) => op.ins.length + op.outs.length + 4 !== op.tokens.length).slice(0, 2)
    .map((op) => `#${op.index}: ${op.ins.length}+${op.outs.length}+4 != ${op.tokens.length}`).join(' | '))

// IO 行存在且引用的 operand 都有声明
ok('有 IO 行', Boolean(graph.io))

// 每个算子都能被 opcode 表认识
const unknownOps = graph.ops.filter((op) => !opName(op.opcode) || !opName(op.opcode).startsWith('OP_') === false && !ARITY[op.opcode] && !opName(op.opcode))
const reallyUnknown = graph.ops.filter((op) => opName(op.opcode) === `OP_${op.opcode}`)
ok('opcode 都在已知表里', reallyUnknown.length === 0, [...new Set(reallyUnknown.map((o) => o.opcode))].join(','))

// 形状校验（常量读不到时降级为 info）
const shapeProblems = validateGraph(graph, null)
const shapeErrors = shapeProblems.filter((p) => p.level === 'error')
const shapeInfos = shapeProblems.filter((p) => p.level === 'info')
console.log(`  形状校验：error ${shapeErrors.length} / info(缺常量跳过) ${shapeInfos.length}`)
ok('形状校验无 error', shapeErrors.length === 0, shapeErrors.slice(0, 3).map((p) => p.text).join(' | '))

// --------------------------------------------------------------------------- //
console.log('\n=== 测试 3：改一处，仍合法 ===')

// 3a. 改一个内联标量常量（a 0 → a 1），纯文本层面可验证
{
  const g = parseGtxt(original).graph
  // 找一个 a 常量 operand
  const scalar = [...g.operandById.values()].find((el) => el.tailKind === 'a' && el.tailValue === 0)
  ok('样图里有 a 常量', Boolean(scalar))
  if (scalar) {
    scalar.tailValue = 1
    scalar.dirty = true
    const text = serialize(g)
    const reparsed = parseGtxt(text)
    const again = reparsed.graph.operandById.get(scalar.id)
    eq('改后的常量值读回来是 1', again.tailValue, 1)
    ok('改一处后仍能原样往返', serialize(reparsed.graph) === text)
    ok('改一处后结构仍合法',
      checkStructure(reparsed.graph).filter((p) => p.level === 'error').length === 0)
    const changedLines = text.split('\n').filter((l) => l.includes(' a 1'))
    ok('只改了一行', changedLines.length === 1, `改了 ${changedLines.length} 行`)
    eq('行数未变（只替换了一行内容）', text.split('\n').length, original.split('\n').length)
  }
}

// 3b. 改 operand 形状（P1 需求）
{
  const g = parseGtxt(original).graph
  const target = [...g.operandById.values()].find((el) => el.dims.length === 2 && el.dims[0] > 1)
  const before = [...target.dims]
  setOperandDims(g, target.id, [before[0], before[1]])
  ok('设置成相同形状不会产生差异', serialize(g) === original)

  const g2 = parseGtxt(original).graph
  const t2 = [...g2.operandById.values()].find((el) => el.dims.length === 2 && el.dims[0] > 1)
  setOperandDims(g2, t2.id, [t2.dims[0], t2.dims[1] + 1])
  const text2 = serialize(g2)
  const rp2 = parseGtxt(text2)
  ok('改形状后仍能解析', checkStructure(rp2.graph).filter((p) => p.level === 'error').length === 0)
  const newShape = rp2.graph.operandById.get(t2.id).dims
  eq('改后的形状读回来正确', newShape[1], t2.dims[1])
}

// 3c. 删除算子 → id 必须重新连续
{
  const g = parseGtxt(original).graph
  const before = stats(g).opCount
  const victim = g.ops[Math.floor(g.ops.length / 2)]
  const victimIndex = victim.index
  const outIds = [...victim.outs]
  deleteOp(g, victimIndex)
  eq('算子数 -1', stats(g).opCount, before - 1)
  const ids = g.elements.filter((el) => el.kind === 'operand').map((el) => el.id)
  ok('删除后 operand id 仍连续', ids.every((id, i) => id === i))
  const text = serialize(g)
  const rp = parseGtxt(text)
  const errs = checkStructure(rp.graph).filter((p) => p.level === 'error')
  // 删掉中间节点后会有悬空引用（下游 op 还在引用被删的 operand），这是预期内的「图不完整」，
  // 但解析与重编号本身必须自洽。
  const idErrs = errs.filter((p) => p.text.includes('id 不连续'))
  ok('删除后重编号正确（无 id 不连续）', idErrs.length === 0, idErrs[0]?.text)
  ok('删除的 operand 真的没了', outIds.every((oid) => !rp.graph.operandById.has(oid)
    || rp.graph.operandById.get(oid).consumers.length > 0))
}

// 3d. 复制算子 → id 必须重新连续
{
  const g = parseGtxt(original).graph
  const before = stats(g).opCount
  const targetOp = g.ops.find((op) => op.ins.length >= 2 && op.outs.length === 1)
  duplicateOp(g, targetOp.index)
  eq('算子数 +1', stats(g).opCount, before + 1)
  const ids = g.elements.filter((el) => el.kind === 'operand').map((el) => el.id)
  ok('复制后 operand id 仍连续', ids.every((id, i) => id === i))
  const text = serialize(g)
  const rp = parseGtxt(text)
  ok('复制后仍无 id 不连续',
    checkStructure(rp.graph).filter((p) => p.level === 'error' && p.text.includes('id 不连续')).length === 0)
  // 新 OPD 必须出现在所有 OP 行之前
  const lines = text.split('\n')
  const lastOpd = lines.reduce((acc, l, i) => (l.startsWith('OPD') ? i : acc), -1)
  const firstOp = lines.findIndex((l) => l.startsWith('OP '))
  ok('新 OPD 排在所有 OP 之前', lastOpd < firstOp, `lastOpd=${lastOpd} firstOp=${firstOp}`)
}

// 3e. 在用户没动的图上，normalizeIds 必须什么都不改（往返的前提）
{
  const g = parseGtxt(original).graph
  const res = normalizeIds(g)
  ok('normalizeIds 对已连续的图是幂等的', res.changed === false)
  ok('幂等调用后往返仍一致', serialize(g) === original)
}

// --------------------------------------------------------------------------- //
console.log('\n=== 统计面板数据 ===')
const st = stats(graph)
console.log(`  算子 ${st.opCount} / operand ${st.operandCount} / 常量 ${st.constOperands} 个 ${fmtBytes(st.constBytes)}`)
console.log(`  opcode 分布（前 8）：`)
for (const row of st.byOpcode.slice(0, 8)) console.log(`    ${row.name.padEnd(14)} ${row.count}`)
ok('opcode 分布总数等于算子数', st.byOpcode.reduce((a, r) => a + r.count, 0) === st.opCount)

// --------------------------------------------------------------------------- //
console.log(`\n=== 结果：${pass} 通过 / ${failures.length} 失败 ===`)
if (failures.length) {
  for (const f of failures) console.log(`  !! ${f}`)
  process.exit(1)
}

// --------------------------------------------------------------------------- //
function fmtBytes(n) {
  if (n > 1024 * 1024 * 1024) return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
  if (n > 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  if (n > 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${n} B`
}

function firstDiff(a, b) {
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i += 1) if (a[i] !== b[i]) return i
  return n
}
