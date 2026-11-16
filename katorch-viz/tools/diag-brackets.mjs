// 临时诊断：扫描括号平衡，定位语法错误的行
import fs from 'node:fs'

const file = process.argv[2] || 'client.js'
const lines = fs.readFileSync(file, 'utf8').split('\n')

let depth = 0
const stack = []
const problems = []

for (let n = 0; n < lines.length; n += 1) {
  const s = lines[n]
  let i = 0
  while (i < s.length) {
    const c = s[i]
    if (c === '/' && s[i + 1] === '/') break
    if (c === '/' && s[i + 1] === '*') {
      const j = s.indexOf('*/', i + 2)
      i = j < 0 ? s.length : j + 2
      continue
    }
    if (c === "'" || c === '"') {
      const q = c
      i += 1
      while (i < s.length) {
        if (s[i] === '\\') { i += 2; continue }
        if (s[i] === q) { i += 1; break }
        i += 1
      }
      continue
    }
    if (c === '`') {
      i += 1
      let nest = 0
      while (i < s.length) {
        if (s[i] === '\\') { i += 2; continue }
        if (s[i] === '`' && nest === 0) { i += 1; break }
        if (s[i] === '$' && s[i + 1] === '{') { nest += 1; i += 2; continue }
        if (s[i] === '}' && nest > 0) { nest -= 1; i += 1; continue }
        i += 1
      }
      continue
    }
    if (c === '(' || c === '[' || c === '{') {
      stack.push({ c, line: n + 1 })
      depth += 1
    } else if (c === ')' || c === ']' || c === '}') {
      const top = stack.pop()
      depth -= 1
      const want = c === ')' ? '(' : (c === ']' ? '[' : '{')
      if (!top) {
        problems.push(`行 ${n + 1}: 多余的 ${c}`)
      } else if (top.c !== want) {
        problems.push(`行 ${n + 1}: 遇到 ${c}，但它开的是 ${top.c}（行 ${top.line}）`)
      }
      if (depth < 0) {
        problems.push(`行 ${n + 1}: 深度变负`)
        console.log(problems.slice(0, 8).join('\n'))
        process.exit(0)
      }
    }
    i += 1
  }
}

console.log(`最终深度 ${depth}（应为 0），未闭合 ${stack.length} 个`)
if (problems.length) console.log(problems.slice(0, 8).join('\n'))
for (const s of stack.slice(-5)) console.log(`  未闭合 ${s.c} @ 行 ${s.line}`)
