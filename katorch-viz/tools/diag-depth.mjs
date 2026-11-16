// 找出函数体内部深度掉到 0 或负的位置（说明前面多了一个右括号）
import fs from 'node:fs'

const file = process.argv[2]
const fromLine = Number(process.argv[3] || 1)
const lines = fs.readFileSync(file, 'utf8').split('\n')

let depth = 0
const stack = []
const events = []

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
      stack.pop()
      depth -= 1
      if (depth <= 0 && n + 1 >= fromLine) {
        events.push({ line: n + 1, depth, text: s.trim().slice(0, 110) })
        if (events.length > 6) { n = lines.length; break }
      }
    }
    i += 1
  }
  if (events.length > 6) break
}

for (const e of events) console.log(`行 ${e.line} 深度=${e.depth}  ${e.text}`)
console.log(`最终深度 ${depth}`)
