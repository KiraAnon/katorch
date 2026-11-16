// 打印每行结束时的括号深度，用来找 render 表达式里被过早闭合的括号
import fs from 'node:fs'

const file = process.argv[2]
const fromLine = Number(process.argv[3] || 1)
const toLine = Number(process.argv[4] || 99999)
const lines = fs.readFileSync(file, 'utf8').split('\n')

let depth = 0
const out = []

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
    if (c === '(' || c === '[' || c === '{') depth += 1
    else if (c === ')' || c === ']' || c === '}') depth -= 1
    i += 1
  }
  if (n + 1 >= fromLine && n + 1 <= toLine) {
    out.push(`${String(n + 1).padStart(4)} d=${String(depth).padStart(3)}  ${lines[n].trim().slice(0, 96)}`)
  }
}
console.log(out.join('\n'))
