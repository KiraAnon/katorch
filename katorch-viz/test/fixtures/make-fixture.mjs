/**
 * 造一份带真实 .bin 常量的合成 g.txt，用来验证形状推导（往返测试里那 105 条
 * 「缺常量跳过」就是靠它补上覆盖的）。
 *
 * 图：
 *   x [1,4,8] --TRANSPOSE perm(0,2,1)--> [1,8,4]
 *             --RESHAPE  shape(8,4)----> [8,4]
 *             --REDUCE_SUM axis(1)-----> [8,1]
 *             --FC W[16,4] b[16] act=0-> [8,16]
 *
 * 跑法：node test/fixtures/make-fixture.mjs
 * 产出：test/fixtures/synthetic/{g.txt, perm.bin, shape.bin, axis.bin, w16x4.bin}
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const dir = path.join(here, 'synthetic')
fs.mkdirSync(dir, { recursive: true })

const int32 = (values) => {
  const b = Buffer.alloc(values.length * 4)
  values.forEach((v, i) => b.writeInt32LE(v, i * 4))
  return b
}

const fi = (name) => path.join(dir, name).replace(/\\/g, '/')

fs.writeFileSync(path.join(dir, 'perm.bin'), int32([0, 2, 1]))
fs.writeFileSync(path.join(dir, 'shape.bin'), int32([8, 4]))
fs.writeFileSync(path.join(dir, 'axis.bin'), int32([1]))
// FC 权重 [16,4] f16 —— 内容无所谓，形状推导不看内容
fs.writeFileSync(path.join(dir, 'w16x4.bin'), Buffer.alloc(16 * 4 * 2, 0))

const lines = [
  'OPD 0 8 3 1 4 8 0 0 i',
  `OPD 1 4 1 3 0 0 w ${fi('perm.bin')}`,
  'OPD 2 8 3 1 8 4 0 0 o',
  `OPD 3 4 1 2 0 0 w ${fi('shape.bin')}`,
  'OPD 4 8 2 8 4 0 0 o',
  `OPD 5 4 1 1 0 0 w ${fi('axis.bin')}`,
  'OPD 6 4 1 1 0 0 a 0',
  'OPD 7 8 2 8 1 0 0 o',
  `OPD 8 8 2 16 4 0 0 w ${fi('w16x4.bin')}`,
  'OPD 9 8 1 16 0 0 o',
  'OPD 10 4 1 1 0 0 a 0',
  'OPD 11 8 2 8 16 0 0 o',
  'IO 1 0 1 11',
  'OP 37 2 0 1 1 2',
  'OP 22 2 2 3 1 4',
  'OP 80 3 4 5 6 1 7',
  'OP 9 4 4 8 9 10 1 11',
  'END',
  '',
]

const gtxt = lines.join('\n')
fs.writeFileSync(path.join(dir, 'g.txt'), gtxt, 'utf8')
console.log(`已生成 ${dir}`)
console.log(`  g.txt  ${gtxt.length} 字节 / ${lines.length - 2} 条有效行`)
