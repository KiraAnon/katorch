/**
 * KAtorch 图格式规格表 —— 唯一一份，parser / shape / export 都从这里取。
 *
 * 权威来源：F:\KAtorch\编译器\host.c 的 build_from_file() 与 shape_infer.py。
 * 改这里之前先回去看那两个文件，不要凭印象改。
 */

/** opcode -> 名称（shape_infer.py:NAMES + 算子清单里出现过的） */
export const OP_NAMES = {
  0: 'ADD',
  1: 'AVGPOOL',
  2: 'CONCAT',
  3: 'CONV_2D',
  6: 'DEQUANTIZE',
  9: 'FC',
  14: 'LOGISTIC',
  17: 'MAXPOOL',
  18: 'MUL',
  19: 'RELU',
  20: 'RELU1',
  21: 'RELU6',
  22: 'RESHAPE',
  25: 'SOFTMAX',
  28: 'TANH',
  30: 'DIV',
  31: 'MEAN',
  36: 'SUB',
  37: 'TRANSPOSE',
  38: 'ABS',
  45: 'CAST',
  49: 'EXP',
  60: 'LOG',
  64: 'LOG_SOFTMAX',
  65: 'MAXIMUM',
  66: 'MINIMUM',
  67: 'NEG',
  72: 'QUANTIZE',
  77: 'REDUCE_MAX',
  80: 'REDUCE_SUM',
  83: 'RSQRT',
  85: 'SIN',
  88: 'SQRT',
  102: 'BATCH_MATMUL',
}

export function opName(code) {
  return OP_NAMES[code] || `OP_${code}`
}

/** 分类：形状推导用（shape_infer.py 原样搬过来） */
export const UNARY = new Set([6, 14, 19, 20, 21, 25, 28, 38, 45, 49, 60, 64, 67, 72, 83, 85, 88])
export const BINARY = new Set([0, 18, 30, 36, 65, 66])
export const REDUCE = new Set([31, 77, 80])

export const ADD = 0
export const CONCAT = 2
export const CONV = 3
export const FC = 9
export const RESHAPE = 22
export const TRANSPOSE = 37
export const SOFTMAX = 25
export const BATCH_MATMUL = 102

/** operand type 码 -> 元信息 */
export const TYPES = {
  0: { name: 'FLOAT32', bytes: 4, tensor: false },
  1: { name: 'INT32', bytes: 4, tensor: false },
  3: { name: 'TENSOR_FLOAT32', bytes: 4, tensor: true },
  4: { name: 'TENSOR_INT32', bytes: 4, tensor: true },
  5: { name: 'QUANT8_ASYMM', bytes: 1, tensor: true },
  7: { name: 'TENSOR_QUANT16_SYMM', bytes: 2, tensor: true },
  8: { name: 'TENSOR_FLOAT16', bytes: 2, tensor: true },
  11: { name: 'QUANT8_SYMM_PER_CHANNEL', bytes: 1, tensor: true },
}

/** 短名（节点上显示用） */
export const TYPE_SHORT = {
  0: 'f32', 1: 'i32', 3: 'f32', 4: 'i32', 5: 'u8', 7: 'i16', 8: 'f16', 11: 'u8pc',
}

export function typeBytes(type) {
  const t = TYPES[type]
  return t ? t.bytes : 0
}

export function typeName(type) {
  const t = TYPES[type]
  return t ? t.name : `TYPE_${type}`
}

export function typeShort(type) {
  return TYPE_SHORT[type] || `t${type}`
}

/** 一个 operand 声明占多少字节（nelem * elem_size） */
export function operandBytes(el) {
  if (!el || !el.dims) return 0
  let n = 1
  for (const d of el.dims) n *= d
  return n * typeBytes(el.type)
}

/**
 * 常量（a / w）的 int32 值。形状推导时 axis / perm / shape 都从这儿来。
 * data 是外部读进来的 Buffer/Uint8Array（可选）。
 */
export function constInts(el, data) {
  if (!el) return null
  if (el.tailKind === 'a') return [el.tailValue]
  if (el.tailKind !== 'w') return null
  if (!data || data.length === 0 || data.length % 4 !== 0) return null
  const n = data.length / 4
  const out = new Array(n)
  const view = new DataView(data.buffer ? data.buffer : data, data.byteOffset || 0, data.byteLength)
  for (let i = 0; i < n; i += 1) out[i] = view.getInt32(i * 4, true)
  return out
}

/**
 * 该 opcode 的「参数 operand 下标」——这些位置的常量才需要读文件。
 * 返回 [] 表示不需要常量数据。
 */
export function paramSlots(opcode) {
  if (REDUCE.has(opcode)) return [1]
  if (opcode === TRANSPOSE) return [1]
  if (opcode === RESHAPE) return [1]
  if (opcode === CONCAT) return [2]
  if (opcode === BATCH_MATMUL) return [2, 3]
  if (opcode === CONV) return [3, 4, 5, 6, 7, 8]
  return []
}

/** 每个算子期望的输入个数（算子表里的「输入个数」，缺省 null = 可变） */
export const ARITY = {
  0: 3, 2: 3, 9: 4, 14: 1, 18: 3, 19: 1, 22: 2, 25: 2, 28: 1, 30: 3,
  31: 3, 36: 3, 37: 2, 38: 1, 45: 1, 49: 1, 60: 1, 64: 1, 65: 3, 66: 3,
  67: 1, 77: 3, 80: 3, 83: 1, 85: 1, 88: 1, 102: 4,
}

/**
 * 节点上的已知坑标注（需求 §10）。
 * 返回 [{level:'warn'|'error', text}]
 */
export function pitfalls(opEl, graph) {
  const out = []
  const code = opEl.opcode
  const ins = opEl.ins || []
  const get = (i) => (graph && graph.operandById ? graph.operandById.get(ins[i]) : undefined)

  const want = ARITY[code]
  if (want != null && ins.length !== want) {
    out.push({
      level: 'warn',
      text: `${opName(code)} 期望 ${want} 个输入，实际 ${ins.length} 个`,
    })
  }

  if (code === FC) {
    const act = get(3)
    // 依据：需求 §10「FC 的 act 写成 type 1 → 静默通过，结果全错」
    // 真实图（n1/g.txt 的 `OP 9 4 14 15 16 9 1 17`）里 act 是 OPD 9 = type 4 的 `a 0`。
    // 所以正解是 type 4（TENSOR_INT32），写成 type 1（INT32 标量）会静默出错。
    if (act && act.type === 1) {
      out.push({
        level: 'error',
        text: 'FC 的 act 是 type 1(INT32 标量)，应为 type 4(TENSOR_INT32) —— 编译会静默通过，结果全错',
      })
    }
  }

  if (code === CONV) {
    for (let i = 3; i <= 8; i += 1) {
      const p = get(i)
      if (!p) continue
      if (p.type !== 1) {
        out.push({ level: 'error', text: `CONV_2D 的 pad/stride(输入${i}) 必须是 type 1，现在是 type ${p.type}` })
      } else if (p.nd !== 0) {
        out.push({ level: 'error', text: `CONV_2D 的 pad/stride(输入${i}) 必须 nd=0，现在是 nd=${p.nd}` })
      }
    }
  }

  if (code === SOFTMAX) {
    const x = get(0)
    if (x && x.type === 8) {
      out.push({
        level: 'warn',
        text: '原生 SOFTMAX 吃 fp16 时输出恒为均匀分布 1/N —— 只有 int16 正确',
      })
    }
  }

  return out
}
