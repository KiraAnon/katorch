/**
 * 侧栏与面板组件：大纲 / 检查器 / 统计 / 问题列表。
 * 全都是纯展示 + 回调，状态在 page.js 里。
 */
import React from 'react'
import {
  opName, typeName, typeShort, operandBytes, pitfalls, paramSlots, constInts, ARITY,
} from '../core/spec.js'
import { statsByGroups } from '../core/export.js'


const fmt = (n) => {
  if (n == null) return '—'
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${n} B`
}

// --------------------------------------------------------------------------- //
// 大纲：分组 + 组内算子
// --------------------------------------------------------------------------- //

export function Outline({ graph, groups, expanded, activeGroupId, onToggle, onFocusOp, onFocusGroup }) {
  if (!graph) {
    return React.createElement('div', { className: 'kv-empty' }, '还没打开图')
  }
  return React.createElement('div', null,
    groups.map((g) => {
      const open = expanded.has(g.id)
      const isActive = activeGroupId === g.id
      const inner = []
      inner.push(React.createElement('div', {
        key: 'g',
        className: `kv-group-row${isActive ? ' active' : ''}`,
        onClick: () => onFocusGroup(g),
      },
      React.createElement('span', {
        className: 'kv-caret',
        onClick: (ev) => { ev.stopPropagation(); onToggle(g.id) },
      }, open ? '▾' : '▸'),
      React.createElement('span', { className: 'kv-group-name' }, g.label),
      React.createElement('span', { className: 'kv-group-meta' }, `${g.end - g.start}`)))

      if (open) {
        for (let i = g.start; i < g.end; i += 1) {
          const op = graph.ops[i]
          if (!op) continue
          const out = graph.operandById.get(op.outs[0])
          inner.push(React.createElement('div', {
            key: `o${i}`,
            className: 'kv-group-row',
            style: { paddingLeft: 22, fontSize: 11.5 },
            onClick: () => onFocusOp(i),
          },
          React.createElement('span', { className: 'kv-group-name' }, `#${i} ${opName(op.opcode)}`),
          React.createElement('span', { className: 'kv-group-meta' },
            out ? `${out.dims.join('×')}` : '')))
        }
      }
      return inner
    }))
}

// --------------------------------------------------------------------------- //
// 检查器
// --------------------------------------------------------------------------- //

export function Inspector({ graph, sel, consts, dirtyConsts, onEditDims, onDeleteOp, onDuplicateOp, onEditConst, onFocusOperand }) {
  if (!graph || !sel) {
    return React.createElement('div', { className: 'kv-empty' },
      '点一个节点看它的输入输出、形状、常量路径。',
      React.createElement('br'),
      '双击折叠的组可以展开；滚轮缩放。')
  }

  if (sel.kind === 'group' || sel.kind === 'frame') {
    const g = sel.group
    let bytes = 0
    let ops = 0
    for (let i = g.start; i < g.end; i += 1) {
      const op = graph.ops[i]
      if (!op) continue
      ops += 1
      for (const oid of op.outs) {
        const el = graph.operandById.get(oid)
        if (el) bytes += operandBytes(el)
      }
    }
    return React.createElement('div', null,
      React.createElement('div', { className: 'kv-kv' },
        React.createElement('dt', null, '分组'), React.createElement('dd', null, g.label),
        React.createElement('dt', null, '算子'), React.createElement('dd', null, String(ops)),
        React.createElement('dt', null, '输出字节'), React.createElement('dd', null, fmt(bytes)),
        React.createElement('dt', null, '区间'), React.createElement('dd', null, `#${g.start} … #${g.end - 1}`)),
      React.createElement('div', { className: 'kv-note' }, '双击画布上的组节点可以展开/折叠。'))
  }

  const op = graph.ops[sel.opIndex]
  if (!op) return React.createElement('div', { className: 'kv-empty' }, '算子不存在')

  const warns = pitfalls(op, graph)
  const slots = paramSlots(op.opcode)
  const want = ARITY[op.opcode]

  const constBytes = () => {
    let sum = 0
    for (const oid of op.ins) {
      const el = graph.operandById.get(oid)
      if (!el) continue
      if (el.tailKind === 'w') sum += el.tailLen != null ? el.tailLen : operandBytes(el)
      else if (el.tailKind === 'a') sum += 4
    }
    for (const oid of op.outs) {
      const el = graph.operandById.get(oid)
      if (el) sum += 0
    }
    return sum
  }

  const tensorRow = (oid, role) => {
    const el = graph.operandById.get(oid)
    if (!el) {
      return React.createElement('div', { key: `${role}${oid}`, className: 'kv-tensor' },
        React.createElement('span', { className: 'kv-tid' }, String(oid)),
        React.createElement('span', { className: 'kv-tshape', style: { color: 'var(--dsw-alias-state-error-primary)' } }, '未声明'))
    }
    const isParam = role === 'in' && slots.includes(op.ins.indexOf(oid))
    return React.createElement('div', {
      key: `${role}${oid}`,
      className: 'kv-tensor',
      onClick: () => onFocusOperand && onFocusOperand(el),
      title: el.tailPath || '',
    },
    React.createElement('span', { className: 'kv-tid' }, String(oid)),
    React.createElement('span', { className: 'kv-tshape' }, `${el.dims.join('×') || 'scalar'}`),
    React.createElement('span', { className: 'kv-ttype' }, typeShort(el.type)),
    el.tailKind === 'w'
      ? React.createElement('span', { className: 'kv-tconst' }, isParam ? '参数' : '常量')
      : null,
    el.tailKind === 'a'
      ? React.createElement('span', { className: 'kv-tconst' }, `a ${el.tailValue}`)
      : null,
    isParam && dirtyConsts && dirtyConsts.has(oid)
      ? React.createElement('span', { className: 'kv-badge warn' }, '改')
      : null)
  }

  return React.createElement('div', null,
    React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 } },
      React.createElement('strong', { style: { fontSize: 13 } }, `#${op.index} ${opName(op.opcode)}`),
      React.createElement('span', { className: 'kv-badge' }, `op ${op.opcode}`),
      want != null && op.ins.length !== want
        ? React.createElement('span', { className: 'kv-badge warn' }, `输入 ${op.ins.length}/${want}`)
        : null),

    warns.length
      ? warns.map((w, i) => React.createElement('div', {
        key: `w${i}`,
        className: `kv-note ${w.level === 'error' ? 'err' : 'warn'}`,
      }, w.text))
      : null,

    React.createElement('div', { className: 'kv-sub' }, `输入 · ${op.ins.length}`),
    op.ins.map((oid) => tensorRow(oid, 'in')),

    React.createElement('div', { className: 'kv-sub' }, `输出 · ${op.outs.length}`),
    op.outs.map((oid) => tensorRow(oid, 'out')),

    React.createElement('div', { className: 'kv-sub' }, '常量占用'),
    React.createElement('div', { className: 'kv-mono kv-muted', style: { fontSize: 11.5 } }, fmt(constBytes())),

    op.outs.length === 1
      ? React.createElement(OutputShapeEditor, {
        graph,
        oid: op.outs[0],
        onEditDims,
      })
      : null,

    slots.length
      ? React.createElement(ParamEditor, {
        graph, op, slots, consts, onEditConst,
      })
      : null,

    React.createElement('div', { className: 'kv-sub' }, '操作'),
    React.createElement('div', { style: { display: 'flex', gap: 6, flexWrap: 'wrap' } },
      React.createElement('button', {
        className: 'kv-btn',
        onClick: () => onDuplicateOp(op.index),
      }, '复制算子'),
      React.createElement('button', {
        className: 'kv-btn kv-btn-danger',
        onClick: () => onDeleteOp(op.index),
      }, '删除算子')))
}

/** 输出形状：改了就触发下游重推 */
function OutputShapeEditor({ graph, oid, onEditDims }) {
  const el = graph.operandById.get(oid)
  const [text, setText] = React.useState(el ? el.dims.join(' ') : '')
  React.useEffect(() => { setText(el ? el.dims.join(' ') : '') }, [oid, el ? el.dims.join(' ') : ''])
  if (!el) return null
  const commit = () => {
    const dims = text.trim().split(/\s+/).filter(Boolean).map((v) => parseInt(v, 10))
    if (!dims.length || dims.some((v) => !Number.isFinite(v) || v < 0)) {
      setText(el.dims.join(' '))
      return
    }
    onEditDims(oid, dims)
  }
  return React.createElement('div', null,
    React.createElement('div', { className: 'kv-sub' }, '输出形状（改了下游自动重推）'),
    React.createElement('div', { className: 'kv-dims' },
      React.createElement('input', {
        className: 'kv-input',
        style: { width: '100%' },
        value: text,
        onChange: (ev) => setText(ev.target.value),
        onBlur: commit,
        onKeyDown: (ev) => { if (ev.key === 'Enter') ev.currentTarget.blur() },
        placeholder: '例如 256 2048',
      })))
}

/** 参数常量（RESHAPE 的 shape / TRANSPOSE 的 perm / REDUCE 的 axis …） */
function ParamEditor({ graph, op, slots, consts, onEditConst }) {
  const rows = slots.map((slotIndex) => {
    const oid = op.ins[slotIndex]
    const el = graph.operandById.get(oid)
    if (!el) return null
    const ints = constInts(el, consts ? consts.get(oid) : null)
    const editable = Boolean(ints) && (el.tailKind === 'a' || el.tailKind === 'w')
    return { slotIndex, oid, el, ints, editable }
  }).filter(Boolean)

  if (!rows.length) return null

  return React.createElement('div', null,
    React.createElement('div', { className: 'kv-sub' }, '参数'),
    rows.map((r) => React.createElement(ParamRow, {
      key: r.oid,
      row: r,
      onEditConst,
    })))
}

function ParamRow({ row, onEditConst }) {
  const { oid, el, ints, editable } = row
  const [text, setText] = React.useState(ints ? ints.join(' ') : '')
  const label = opNameForParam(el)
  React.useEffect(() => {
    setText(ints ? ints.join(' ') : '')
  }, [oid, ints ? ints.join(' ') : ''])

  if (!ints) {
    return React.createElement('div', { className: 'kv-note' },
      `operand ${oid} 是常量（${el.tailKind === 'w' ? el.tailPath : `a ${el.tailValue}`}），但读不到内容 —— 形状推导会跳过这个算子。`)
  }

  const commit = () => {
    const next = text.trim().split(/[\s,]+/).filter(Boolean).map((v) => parseInt(v, 10))
    if (!next.length || next.some((v) => !Number.isFinite(v))) {
      setText(ints.join(' '))
      return
    }
    onEditConst(oid, next)
  }

  return React.createElement('div', { style: { marginBottom: 6 } },
    React.createElement('div', { className: 'kv-muted', style: { fontSize: 10.5, marginBottom: 2 } },
      `${label} · operand ${oid}${editable ? '' : '（只读）'}`),
    React.createElement('input', {
      className: 'kv-input',
      value: text,
      readOnly: !editable,
      onChange: (ev) => setText(ev.target.value),
      onBlur: commit,
      onKeyDown: (ev) => { if (ev.key === 'Enter') ev.currentTarget.blur() },
    }),
    el.tailKind === 'w'
      ? React.createElement('div', { className: 'kv-muted', style: { fontSize: 10, marginTop: 2, wordBreak: 'break-all' } },
        el.tailPath, el.tailOff != null ? ` @${el.tailOff}+${el.tailLen}` : '')
      : null)
}

function opNameForParam(el) {
  if (!el) return '参数'
  return el.dims && el.dims.length ? `int32[${el.dims.join('×')}]` : 'int32'
}

// --------------------------------------------------------------------------- //
// 统计
// --------------------------------------------------------------------------- //

export function StatsView({ graph, st, groups }) {
  if (!graph || !st) return React.createElement('div', { className: 'kv-empty' }, '还没打开图')
  const byGroup = React.useMemo(() => statsByGroups(graph, groups), [graph, groups])
  const maxCount = st.byOpcode.length ? st.byOpcode[0].count : 1

  return React.createElement('div', null,
    React.createElement('div', { className: 'kv-kv' },
      React.createElement('dt', null, '算子'), React.createElement('dd', null, String(st.opCount)),
      React.createElement('dt', null, 'operand'), React.createElement('dd', null, String(st.operandCount)),
      React.createElement('dt', null, '常量个数'), React.createElement('dd', null, String(st.constOperands)),
      React.createElement('dt', null, '常量字节'), React.createElement('dd', null, fmt(st.constBytes)),
      React.createElement('dt', null, '张量字节'), React.createElement('dd', null, fmt(st.operandBytes)),
      React.createElement('dt', null, '常量文件'), React.createElement('dd', null, String(st.uniqueConstPaths))),

    React.createElement('div', { className: 'kv-sub' }, 'opcode 分布'),
    st.byOpcode.map((row) => React.createElement('div', { key: row.code, className: 'kv-bar' },
      React.createElement('span', { className: 'kv-bar-name', title: `op ${row.code}` }, row.name),
      React.createElement('span', { className: 'kv-bar-track' },
        React.createElement('span', {
          className: 'kv-bar-fill',
          style: { width: `${Math.max(2, (row.count / maxCount) * 100)}%` },
        })),
      React.createElement('span', { className: 'kv-bar-num' }, String(row.count)))),

    React.createElement('div', { className: 'kv-sub' }, '按分组'),
    React.createElement('table', { style: { width: '100%', fontSize: 11.5, borderCollapse: 'collapse' } },
      React.createElement('thead', null,
        React.createElement('tr', { style: { color: 'var(--dsw-alias-label-secondary)' } },
          React.createElement('th', { style: { textAlign: 'left', padding: '3px 4px' } }, '组'),
          React.createElement('th', { style: { textAlign: 'right', padding: '3px 4px' } }, '算子'),
          React.createElement('th', { style: { textAlign: 'right', padding: '3px 4px' } }, '字节'))),
      React.createElement('tbody', null,
        byGroup.map((g) => React.createElement('tr', { key: g.id },
          React.createElement('td', { style: { padding: '3px 4px' } },
            `${g.id + 1}. #${g.start}–${g.end - 1}`),
          React.createElement('td', { style: { textAlign: 'right', padding: '3px 4px' } }, String(g.opCount)),
          React.createElement('td', { style: { textAlign: 'right', padding: '3px 4px' } }, fmt(g.bytes)))))))
}

// --------------------------------------------------------------------------- //
// 问题列表
// --------------------------------------------------------------------------- //

export function ProblemsView({ problems, onFocusProblem }) {
  if (!problems || !problems.length) {
    return React.createElement('div', { className: 'kv-empty' },
      React.createElement('span', { className: 'kv-badge ok' }, '自检通过'),
      ' 结构、id 连续性、形状推导都没问题。')
  }
  return React.createElement('div', null,
    problems.map((p, i) => React.createElement('div', {
      key: `${i}`,
      className: `kv-problem ${p.level === 'error' ? 'err' : (p.level === 'warn' ? 'warn' : 'info')}`,
      onClick: () => onFocusProblem && onFocusProblem(p),
    },
    React.createElement('span', { className: 'lv' }, p.level === 'error' ? '✕' : (p.level === 'warn' ? '!' : 'i')),
    React.createElement('span', null, p.text))))
}
