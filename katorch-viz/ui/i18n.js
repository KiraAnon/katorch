/**
 * 文案。走宿主的 locale 服务，这样跟随界面语言。
 * key 很少，够用就行。
 */
const zh = {
  'nav': 'KAtorch 图',
  'title': 'KAtorch 可视化编译器',
  'path.ph': 'g.txt 的绝对路径，例如 F:\\KAtorch\\PC编译\\wsl-sysroot\\n1\\g.txt',
  'open': '打开',
  'browse': '浏览',
  'export': '导出',
  'save': '写回',
  'check': '自检',
  'group': '分组',
  'group.auto': '自动按层',
  'group.size': '按算子数',
  'group.none': '不分组',
  'group.size.ph': '每层算子数',
  'expandAll': '全部展开',
  'collapseAll': '全部折叠',
  'outline': '大纲',
  'inspect': '检查器',
  'stats': '统计',
  'problems': '问题',
  'noProblems': '自检通过',
  'dirty': '有未导出的改动',
  'empty.title': '还没有打开图',
  'empty.hint': '填 g.txt 路径后点「打开」，或从大纲里选一个组。',
  'menu.expand': '展开这一组',
  'menu.collapse': '折叠这一组',
  'menu.duplicate': '复制算子',
  'menu.delete': '删除算子',
  'menu.focus': '定位到画布中央',
  'needOpen': '先打开一个 g.txt',
  'exported': '已写回',
  'checked': '自检完成',
  'constsWritten': '常量文件已写回',
}

const en = {
  'nav': 'KAtorch Graph',
  'title': 'KAtorch Visual Compiler',
  'path.ph': 'absolute path to g.txt',
  'open': 'Open',
  'browse': 'Browse',
  'export': 'Export',
  'save': 'Write back',
  'check': 'Validate',
  'group': 'Groups',
  'group.auto': 'Auto by layer',
  'group.size': 'By op count',
  'group.none': 'No groups',
  'group.size.ph': 'ops per layer',
  'expandAll': 'Expand all',
  'collapseAll': 'Collapse all',
  'outline': 'Outline',
  'inspect': 'Inspector',
  'stats': 'Stats',
  'problems': 'Problems',
  'noProblems': 'Validation passed',
  'dirty': 'Unsaved changes',
  'empty.title': 'No graph loaded',
  'empty.hint': 'Enter a g.txt path and press Open.',
  'menu.expand': 'Expand group',
  'menu.collapse': 'Collapse group',
  'menu.duplicate': 'Duplicate op',
  'menu.delete': 'Delete op',
  'menu.focus': 'Center on canvas',
  'needOpen': 'Open a g.txt first',
  'exported': 'Written back',
  'checked': 'Validation done',
  'constsWritten': 'Const files written',
}

export const DICTS = { zh, en }

let translate = null

export function setTranslator(fn) {
  translate = fn
}

export function t(key) {
  if (!translate) return zh[key] || key
  const got = translate(key)
  return got == null || got === key ? (zh[key] || key) : got
}
