# KAtorch 可视化编译器

用节点图**看见并修改** KAtorch 编译器 lowering 出来的 NPU 图（`g.txt`），导出的结果直接交给
`host.c` 编译。宿主是 DSH 的一个插件，在 Web 界面里打开。

需求来源：`C:\Users\Administrator\Downloads\KAtorch可视化编辑器-需求清单.md`

---

## 1. 怎么打开

装好之后，左侧栏会多一个图标 **「KAtorch 图」**，点开就是编辑器。

顶部路径框填 `g.txt` 的绝对路径，点「打开」。默认预填：

```
F:\KAtorch\PC编译\wsl-sysroot\n1\g.txt
```

想知道这个路径从哪来、旁边还有哪些图：点「浏览」，它列目录里所有 `.txt`。

> **这个插件需要重启 DSH 才会出现在侧栏。** Host 半的插件行是启动时从
> `cordis.patch.yml` 读的，本次会话里改完配置不会热生效。

---

## 2. 界面

```
┌─ 工具栏：路径 / 打开 / 浏览 / 自检 / 导出 / 另存为 / 下载 / 分组 / 展开折叠 ─┐
├────────────┬──────────────────────────────────────┬──────────────────────┤
│ 大纲        │              画布                     │ 检查器 / 统计         │
│ 每层一组    │  节点 = 算子，连线 = operand 引用         │ 选中节点的全部细节     │
├────────────┴──────────────────────────────────────┴──────────────────────┤
│ 问题面板：结构错误 / id 不连续 / 形状推导不一致                             │
└──────────────────────────────────────────────────────────────────────────┘
```

**画布操作**

| 操作 | 效果 |
|---|---|
| 滚轮 | 以光标为中心缩放 |
| 拖拽空白 | 平移 |
| 点节点 | 选中，右栏显示细节 |
| 双击组节点 | 展开 / 折叠这一层 |
| 右键 | 定位 / 复制 / 删除 |

**分组**：默认「自动按层」——把算子序列当字符串做周期检测，找出重复的一层。
`n1/g.txt`（284 个算子）检不出周期就整体一组；`b00/g.txt`（569 个算子）会按 284 分开。
检不准就切「按算子数」，自己填每层多少个；或者「不分组」全摊开。

**节点显示**：`opcode 名称` + `输出形状 · 类型`（例：`RESHAPE` / `256×2048 · f16`）。

### 切到别的面板再回来，状态还在

「KAtorch 图」是主面板的一个 key，点侧栏「对话」时这个面板会被 React 卸载。
所以图、展开状态、选中、缩放位置都放在**模块级**而不是组件 state 里，切回来能接上：

- 内存里：图本身、常量数据、展开的组、选中节点、画布的缩放/平移、未导出的改动
- `localStorage`（刷新页面也在）：打开的路径、分组方式、每层算子数、展开的组、右栏标签页

副作用是**改动不会因为切走而丢**——没点「导出」的改动仍然留在面板里，工具栏会一直显示「有未导出的改动」。

---

## 3. 能改什么

选中一个算子，右栏可以：

- **改输出形状** —— 输入框里按 `256 2048` 这样写，回车提交。改完自动往下游重推导，
  推不动的节点会在问题面板里点名（比如「FC: x[-1]=8 != w[-1]=4」）。
- **改参数常量** —— `RESHAPE` 的目标形状、`TRANSPOSE` 的 perm、`REDUCE` 的 axis、`CONCAT` 的 axis
  这些，检查器会把常量数组摊成一行数字让你改。内联的 `a` 常量直接改文本；`w` 指向 `.bin` 的，
  导出时会写回那个文件（先留一份 `.bak-<时间戳>`）。
- **删除算子** —— 它的独占输出张量一起删，后面所有 operand id 自动重新连续编号。
- **复制算子** —— 新增的 OPD 会插到所有 OP 行之前（host.c 是流式解析，OP 引用时 operand 必须已声明）。

> 提示：`RESHAPE` 数量多不是问题（编译期会被消除），编辑器不会提示你「优化」它。

---

## 4. 怎么导出 / 怎么验

**导出**（工具栏）会做三件事：

1. 把改过的参数常量写回 `.bin`（有备份）
2. **强制重编号** operand —— 这是硬要求，`host.c` 的 `M_addOp` 按 OPD 出现顺序编号、
   `M_setVal` 用文件里写的 id，两者不一致编译返回 `-304`
3. 写盘前再自检一遍，**有错误就不写**，避免把坏图交出去

**自检**按钮跑三件事（全在浏览器里，不需要服务器）：

- `host.c` 的结构等价检查：id 连续性、OP 引用的 operand 存不存在、一个 operand 是否被多个算子写
- 形状推导：每个算子按上游形状重算输出，跟 OPD 里声明的比。对不上就报
  「声明 [H,T,T] != 推导 [H,T,1] —— adapter 会静默出错」
- 需求 §10 的坑：FC 的 act 写成 type 1、CONV 的 pad/stride 不是 type 1 + nd=0、
  fp16 走原生 SOFTMAX

**手机端编译验证**（本机没有 NPU 适配器库，只能上真机）：

```bash
# 在 Termux 里，host 已编好
./tools/verify-on-phone.sh /data/local/tmp/e2e_dbg/b00/g.txt ./host
```

把它导出的 `g.txt` 喂给 host，脚本会先做结构预检（不依赖 NPU），再发
`{"cmd":"build"}` + `{"cmd":"compile"}`，判读返回码。

---

## 5. agent 工具

同一个数据面注册成了工具 `katorch_viz`，可以直接让 agent 干活：

```
katorch_viz action=roundtrip path=F:\KAtorch\...\g.txt   # 往返自检
katorch_viz action=analyze    text="<图文本>"             # 校验一段图
katorch_viz action=stats      path=...                   # 统计
katorch_viz action=export     path=... text=...           # 自检通过才写盘
katorch_viz action=list       path=F:\KAtorch\PC编译
```

UI 和工具走的是**同一份 Host 实现**，不存在两套逻辑。

---

## 6. 架构

```
core/            纯数据层，不依赖 React 也不依赖 Node
  spec.js        opcode/type/分类表 + 坑标注（唯一一份规格）
  parser.js      g.txt → Graph（保留每行原文，往返一致的基础）
  shape.js       形状推导（shape_infer.py 逐条移植）
  export.js      重编号 + 序列化 + 图操作 + 统计
ui/              渲染层，换框架不影响 core
  layout.js      分组检测 + 布局（纯函数，可单测）
  canvas.js      Canvas 2D 画布（视口裁剪 + 缩放平移）
  panels.js      大纲 / 检查器 / 统计 / 问题列表
  page.js        主页面 + apply()
  theme.js       样式（只用 --dsw-alias-* 主题 token）
  i18n.js        文案
index.js         Host 半：webServer 路由 + agent 工具
build.mjs        把 core/ + ui/ 拼成 client.js
client.js        产物（自动生成，别手改）
```

**往返一致性**的做法：每个元素都存了原始行文本 `raw` 和 `tokens`。导出时只要这行
没被改过、id 也没重编号，就原样吐出 `raw`。所以「导入 → 什么都不动 → 导出」
天然逐字节一致，不依赖我猜对数字的格式化方式。改过的行才重建。

**为什么自己拼 bundle**：core 同时被 Host（原生 ESM import）和 Client（浏览器）使用。
把 core 的代码复制一份到 client.js 会让两边漂移，所以 `build.mjs` 直接把同一批源文件
拼进 client.js 的作用域——界面显示的和导出的必然是同一套逻辑。

---

## 7. 验收结果

### 测试 1：往返一致 ✅

```
样图：F:\KAtorch\PC编译\wsl-sysroot\n1\g.txt（真实图，805 行 / 284 算子 / 519 operand）
PASS 导出长度 == 原文件长度
PASS 逐字节一致
PASS 二次往返仍一致
PASS CRLF 原样保留
PASS 注释与空行原样保留
PASS 缺 END 也能往返
```

需求 §8 点名的 `F:\KAtorch\测试\data\sample.txt` 在这台机器上不存在，用的是上面这份真实的
`n1/g.txt`（同样是 lowering 产物）。

### 测试 2：能编译 —— 本机能验的部分 ✅ / 真机待跑

本机**没有** MediaTek NPU 的适配器库，`host.c` 编不出来也跑不了，所以「build/compile 返回 rc=0」
这一步只能在 Termux 上验（脚本见上）。本机做了等价验证：

```
PASS 结构检查无错误
PASS operand id 是 0..N-1
PASS 所有 OP 行 token 数量自洽
PASS 有 IO 行（缺了 host.c 返回 -107）
PASS 形状校验无 error（14 处 FC 输出数错误已修）
```

外加**和 Python 权威实现对拍**——用 `F:\KAtorch\编译器\shape_infer.py` 的
`validate_txt()` 跑同一份图，结论一致：

```
Python validate_txt 报 0 条
PASS Python 也认为这份图没有形状错误
PASS Python 也认为错误的图有错
PASS JS 与 Python 在错误图上结论一致
```

### 测试 3：改一处，能编 ✅

```
PASS 改后的常量值读回来是 1
PASS 改一处后仍能原样往返
PASS 只改了一行
PASS 行数未变（只替换了一行内容）
PASS RESHAPE 输出跟着变成 [4,8]
PASS REDUCE_SUM 输出跟着变成 [4,1]
PASS 下游 FC 不兼容时被报出来
PASS 改完仍能往返
PASS 只改了受影响的行
```

改上游形状 → 下游重推 → 不兼容会被点名，是这个链路要的。

### 图操作

```
PASS 删除后 operand id 仍连续
PASS 复制后 operand id 仍连续
PASS 新 OPD 排在所有 OP 之前
PASS normalizeIds 对已连续的图是幂等的
```

### 全部

```
node test/all.mjs
  数据层：往返一致 + 图操作              32 通过 / 0 失败
  数据层：形状推导（与 Python 对拍）      25 通过 / 0 失败
  Host 半：HTTP 路由 + agent 工具        42 通过 / 0 失败
  Client 半：加载 + apply + 渲染         18 通过 / 0 失败
```

---

## 8. 已知限制

1. **本机跑不了编译**。Windows 上没有 `libneuron`，`{"cmd":"compile"}` 只能在手机上验。
   编辑器的「自检」覆盖了会导致编译失败的大部分原因（尤其 `-304`），但不是全部。
2. **常量文件读不到时形状推导会跳过**。图里写的是手机路径
   `/data/data/com.termux/files/home/npt/...`，本机没有这些 `.bin`。这时检查器会显示
   「读不到内容 —— 形状推导会跳过这个算子」，问题面板标 `info` 而不是 `error`。
   如果本机有一份拷贝，在 `cordis.patch.yml` 的 `constRewrite` 里配前缀映射，就能读到了。
3. **拖拽连线还没做**（需求里 P0 明确说不做）。改连接要用 agent 工具或直接改文本。
4. **子图模板 / 两版对比 / 从 PyTorch 导入**（P1/P2 的后半）没做。模板的存储格式和联动规则
   需求里没定死，先没猜。
5. **分组靠周期检测**，对不重复的图会退化成一整组，需要手动切「按算子数」。

---

## 9. 排障

**侧栏没有「KAtorch 图」图标**

1. 查 `{DSH_HOME}\profiles\web-desktop\cordis.patch.yml` 末尾。**新增插件必须写成 `- insert:`，
   不能直接写 `- id: katorch-viz`** —— 这个文件的顶层行语义是「按 id 覆盖已有行」，
   覆盖一个不存在的 id 会被**静默忽略**，不报错也不生效。第一次装就踩了这个坑：
   重启完什么都没发生。正确形态：

   ```yaml
   - insert:
       - id: katorch-viz
         name: '@local/katorch-viz'
         config:
           defaultPath: 'F:\KAtorch\PC编译\wsl-sysroot\n1\g.txt'
   ```

2. 确认 `package.json` 的 dependencies 里有 `"@local/katorch-viz": "link:F:/KAtorch/katorch-viz"`。
3. 确认 `node_modules\@local\katorch-viz` 是指向 `F:\KAtorch\katorch-viz` 的 junction。
4. 改完 patch 会热重载 Host 半；**Client 半（侧栏图标、面板）要刷新页面**才出现。

自查有没有真的加载（不用翻页面）：
- `katorch_viz` 工具出现在工具列表里 → Host 半活着
- Client 半看 `sidebar.panellist` 的 occupants 里有没有 `katorch-viz`

**装不上的时候**：`install_bundle` 曾经因为 `node_modules\dsh-meme` 被运行中的 DSH 占用
（pnpm rename 报 EPERM）而整体失败，这与本插件无关。当时的处理是走手工链接，
备份在 `package.json.bak-katorch` / `cordis.patch.yml.bak-katorch`。

**面板白屏 / 控制台报 `slot entry crashed`**

先 `node build.mjs` 重新生成 client.js（会做语法检查），再 `node test/all.mjs`。
真要回滚，把上节那两个 `.bak-katorch` 拷回去再重启。

**改过源码之后**

```
node build.mjs      # 重新生成 client.js
node test/all.mjs   # 跑全部测试
```
client.js 是产物，不重新生成的话界面上看到的还是旧的。
Host 半（index.js）改动要重启 DSH。

---

## 10. 文件清单

| 路径 | 说明 |
|---|---|
| `core/` `ui/` `index.js` | 源码 |
| `client.js` | 生成的浏览器产物 |
| `build.mjs` | 生成器（拼 core + ui） |
| `test/` | 四组测试 |
| `tools/verify-on-phone.sh` | 手机端 build/compile 验证 |
| `tools/install-manual.mjs` | 手工安装兜底（替代 install_bundle） |
| `tools/diag-*.mjs` | 排查脚本（括号平衡、行深度），改大段 JSX 式代码时用过 |
| `cordis.patch.yml` | bundle 的插件行 |
