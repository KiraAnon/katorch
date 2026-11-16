/**
 * 样式 —— 只用宿主主题 token（--dsw-alias-*），不写死颜色。
 * 类名统一 kv- 前缀，避免和宿主/其它插件打架。
 */
export const CSS = `
.kv-root {
  display: flex; flex-direction: column; height: 100%; min-height: 0;
  background: var(--dsw-alias-bg-base); color: var(--dsw-alias-label-primary);
  font-size: 13px; position: relative;
}
.kv-toolbar {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 9px 12px; border-bottom: 1px solid var(--dsw-alias-border-l1);
  background: var(--dsw-alias-bg-layer-1); flex: none;
}
.kv-title { font-weight: 600; font-size: 13.5px; margin-right: 4px; white-space: nowrap; }
.kv-path {
  flex: 1 1 260px; min-width: 180px; box-sizing: border-box;
  background: var(--dsw-alias-bg-base); color: var(--dsw-alias-label-primary);
  border: 1px solid var(--dsw-alias-border-l1); border-radius: 7px;
  padding: 6px 9px; font-size: 12px; font-family: ui-monospace, Consolas, monospace;
}
.kv-path:focus { outline: none; border-color: var(--dsw-alias-brand-primary); }
.kv-btn {
  appearance: none; cursor: pointer; border-radius: 7px; padding: 6px 11px; font-size: 12px;
  border: 1px solid var(--dsw-alias-border-l1); background: var(--dsw-alias-bg-base);
  color: var(--dsw-alias-label-primary); white-space: nowrap;
}
.kv-btn:hover:not(:disabled) { background: var(--dsw-alias-bg-layer-2); border-color: var(--dsw-alias-border-l2); }
.kv-btn:disabled { opacity: .45; cursor: default; }
.kv-btn-primary { background: var(--dsw-alias-brand-primary); border-color: var(--dsw-alias-brand-primary); color: #fff; font-weight: 500; }
.kv-btn-primary:hover:not(:disabled) { filter: brightness(1.08); }
.kv-btn-danger { color: var(--dsw-alias-state-error-primary); border-color: var(--dsw-alias-state-error-primary); }
.kv-select {
  background: var(--dsw-alias-bg-base); color: var(--dsw-alias-label-primary);
  border: 1px solid var(--dsw-alias-border-l1); border-radius: 7px; padding: 6px 8px; font-size: 12px;
}
.kv-spacer { flex: 1 1 auto; }

.kv-body { display: flex; flex: 1 1 auto; min-height: 0; }
.kv-outline {
  width: 216px; flex: none; display: flex; flex-direction: column; min-height: 0;
  border-right: 1px solid var(--dsw-alias-border-l1); background: var(--dsw-alias-bg-layer-1);
}
.kv-side {
  width: 316px; flex: none; display: flex; flex-direction: column; min-height: 0;
  border-left: 1px solid var(--dsw-alias-border-l1); background: var(--dsw-alias-bg-layer-1);
}
.kv-center { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; min-height: 0; }

.kv-panel-head {
  display: flex; align-items: center; gap: 6px; padding: 8px 11px; flex: none;
  border-bottom: 1px solid var(--dsw-alias-border-l1);
  font-size: 12px; font-weight: 600; color: var(--dsw-alias-label-secondary);
}
.kv-panel-body { overflow: auto; flex: 1 1 auto; min-height: 0; padding: 8px; }
.kv-panel-body.pad0 { padding: 0; }

.kv-group-row {
  display: flex; align-items: center; gap: 6px; padding: 6px 8px; border-radius: 7px;
  cursor: pointer; font-size: 12px; line-height: 1.35;
}
.kv-group-row:hover { background: var(--dsw-alias-bg-layer-2); }
.kv-group-row.active { background: var(--dsw-alias-bg-layer-2); box-shadow: inset 2px 0 0 var(--dsw-alias-brand-primary); }
.kv-group-name { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kv-group-meta { font-size: 10.5px; color: var(--dsw-alias-label-secondary); white-space: nowrap; }
.kv-caret { width: 12px; text-align: center; color: var(--dsw-alias-label-secondary); flex: none; }

.kv-canvas-wrap { position: relative; flex: 1 1 auto; min-height: 0; overflow: hidden; }
.kv-canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.kv-zoom-chip {
  position: absolute; right: 10px; bottom: 10px; font-size: 11px; padding: 3px 8px; border-radius: 999px;
  background: var(--dsw-alias-bg-layer-2); color: var(--dsw-alias-label-secondary);
  border: 1px solid var(--dsw-alias-border-l1); pointer-events: none;
}
.kv-canvas-hint {
  position: absolute; left: 10px; bottom: 10px; font-size: 11px; color: var(--dsw-alias-label-secondary);
  pointer-events: none; opacity: .75;
}

.kv-kv { display: grid; grid-template-columns: 88px 1fr; gap: 3px 8px; font-size: 12px; padding: 2px 2px 8px; }
.kv-kv dt { color: var(--dsw-alias-label-secondary); }
.kv-kv dd { margin: 0; word-break: break-all; font-family: ui-monospace, Consolas, monospace; font-size: 11.5px; }

.kv-sub { font-size: 11px; font-weight: 600; color: var(--dsw-alias-label-secondary); margin: 10px 0 5px; letter-spacing: .02em; }
.kv-sub:first-child { margin-top: 2px; }

.kv-tensor {
  display: flex; align-items: center; gap: 6px; padding: 4px 7px; border-radius: 6px;
  border: 1px solid var(--dsw-alias-border-l1); background: var(--dsw-alias-bg-base);
  font-size: 11.5px; margin-bottom: 4px; cursor: pointer;
}
.kv-tensor:hover { border-color: var(--dsw-alias-border-l2); }
.kv-tid { font-family: ui-monospace, Consolas, monospace; color: var(--dsw-alias-label-secondary); flex: none; min-width: 26px; }
.kv-tshape { flex: 1 1 auto; font-family: ui-monospace, Consolas, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kv-ttype { color: var(--dsw-alias-label-secondary); flex: none; font-size: 10.5px; }
.kv-tconst { color: var(--dsw-alias-brand-primary); flex: none; font-size: 10px; }

.kv-badge {
  display: inline-block; font-size: 10.5px; padding: 1px 6px; border-radius: 999px; flex: none;
  border: 1px solid var(--dsw-alias-border-l1); color: var(--dsw-alias-label-secondary);
}
.kv-badge.err { color: var(--dsw-alias-state-error-primary); border-color: var(--dsw-alias-state-error-primary); }
.kv-badge.warn { color: var(--dsw-alias-state-warn-primary); border-color: var(--dsw-alias-state-warn-primary); }
.kv-badge.ok { color: var(--dsw-alias-state-success-primary); border-color: var(--dsw-alias-state-success-primary); }

.kv-note {
  font-size: 11.5px; line-height: 1.5; padding: 7px 9px; border-radius: 7px; margin: 6px 0;
  border: 1px solid var(--dsw-alias-border-l1); color: var(--dsw-alias-label-secondary);
  background: var(--dsw-alias-bg-base);
}
.kv-note.err {
  border-color: var(--dsw-alias-state-error-primary); color: var(--dsw-alias-state-error-primary);
  background: color-mix(in srgb, var(--dsw-alias-state-error-primary) 8%, transparent);
}
.kv-note.warn {
  border-color: var(--dsw-alias-state-warn-primary); color: var(--dsw-alias-state-warn-primary);
  background: color-mix(in srgb, var(--dsw-alias-state-warn-primary) 8%, transparent);
}
.kv-note.ok {
  border-color: var(--dsw-alias-state-success-primary); color: var(--dsw-alias-state-success-primary);
  background: color-mix(in srgb, var(--dsw-alias-state-success-primary) 8%, transparent);
}

.kv-input {
  box-sizing: border-box; background: var(--dsw-alias-bg-base); color: var(--dsw-alias-label-primary);
  border: 1px solid var(--dsw-alias-border-l1); border-radius: 6px; padding: 4px 7px;
  font-size: 11.5px; font-family: ui-monospace, Consolas, monospace; width: 100%;
}
.kv-input:focus { outline: none; border-color: var(--dsw-alias-brand-primary); }

.kv-dims { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }
.kv-dims .kv-input { width: 58px; }

.kv-foot {
  flex: none; max-height: 168px; display: flex; flex-direction: column;
  border-top: 1px solid var(--dsw-alias-border-l1); background: var(--dsw-alias-bg-layer-1);
}
.kv-foot.collapsed { max-height: 33px; }
.kv-problem {
  display: flex; gap: 7px; align-items: flex-start; padding: 4px 10px; font-size: 11.5px;
  line-height: 1.45; cursor: pointer; border-radius: 5px;
}
.kv-problem:hover { background: var(--dsw-alias-bg-layer-2); }
.kv-problem .lv { flex: none; width: 14px; text-align: center; font-weight: 700; }
.kv-problem.err .lv { color: var(--dsw-alias-state-error-primary); }
.kv-problem.warn .lv { color: var(--dsw-alias-state-warn-primary); }
.kv-problem.info .lv { color: var(--dsw-alias-label-secondary); }

.kv-tabs { display: flex; gap: 2px; padding: 0 8px; border-bottom: 1px solid var(--dsw-alias-border-l1); flex: none; }
.kv-tab {
  appearance: none; background: none; border: none; cursor: pointer; padding: 7px 9px; font-size: 12px;
  color: var(--dsw-alias-label-secondary); border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.kv-tab.active { color: var(--dsw-alias-label-primary); border-bottom-color: var(--dsw-alias-brand-primary); }

.kv-bar { display: flex; align-items: center; gap: 6px; font-size: 11.5px; margin-bottom: 3px; }
.kv-bar-name { width: 96px; flex: none; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kv-bar-track { flex: 1 1 auto; height: 7px; border-radius: 999px; background: var(--dsw-alias-bg-layer-2); overflow: hidden; }
.kv-bar-fill { height: 100%; background: var(--dsw-alias-brand-primary); border-radius: 999px; }
.kv-bar-num { width: 34px; text-align: right; flex: none; color: var(--dsw-alias-label-secondary); }

.kv-menu {
  position: fixed; z-index: 60; min-width: 152px; padding: 4px;
  background: var(--dsw-alias-bg-layer-2); border: 1px solid var(--dsw-alias-border-l1);
  border-radius: 9px; box-shadow: 0 8px 28px rgba(0,0,0,.28);
}
.kv-menu button {
  display: block; width: 100%; text-align: left; appearance: none; background: none; border: none;
  color: var(--dsw-alias-label-primary); font-size: 12px; padding: 6px 9px; border-radius: 6px; cursor: pointer;
}
.kv-menu button:hover:not(:disabled) { background: var(--dsw-alias-bg-base); }
.kv-menu button:disabled { opacity: .4; cursor: default; }
.kv-menu .sep { height: 1px; background: var(--dsw-alias-border-l1); margin: 4px 2px; }

.kv-empty { padding: 22px 14px; text-align: center; color: var(--dsw-alias-label-secondary); font-size: 12px; line-height: 1.6; }

.kv-toast {
  position: absolute; left: 50%; top: 58px; transform: translateX(-50%); z-index: 50;
  padding: 7px 14px; border-radius: 9px; font-size: 12px; max-width: 78%;
  background: var(--dsw-alias-bg-layer-2); border: 1px solid var(--dsw-alias-border-l1);
  box-shadow: 0 8px 26px rgba(0,0,0,.24);
}
.kv-toast.err { border-color: var(--dsw-alias-state-error-primary); color: var(--dsw-alias-state-error-primary); }
.kv-toast.ok { border-color: var(--dsw-alias-state-success-primary); color: var(--dsw-alias-state-success-primary); }

.kv-dirty { color: var(--dsw-alias-state-warn-primary); font-size: 11.5px; white-space: nowrap; }
.kv-mono { font-family: ui-monospace, Consolas, monospace; }
.kv-muted { color: var(--dsw-alias-label-secondary); }
`
