// 最小 service worker：让 Chrome 认这是一个可安装的 PWA
// （没有它，"添加到主屏幕"只会创建普通书签快捷方式，不会全屏独立窗口）
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  // 纯网络优先，不做缓存（页面本身很轻，且生成结果必须实时）
});
