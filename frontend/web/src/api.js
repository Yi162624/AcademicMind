// API 客户端：统一封装对 FastAPI 后端的调用
// 作用：前端组件不直接写 fetch 细节，全走这里的函数；/api 由 Vite 代理到 localhost:8000

async function _post(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`请求失败 ${resp.status}：${text || resp.statusText}`)
  }
  return resp.json()
}

// 启动任务：mode=调研/论文；question=调研问题；papers=[{title,link}]
export function startTask({ mode, question, papers, threadId }) {
  return _post('/api/task/start', {
    mode,
    question: question || '',
    papers: papers || [],
    thread_id: threadId || null,
  })
}

// 恢复被挂起的任务：choice=accept/revise/continue/cancel
export function resumeTask({ threadId, choice, feedback }) {
  return _post('/api/task/resume', {
    thread_id: threadId,
    choice,
    feedback: feedback || '',
  })
}

// 读模式偏好
export async function getConfig() {
  const resp = await fetch('/api/config')
  return resp.json()
}

// 存模式偏好
export function saveConfig(mode) {
  return _post('/api/config', { mode })
}

// 上传 PDF：multipart 表单，返回 {title, link}
export async function uploadPdf(file) {
  const fd = new FormData()
  fd.append('file', file)
  const resp = await fetch('/api/upload', { method: 'POST', body: fd })
  if (!resp.ok) {
    // 后端校验失败会带 detail（如"PDF 不完整/损坏"），解析出来给用户看具体原因
    const data = await resp.json().catch(() => null)
    throw new Error(data?.detail || `上传失败 ${resp.status}`)
  }
  return resp.json()
}

// 意图分类：判断输入走哪条路（simple 直接答 / research 走调研 / other 拒答）
export function classify(text) {
  return _post('/api/classify', { text })
}

// 简单问题直答：只调一次 LLM 直接回答，不走 5 Agent
export function simpleChat(text) {
  return _post('/api/chat/simple', { text })
}
