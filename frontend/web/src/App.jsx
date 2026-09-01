// App 主组件：对话式研究助手
// 核心设计（对应需求）：
//   1. 双模式独立会话：调研/论文各存各的对话历史（conversations 按模式分开），互不串
//   2. 对话进行中锁定模式切换：running=true 时侧边栏按钮禁用
//   3. 对话历史不丢：消息存 React state（不整页刷新），后端跑完直接 append
//   4. HIL 交互：后端返回 interrupt 时渲染确认卡片，用户点按钮调 resume 接着跑
import { useEffect, useRef, useState } from 'react'
import { classify, getConfig, resumeTask, saveConfig, simpleChat, startTask, uploadPdf } from './api'

// 模式常量：和后端 core/schemas.py 的 MODE_SURVEY / MODE_PAPER 对齐
const SURVEY = 'survey'
const PAPER = 'paper'

// 空会话的初始状态：每个模式一份，互不干扰
function emptyConv() {
  return { messages: [], threadId: null, pending: null, running: false, papers: [] }
}

// 判断输入是不是"在给论文"：含链接或像 PDF 路径
function looksLikePaper(text) {
  const t = text.trim().toLowerCase()
  return t.startsWith('http://') || t.startsWith('https://') || t.includes('arxiv.org') || t.endsWith('.pdf')
}

// 把后端 interrupt/result 统一转成一条"AI 消息"塞进对话
function packAssistantMessage(resp) {
  if (resp.type === 'interrupt') {
    return { role: 'ai', kind: 'interrupt', stage: resp.stage, data: resp.data, threadId: resp.thread_id }
  }
  return { role: 'ai', kind: 'result', data: resp.data, threadId: resp.thread_id }
}

export default function App() {
  const [mode, setMode] = useState(SURVEY)
  // 双模式独立会话：{ survey: {...}, paper: {...} }，切模式只切"显示哪份"，历史都在
  const [convs, setConvs] = useState({ [SURVEY]: emptyConv(), [PAPER]: emptyConv() })
  const [input, setInput] = useState('')
  const chatEndRef = useRef(null)
  const fileRef = useRef(null)

  const conv = convs[mode]           // 当前模式的会话
  const running = conv.running

  // 进页面恢复上次用的模式
  useEffect(() => {
    getConfig().then(c => c.mode && setMode(c.mode)).catch(() => {})
  }, [])

  // 对话到底部自动滚动
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conv.messages, conv.running])

  // 更新当前模式会话的辅助函数（只动当前模式那份，不影响另一个模式）
  const patch = (p) => setConvs(prev => ({ ...prev, [mode]: { ...prev[mode], ...p } }))
  const pushMsg = (m) => setConvs(prev => ({ ...prev, [mode]: { ...prev[mode], messages: [...prev[mode].messages, m] } }))

  // 切换模式：运行中锁死（需求2），空闲时才允许切；切完持久化偏好
  const switchMode = (m) => {
    if (running || m === mode) return
    setMode(m)
    saveConfig(m).catch(() => {})
  }

  // 跑后端并把结果加进对话（启动和恢复共用）
  const runAndAppend = async (promise) => {
    try {
      const resp = await promise
      const msg = packAssistantMessage(resp)
      setConvs(prev => ({
        ...prev, [mode]: {
          ...prev[mode],
          messages: [...prev[mode].messages, msg],
          threadId: resp.thread_id,
          pending: resp.type === 'interrupt' ? msg : null,
          running: false,
        },
      }))
    } catch (e) {
      pushMsg({ role: 'ai', kind: 'error', text: String(e.message || e) })
      patch({ running: false })
    }
  }

  // 发送消息（启动任务）：先做意图分流
  //   simple   → AI 直接答（不走 5 Agent）
  //   research → 走调研/论文分析流程
  //   other    → 礼貌拒答，说明原因
  const send = async () => {
    const text = input.trim()
    if (!text && conv.papers.length === 0) return
    setInput('')

    if (mode === PAPER) {
      // 论文模式：把输入按行拆开，每行一个链接/标题，全部收集进论文列表
      const papers = [...conv.papers]
      // 收集本次输入里的链接（支持一次粘多行链接）
      const linkLines = text.split('\n').map(l => l.trim()).filter(Boolean)
      const newLinks = linkLines.filter(l => looksLikePaper(l))
      if (newLinks.length > 0) {
        for (const link of newLinks) {
          const p = { title: link.split('/').pop() || link, link }
          papers.push(p)
          pushMsg({ role: 'user', kind: 'text', text: `📎 ${p.title}` })
        }
        patch({ running: true, pending: null, papers: [] })
        runAndAppend(startTask({ mode, question: '', papers, threadId: conv.threadId }))
        return
      }
      // 输入里没有链接（可能是普通文字）：先分类再分流
      pushMsg({ role: 'user', kind: 'text', text })
      patch({ running: true })
      await handleClassified(text, mode)
      return
    }

    // 调研模式：先记录用户消息，再分流
    pushMsg({ role: 'user', kind: 'text', text })
    patch({ running: true })
    await handleClassified(text, mode)
  }

  // 意图分流：分类 → 按结果走 直接答 / 调研 / 拒答
  const handleClassified = async (text, m) => {
    try {
      const c = await classify(text)
      if (c.type === 'simple') {
        const resp = await simpleChat(text)
        pushMsg({ role: 'ai', kind: 'text', text: resp.answer })
        patch({ running: false })
        return
      }
      if (c.type === 'other') {
        pushMsg({ role: 'ai', kind: 'text',
          text: `抱歉，这个问题我暂时无法回答。\n\n我是学术研究助手，专注于领域调研和论文分析。您提的这个问题属于其他领域（${c.reason || '不在我的能力范围内'}），建议您使用专业的工具或咨询对应领域的专家。\n\n如果您有学术调研或论文分析的需求，随时告诉我。` })
        patch({ running: false })
        return
      }
      // research：走正经流程
      if (m === PAPER) {
        const papers = conv.papers
        if (papers.length === 0) {
          pushMsg({ role: 'ai', kind: 'text', text: '请先粘贴论文链接，或点右侧回形针上传 PDF。' })
          patch({ running: false })
          return
        }
        patch({ running: true, pending: null, papers: [] })
        runAndAppend(startTask({ mode: m, question: '', papers, threadId: conv.threadId }))
      } else {
        patch({ running: true, pending: null })
        runAndAppend(startTask({ mode: m, question: text, papers: [], threadId: conv.threadId }))
      }
    } catch (e) {
      pushMsg({ role: 'ai', kind: 'error', text: String(e.message || e) })
      patch({ running: false })
    }
  }

  // 上传 PDF：先传给后端存好，加入论文列表，提示用户
  const onUpload = async (e) => {
    const files = Array.from(e.target.files || [])
    e.target.value = ''
    if (!files.length) return
    for (const f of files) {
      try {
        const saved = await uploadPdf(f)
        setConvs(prev => ({ ...prev, [mode]: { ...prev[mode], papers: [...prev[mode].papers, saved] } }))
        pushMsg({ role: 'user', kind: 'text', text: `📎 上传论文：${saved.title}` })
      } catch (err) {
        pushMsg({ role: 'ai', kind: 'error', text: `上传失败：${err.message}` })
      }
    }
  }

  // HIL 确认：把用户决定喂回后端 resume（accept/revise/continue/cancel）
  const respond = (choice, feedback = '') => {
    patch({ running: true, pending: null })
    pushMsg({ role: 'user', kind: 'text', text: feedback ? `修改意见：${feedback}` : `已选择：${choice}` })
    runAndAppend(resumeTask({ threadId: conv.threadId, choice, feedback }))
  }

  const removePaper = (i) => {
    setConvs(prev => ({ ...prev, [mode]: { ...prev[mode], papers: prev[mode].papers.filter((_, x) => x !== i) } }))
  }

  return (
    <div className="app">
      {/* 侧边栏：logo + 双模式导航（运行中禁用切换） */}
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-mark">🎓</div>
          <span>AcademicMind</span>
        </div>
        <div className="nav-label">研究模式</div>
        <button className={`nav-item ${mode === SURVEY ? 'active' : ''}`} disabled={running}
          onClick={() => switchMode(SURVEY)} title={running ? '研究进行中，暂不能切换' : ''}>
          <span className="nav-icon">🔍</span> 领域调研
        </button>
        <button className={`nav-item ${mode === PAPER ? 'active' : ''}`} disabled={running}
          onClick={() => switchMode(PAPER)} title={running ? '研究进行中，暂不能切换' : ''}>
          <span className="nav-icon">📄</span> 论文分析
        </button>
        <div className="sidebar-footer">
          多智能体深度研究助手<br />调研 / 论文双模式 · 人类在环确认
        </div>
      </aside>

      {/* 主区：顶栏 + 对话流 + 输入区 */}
      <main className="main">
        <div className="topbar">
          <span className="topbar-title">{mode === SURVEY ? '领域调研' : '论文分析'}</span>
          <span className="topbar-mode">{mode === SURVEY ? '输入研究问题，生成调研报告' : '输入论文，生成精读/对比报告'}</span>
        </div>

        <div className="chat"><div className="chat-inner">
          {conv.messages.length === 0 && !running && (
            <div className="empty">
              <div className="empty-icon">{mode === SURVEY ? '🔍' : '📄'}</div>
              <h2>{mode === SURVEY ? '开始一次领域调研' : '分析你的论文'}</h2>
              <p>{mode === SURVEY
                ? '输入一个研究问题（如"大模型在医疗影像的应用"），我会拆解成大纲让你确认，然后自动搜集证据、生成图文报告和研究方向建议。'
                : '粘贴论文链接或上传 PDF：1 篇走精读报告，多篇走对比综述。全程会在关键步骤请你确认。'}</p>
            </div>
          )}

          {conv.messages.map((m, i) => (
            <Message key={i} msg={m}
              // 只有"最后一条且当前待确认"的消息才可交互，历史确认只读
              interactive={m === conv.pending}
              onRespond={respond} />
          ))}

          {running && (
            <div className="msg ai">
              <div className="avatar"><AiAvatar /></div>
              <div className="bubble thinking">
                <div className="dots"><span /><span /><span /></div>
                AI 正在研究…（多 Agent 并行工作，可能要几十秒到几分钟）
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div></div>

        {/* 输入区：论文模式带 PDF 上传和待分析列表 */}
        <div className="input-area"><div className="input-inner">
          {mode === PAPER && conv.papers.length > 0 && (
            <div className="paper-chips">
              {conv.papers.map((p, i) => (
                <span className="chip" key={i}>📎 {p.title}
                  <button onClick={() => removePaper(i)} title="移除">✕</button>
                </span>
              ))}
            </div>
          )}
          <div className="input-row">
            {mode === PAPER && (
              <>
                <input type="file" accept="application/pdf" multiple hidden ref={fileRef} onChange={onUpload} />
                <button className="icon-btn" title="上传 PDF" onClick={() => fileRef.current?.click()} disabled={running}>📎</button>
              </>
            )}
            <textarea
              rows={1}
              value={input}
              placeholder={mode === SURVEY ? '输入研究问题，回车发送…' : '粘贴论文链接（可多篇），回车开始分析…'}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
              disabled={running}
            />
            <button className="icon-btn send" onClick={send}
              disabled={running || (!input.trim() && conv.papers.length === 0)} title="发送">➤</button>
          </div>
          <div className="input-hint">
            {running ? '研究进行中，请稍候…' : (conv.pending ? '请先在上方完成确认' : 'Enter 发送 · Shift+Enter 换行')}
          </div>
        </div></div>
      </main>
    </div>
  )
}

// 单条消息渲染：按 kind 分发（文本/错误/结果/确认卡片）
function Message({ msg, interactive, onRespond }) {
  if (msg.role === 'user') {
    return (
      <div className="msg user">
        <div className="avatar"><UserAvatar /></div>
        <div className="bubble">{msg.text}</div>
      </div>
    )
  }
  return (
    <div className="msg ai">
      <div className="avatar"><AiAvatar /></div>
      <div className="bubble"><AiContent msg={msg} interactive={interactive} onRespond={onRespond} /></div>
    </div>
  )
}

// AI 消息内容：按类型渲染
function AiContent({ msg, interactive, onRespond }) {
  const [fb, setFb] = useState('')
  if (msg.kind === 'error') {
    return <div>⚠️ 任务出错：{msg.text}<br /><small>请检查 .env 的 API Key 与网络。</small></div>
  }
  if (msg.kind === 'text') return <div>{msg.text}</div>

  if (msg.kind === 'interrupt') return <InterruptCard msg={msg} interactive={interactive} onRespond={onRespond} fb={fb} setFb={setFb} />

  if (msg.kind === 'result') return <ResultView data={msg.data} />
  return null
}

// HIL 确认卡片：相关性提示 / 大纲确认 / 报告复核
function InterruptCard({ msg, interactive, onRespond, fb, setFb }) {
  const { stage, data } = msg
  if (stage === 'relevance') {
    return (
      <div>
        <h3>⚠️ 论文相关性较低</h3>
        <p>{data.note}</p>
        {interactive && (
          <div className="btn-row">
            <button className="btn primary" onClick={() => onRespond('continue')}>继续生成（仅供参考）</button>
            <button className="btn ghost-danger" onClick={() => onRespond('cancel')}>取消，分开分析</button>
          </div>
        )}
      </div>
    )
  }
  if (stage === 'outline') {
    const o = data.outline || {}
    return (
      <div>
        <h3>📋 请确认研究大纲</h3>
        {o.subtopics?.length > 0 && <p><b>子主题：</b>{o.subtopics.join('、')}</p>}
        {o.sections?.length > 0 && <p><b>章节：</b>{o.sections.join(' → ')}</p>}
        {o.analysis_dimensions?.length > 0 && <p><b>分析维度：</b>{o.analysis_dimensions.join('、')}</p>}
        {o.research_tasks?.length > 0 && (
          <div>
            <b>研究任务单：</b>
            <ul>{o.research_tasks.map(t => <li key={t.id}><b>{t.id}</b> {t.question}</li>)}</ul>
          </div>
        )}
        {interactive && (
          <div className="confirm-card">
            <div className="btn-row">
              <button className="btn primary" onClick={() => onRespond('accept')}>✅ 接受并开始研究</button>
            </div>
            <input className="feedback-input" placeholder="或输入修改意见（如：增加一个关于 XX 的子主题）"
              value={fb} onChange={e => setFb(e.target.value)} />
            <div className="btn-row">
              <button className="btn" onClick={() => onRespond('revise', fb)}>↩️ 按意见调整大纲</button>
            </div>
          </div>
        )}
      </div>
    )
  }
  if (stage === 'report') {
    return (
      <div>
        <h3>⚠️ 报告未完全通过质量检查</h3>
        {(data.issues || []).map((it, i) => (
          <p className="issue" key={i}>[{it.level}] {it.message}{it.location ? `（${it.location}）` : ''}</p>
        ))}
        {data.feedback && <p><i>验证员意见：{data.feedback}</i></p>}
        {interactive && (
          <div className="confirm-card">
            <div className="btn-row">
              <button className="btn primary" onClick={() => onRespond('accept')}>✅ 接受当前版本并继续</button>
            </div>
            <input className="feedback-input" placeholder="或输入修改意见（如：第2章引用不准确）"
              value={fb} onChange={e => setFb(e.target.value)} />
            <div className="btn-row">
              <button className="btn" onClick={() => onRespond('revise', fb)}>↩️ 按意见修改报告</button>
            </div>
          </div>
        )}
      </div>
    )
  }
  return <div>待确认：{stage}</div>
}

// 最终结果渲染：报告(HTML) / 建议 / 精读(Markdown)
function ResultView({ data }) {
  if (!data) return null
  const warnings = data.warning_flags || []
  return (
    <div>
      {warnings.map((w, i) => <p key={i} className="issue">⚠️ {w}</p>)}

      {/* 调研/对比报告：后端给的是完整 HTML，用 iframe + srcDoc 渲染 */}
      {data.report?.html && (
        <div>
          <h3>📄 研究报告</h3>
          <iframe className="report-frame" title="report" srcDoc={data.report.html} />
        </div>
      )}

      {/* 行动建议 */}
      {data.suggestion && <SuggestionView sug={data.suggestion} />}

      {/* 单篇精读报告：Markdown 直接当文本块（轻量，不引 md 解析器） */}
      {data.deep_read && (
        <div>
          <h3>📖 论文精读报告</h3>
          {data.deep_read.one_line_summary && <p><b>一句话看懂：</b>{data.deep_read.one_line_summary}</p>}
          <Markdown text={data.deep_read.full_report || ''} />
        </div>
      )}
    </div>
  )
}

// 建议渲染：方向 + 论文 + 行动
function SuggestionView({ sug }) {
  return (
    <div>
      {sug.directions?.length > 0 && (
        <div><h3>💡 研究方向</h3><ol>{sug.directions.map((d, i) => <li key={i}>{d}</li>)}</ol></div>
      )}
      {sug.papers?.length > 0 && (
        <div><h3>📚 推荐论文</h3><ul>
          {sug.papers.map((p, i) => (
            <li key={i}><b>{p.title}</b>{p.reason ? ` — ${p.reason}` : ''}
              {p.link && <>（<a href={p.link} target="_blank" rel="noreferrer">链接</a>）</>}
            </li>
          ))}
        </ul></div>
      )}
      {sug.actions?.length > 0 && (
        <div><h3>✅ 下一步行动</h3><ul>{sug.actions.map((a, i) => <li key={i}>{a}</li>)}</ul></div>
      )}
    </div>
  )
}

// 极简 Markdown 渲染：把 # 标题/列表/粗体转成 HTML（精读报告用，不引第三方解析器）
function Markdown({ text }) {
  const html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/^###### (.*)$/gm, '<h6>$1</h6>')
    .replace(/^##### (.*)$/gm, '<h5>$1</h5>')
    .replace(/^#### (.*)$/gm, '<h4>$1</h4>')
    .replace(/^### (.*)$/gm, '<h3>$1</h3>')
    .replace(/^## (.*)$/gm, '<h3>$1</h3>')
    .replace(/^# (.*)$/gm, '<h3>$1</h3>')
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/^[-*] (.*)$/gm, '<li>$1</li>')
    .replace(/\n{2,}/g, '<br/><br/>')
  return <div dangerouslySetInnerHTML={{ __html: html }} />
}

// 用户头像：内联 SVG 人物剪影，比 emoji 更精致、跨平台显示一致
function UserAvatar() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  )
}

// AI 头像：学士帽 SVG（呼应学术研究助手的定位）
function AiAvatar() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M22 9 12 4 2 9l10 5 10-5z" />
      <path d="M6 11.5V16c0 1.5 2.7 3 6 3s6-1.5 6-3v-4.5" />
      <path d="M22 9v6" />
    </svg>
  )
}
