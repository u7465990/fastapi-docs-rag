import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Empty means same-origin `/api`, which works behind nginx and Vite's proxy.
// Capacitor/mobile builds can set VITE_API_BASE_URL to the public API origin.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
// Optional shared demo token; this is visible in the bundled JS, so it is a
// gate for demos, not real user authentication.
const API_TOKEN = import.meta.env.VITE_API_TOKEN ?? "";

type Source = {
  index: number;
  title: string;
  url: string;
  section_path: string[];
  anchor: string | null;
  docs_version: string;
  is_deprecated: boolean;
  deprecation_hint: boolean;
  snippet: string;
};

const EXAMPLES = [
  "FastAPI 里 on_event 怎么迁移到 lifespan？",
  "Depends 和 Annotated 怎么一起用？",
  "怎么定义带路径参数和查询参数的 GET 接口？",
  "BackgroundTasks 怎么用？",
  "Pydantic v1 的 validator 在 v2 怎么写？",
];

export default function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function ask(question = query) {
    const text = question.trim();
    if (!text || loading) return;
    setQuery(text);
    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (API_TOKEN) {
        headers.Authorization = `Bearer ${API_TOKEN}`;
      }
      const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify({ query: text, limit: 6 }),
      });
      if (!response.ok) {
        const body = await response.text();
        throw new Error(`HTTP ${response.status}: ${body.slice(0, 300)}`);
      }
      if (!response.body) {
        throw new Error("streaming response has no body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = buffer.replace(/\r\n/g, "\n");
        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          let event = "message";
          let data = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (data) {
            const payload = JSON.parse(data) as Record<string, unknown>;
            if (event === "sources") {
              setSources((payload.sources as Source[]) ?? []);
            } else if (event === "token") {
              setAnswer((previous) => previous + String(payload.text ?? ""));
            } else if (event === "done") {
              setModel(String(payload.model ?? ""));
            } else if (event === "error") {
              throw new Error(String(payload.message ?? "generation stream failed"));
            }
          }
          boundary = buffer.indexOf("\n\n");
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <header>
        <h1>FastAPI Docs RAG</h1>
        <p>中文提问，检索英文官方文档，回答带可点击出处。</p>
      </header>

      <section className="examples">
        {EXAMPLES.map((example) => (
          <button key={example} onClick={() => ask(example)} disabled={loading}>
            {example}
          </button>
        ))}
      </section>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          void ask();
        }}
      >
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="问一个 FastAPI 文档问题，例如：on_event 怎么迁移到 lifespan？"
          rows={3}
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? "检索并生成中…" : "提问"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}
      {loading && <div className="loading">正在检索文档并生成回答…</div>}

      {answer && (
        <section className="answer" aria-live="polite">
          <div className="answer-head">
            <h2>回答</h2>
            {model && <span className="model">{model}</span>}
          </div>
          <div className="markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
          </div>
        </section>
      )}

      {sources.length > 0 && (
        <section className="sources">
          <h2>来源</h2>
          <ol>
            {sources.map((source) => (
              <li key={`${source.index}-${source.url}`}>
                <a href={source.url} target="_blank" rel="noreferrer">
                  [{source.index}] {source.title}
                </a>
                <div className="source-meta">
                  {source.section_path.join(" > ")} · v{source.docs_version}
                  {source.is_deprecated && <span className="badge">deprecated</span>}
                  {!source.is_deprecated && source.deprecation_hint && (
                    <span className="badge hint">含废弃提示</span>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}
