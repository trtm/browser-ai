"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type Availability = "available" | "downloadable" | "downloading" | "unavailable" | "missing" | "error" | "checking" | string;
type ApiId = "prompt" | "summarizer" | "writer" | "rewriter" | "proofreader";
type Role = "user" | "assistant" | "system";
type SpeechMode = "dictation" | "voice" | "";

type PromptPart =
  | { type: "text"; value: string }
  | { type: "image"; value: Blob | File | HTMLCanvasElement | HTMLImageElement }
  | { type: "audio"; value: Blob | ArrayBuffer };

type PromptMessage = {
  role: "user" | "assistant" | "system";
  content: string | PromptPart[];
  prefix?: boolean;
};

type CreateMonitor = EventTarget;
type CreateOptions = Record<string, unknown> & {
  monitor?: (target: CreateMonitor) => void;
  signal?: AbortSignal;
};
type CallOptions = Record<string, unknown> & { signal?: AbortSignal };

type LanguageModelParams = Record<string, unknown> & {
  defaultTopK?: number;
  maxTopK?: number;
  defaultTemperature?: number;
  maxTemperature?: number;
};

interface LanguageModelSession extends EventTarget {
  readonly contextUsage?: number;
  readonly contextWindow?: number;
  readonly inputUsage?: number;
  readonly inputQuota?: number;
  prompt(input: string | PromptMessage[], options?: CallOptions): Promise<unknown>;
  promptStreaming?(input: string | PromptMessage[], options?: CallOptions): AsyncIterable<unknown> | ReadableStream<unknown>;
  append?(input: string | PromptMessage[], options?: CallOptions): Promise<void>;
  measureContextUsage?(input: string | PromptMessage[], options?: CallOptions): Promise<number>;
  measureInputUsage?(input: string | PromptMessage[], options?: CallOptions): Promise<number>;
  clone?(options?: CreateOptions): Promise<LanguageModelSession>;
  destroy?(): void;
}

interface LanguageModelConstructor {
  availability(options?: CreateOptions): Promise<Availability>;
  create(options?: CreateOptions): Promise<LanguageModelSession>;
  params?(): Promise<LanguageModelParams>;
}

interface UtilitySession {
  readonly inputUsage?: number;
  readonly inputQuota?: number;
  readonly contextUsage?: number;
  readonly contextWindow?: number;
  destroy?(): void;
  summarize?(input: string, options?: CallOptions): Promise<unknown>;
  summarizeStreaming?(input: string, options?: CallOptions): AsyncIterable<unknown> | ReadableStream<unknown>;
  write?(input: string, options?: CallOptions): Promise<unknown>;
  writeStreaming?(input: string, options?: CallOptions): AsyncIterable<unknown> | ReadableStream<unknown>;
  rewrite?(input: string, options?: CallOptions): Promise<unknown>;
  rewriteStreaming?(input: string, options?: CallOptions): AsyncIterable<unknown> | ReadableStream<unknown>;
  proofread?(input: string, options?: CallOptions): Promise<unknown>;
}

interface UtilityConstructor {
  availability(options?: CreateOptions): Promise<Availability>;
  create(options?: CreateOptions): Promise<UtilitySession>;
}

declare global {
  interface Window {
    LanguageModel?: LanguageModelConstructor;
    Summarizer?: UtilityConstructor;
    Writer?: UtilityConstructor;
    Rewriter?: UtilityConstructor;
    Proofreader?: UtilityConstructor;
  }
}

type ApiConfig = {
  id: ApiId;
  label: string;
  globalName: keyof Pick<Window, "LanguageModel" | "Summarizer" | "Writer" | "Rewriter" | "Proofreader">;
  model: string;
  description: string;
  sample: string;
};

type ApiStatus = {
  exists: boolean;
  availability: Availability;
  error: string;
};

type ProgressState = {
  kind: "download" | "session";
  text: string;
  ratio?: number;
  showBanner?: boolean;
  startedAt?: number;
};

type BrowserAiOptions = {
  stream: boolean;
  keepPromptSession: boolean;
  inputLanguage: string;
  outputLanguage: string;
  contextLanguage: string;
  systemPrompt: string;
  requestContext: string;
  sharedContext: string;
  promptJsonSchema: string;
  promptOmitResponseConstraintInput: boolean;
  promptTemperature: string;
  promptTopK: string;
  includeImage: boolean;
  includeAudio: boolean;
  voiceAutoSpeak: boolean;
  voiceMaxSeconds: string;
  outputVoiceId: string;
  outputVoiceRate: string;
  outputVoicePitch: string;
  outputVoiceVolume: string;
  summarizerType: string;
  summarizerFormat: string;
  summarizerLength: string;
  summarizerPreference: string;
  writerTone: string;
  writerFormat: string;
  writerLength: string;
  rewriterTone: string;
  rewriterFormat: string;
  rewriterLength: string;
  proofreaderInputLanguage: string;
};

type StoredMedia =
  | {
      type: "voice";
      audioId: string;
      duration: number;
      mimeType: string;
      size: number;
    }
  | {
      type: "image";
      count: number;
    };

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  apiLabel: string;
  timestamp: number;
  media?: StoredMedia;
  error?: boolean;
};

type Chat = {
  id: string;
  title: string;
  apiId: ApiId;
  timestamp: number;
  messages: ChatMessage[];
};

type VoiceRecord = {
  id: string;
  blob: Blob;
  duration: number;
  mimeType: string;
  size: number;
  createdAt: number;
};

type RunMedia = {
  audioParts?: PromptPart[];
  userDisplayText?: string;
  userAudioId?: string;
  userAudioDuration?: number;
  userAudioMimeType?: string;
  userAudioSize?: number;
  isVoiceMessage?: boolean;
};

const API_CONFIGS: ApiConfig[] = [
  {
    id: "prompt",
    label: "Prompt API",
    globalName: "LanguageModel",
    model: "Gemini Nano",
    description: "Freeform local chat with text, image input, audio input, streaming, JSON constraints, cloning, and context usage.",
    sample: "Explain the Chrome built-in Prompt API in three practical bullets.",
  },
  {
    id: "summarizer",
    label: "Summarizer API",
    globalName: "Summarizer",
    model: "Gemini Nano summarization model",
    description: "Summaries as key points, TLDR, teaser, or headline with length and format controls.",
    sample: "Chrome can run Gemini Nano locally for built-in AI APIs. Summarize why local inference changes privacy, latency, and offline behavior.",
  },
  {
    id: "writer",
    label: "Writer API",
    globalName: "Writer",
    model: "Gemini Nano writing model",
    description: "Create new text from a writing task with tone, format, length, and context.",
    sample: "Write a concise product update announcing a local AI playground for Chrome.",
  },
  {
    id: "rewriter",
    label: "Rewriter API",
    globalName: "Rewriter",
    model: "Gemini Nano rewriting model",
    description: "Revise input text with formality, length, format, shared context, and per-request context.",
    sample: "hey team, i built a chrome ai test page. it can check models and stream outputs. thoughts?",
  },
  {
    id: "proofreader",
    label: "Proofreader API",
    globalName: "Proofreader",
    model: "Gemini Nano proofreading model",
    description: "Correct grammar, spelling, and punctuation, then show corrected text and correction details.",
    sample: "I seen him yesterday at the store, and he bought two loafs of bread.",
  },
];

const DEFAULT_OPTIONS: BrowserAiOptions = {
  stream: true,
  keepPromptSession: true,
  inputLanguage: "en",
  outputLanguage: "en",
  contextLanguage: "en",
  systemPrompt: [
    "You are Gemini Nano in Chrome, a concise and accurate on-device assistant available through Chrome's built-in AI Prompt API.",
    "You run locally in Google Chrome after the model is installed. You do not have live internet access unless the surrounding page provides information.",
    "Current local context for this request:",
    "- Date: {date}",
    "- Time: {time}",
    "- Day of week: {weekday}",
    "- ISO calendar week: {calendar_week}",
    "- Time zone: {timezone}",
    "Use this context as authoritative when answering the user's message.",
  ].join("\n\n"),
  requestContext: "",
  sharedContext: "",
  promptJsonSchema: "",
  promptOmitResponseConstraintInput: false,
  promptTemperature: "",
  promptTopK: "",
  includeImage: true,
  includeAudio: true,
  voiceAutoSpeak: false,
  voiceMaxSeconds: "15",
  outputVoiceId: "auto",
  outputVoiceRate: "1",
  outputVoicePitch: "1",
  outputVoiceVolume: "1",
  summarizerType: "key-points",
  summarizerFormat: "markdown",
  summarizerLength: "short",
  summarizerPreference: "auto",
  writerTone: "neutral",
  writerFormat: "markdown",
  writerLength: "short",
  rewriterTone: "as-is",
  rewriterFormat: "as-is",
  rewriterLength: "as-is",
  proofreaderInputLanguage: "en",
};

const CHAT_STORAGE_KEY = "browserAiChats";
const OPTIONS_STORAGE_KEY = "browserAiOptions";
const LEFT_STORAGE_KEY = "browserAiLeftClosed";
const RIGHT_STORAGE_KEY = "browserAiRightClosed";
const WELCOME_STORAGE_KEY = "chromeAiWelcomed";
const VOICE_DB_NAME = "browserAiVoice";
const VOICE_DB_VERSION = 1;
const VOICE_STORE = "voiceMessages";
const MOBILE_LAYOUT_BREAKPOINT = 900;

let voiceDbPromise: Promise<IDBDatabase> | null = null;

function openVoiceDb(): Promise<IDBDatabase> {
  if (voiceDbPromise) return voiceDbPromise;
  voiceDbPromise = new Promise((resolve, reject) => {
    if (!("indexedDB" in window)) {
      reject(new Error("IndexedDB is not available."));
      return;
    }
    const request = indexedDB.open(VOICE_DB_NAME, VOICE_DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(VOICE_STORE)) {
        db.createObjectStore(VOICE_STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => {
      voiceDbPromise = null;
      reject(request.error || new Error("Failed to open voice database."));
    };
    request.onblocked = () => {
      voiceDbPromise = null;
      reject(new Error("Voice database is blocked by another tab."));
    };
  });
  return voiceDbPromise;
}

async function voiceStoreRequest<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openVoiceDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(VOICE_STORE, mode);
    const store = tx.objectStore(VOICE_STORE);
    let request: IDBRequest<T>;
    tx.oncomplete = () => resolve(request.result);
    tx.onerror = () => reject(tx.error || new Error("Voice storage transaction failed."));
    tx.onabort = () => reject(tx.error || new Error("Voice storage transaction aborted."));
    try {
      request = run(store);
    } catch (error) {
      tx.abort();
      reject(error);
    }
  });
}

function saveVoiceAudio(record: VoiceRecord): Promise<IDBValidKey> {
  return voiceStoreRequest("readwrite", (store) => store.put(record));
}

function loadVoiceAudio(id: string): Promise<VoiceRecord | undefined> {
  return voiceStoreRequest("readonly", (store) => store.get(id) as IDBRequest<VoiceRecord | undefined>);
}

function deleteVoiceAudio(id: string): Promise<undefined> {
  return voiceStoreRequest("readwrite", (store) => store.delete(id) as IDBRequest<undefined>);
}

function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function parseMarkdown(text: string): string {
  if (!text) return "";
  let html = escapeHtml(text);
  if ((html.match(/```/g) || []).length % 2 !== 0) html += "\n```";
  html = html.replace(/```(?:[a-z0-9_-]+)?\n([\s\S]*?)```/gi, "<pre><code>$1</code></pre>");
  const parts = html.split(/(<pre><code>[\s\S]*?<\/code><\/pre>)/);
  return parts
    .map((part) => {
      if (part.startsWith("<pre>")) return part;
      let next = part;
      next = next.replace(/^### (.*$)/gim, "<h3>$1</h3>");
      next = next.replace(/^## (.*$)/gim, "<h2>$1</h2>");
      next = next.replace(/^# (.*$)/gim, "<h1>$1</h1>");
      next = next.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      next = next.replace(/\*([^*]+)\*/g, "<em>$1</em>");
      next = next.replace(/`([^`]+)`/g, "<code>$1</code>");
      next = next.replace(/^\s*[-*] (.*$)/gim, "<li>$1</li>");
      next = next.replace(/(<li>[\s\S]*?<\/li>)/g, (match) => "<ul>" + match + "</ul>");
      next = next.replace(/<\/ul>\s*<ul>/g, "");
      return next
        .split("\n\n")
        .map((paragraph) => {
          const trimmed = paragraph.trim();
          if (!trimmed || trimmed.startsWith("<h") || trimmed.startsWith("<ul") || trimmed.startsWith("<pre")) return trimmed;
          return "<p>" + trimmed.replace(/\n/g, "<br>") + "</p>";
        })
        .join("\n");
    })
    .join("");
}

function stringify(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (value instanceof Error) return value.message || String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function normalizeStoredOptions(raw: unknown): BrowserAiOptions {
  const options: BrowserAiOptions = { ...DEFAULT_OPTIONS };
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return options;
  const object = raw as Record<string, unknown>;
  for (const [key, defaultValue] of Object.entries(DEFAULT_OPTIONS)) {
    if (!Object.prototype.hasOwnProperty.call(object, key)) continue;
    const value = object[key];
    const optionKey = key as keyof BrowserAiOptions;
    if (typeof defaultValue === "boolean") {
      (options[optionKey] as boolean) = value === true || value === "true";
    } else {
      (options[optionKey] as string) = value == null ? defaultValue : String(value);
    }
  }
  return options;
}

function normalizeChats(value: unknown): Chat[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((chat): chat is Chat => !!chat && typeof chat === "object" && "id" in chat && "messages" in chat)
    .map((chat) => ({
      id: String(chat.id),
      title: String(chat.title || "New Chat"),
      apiId: isApiId(chat.apiId) ? chat.apiId : "prompt",
      timestamp: Number(chat.timestamp) || Date.now(),
      messages: Array.isArray(chat.messages)
        ? chat.messages
            .filter((message): message is ChatMessage => !!message && typeof message === "object" && "role" in message)
            .map((message) => ({
              id: String(message.id || generateId()),
              role: isRole(message.role) ? message.role : "user",
              content: String(message.content || ""),
              apiLabel: String(message.apiLabel || "Prompt API"),
              timestamp: Number(message.timestamp) || Date.now(),
              media: normalizeMedia(message.media),
              error: Boolean(message.error),
            }))
        : [],
    }));
}

function normalizeMedia(value: unknown): StoredMedia | undefined {
  if (!value || typeof value !== "object") return undefined;
  const media = value as Record<string, unknown>;
  if (media.type === "voice") {
    return {
      type: "voice",
      audioId: String(media.audioId || ""),
      duration: Number(media.duration) || 0,
      mimeType: String(media.mimeType || ""),
      size: Number(media.size) || 0,
    };
  }
  if (media.type === "image") {
    return { type: "image", count: Math.max(1, Number(media.count) || 1) };
  }
  return undefined;
}

function isApiId(value: unknown): value is ApiId {
  return value === "prompt" || value === "summarizer" || value === "writer" || value === "rewriter" || value === "proofreader";
}

function isRole(value: unknown): value is Role {
  return value === "user" || value === "assistant" || value === "system";
}

function apiById(id: ApiId): ApiConfig {
  return API_CONFIGS.find((api) => api.id === id) || API_CONFIGS[0];
}

function chromeVersion(): string {
  if (typeof navigator === "undefined") return "unknown";
  const match = navigator.userAgent.match(/Chrome\/([0-9.]+)/);
  return match ? match[1] : "not Chrome";
}

function initialWelcomeMessage(): string {
  const isChrome = chromeVersion() !== "not Chrome";
  if (!isChrome) {
    return "⚠️ **This browser is not supported.**\n\nChrome Built-in AI requires **Google Chrome** (version 138+, Dev or Canary channel recommended).\n\nPlease open this page in Chrome to use local Gemini Nano models.";
  }
  if (typeof window.LanguageModel !== "function") {
    return "👋 Welcome! You're using Chrome, but the built-in AI APIs aren't enabled yet.\n\nTo get started:\n1. Go to `chrome://flags` in your address bar\n2. Search for **Prompt API for Gemini Nano** and set it to *Enabled*\n3. Also enable **Optimization Guide On Device Model**\n4. Restart Chrome\n\nAfter restarting, reload this page and you'll be ready to go!";
  }
  return "👋 Hi! I'm **Gemini Nano**, running locally right here in your browser.\n\nBecause I run on-device, your data never leaves your computer and I can respond instantly — even offline!\n\nSelect an API from the toolbar below and send a prompt to get started.";
}

function languageList(value: string): string[] {
  return value && value !== "auto" ? [value] : [];
}

function assignIfValue(target: Record<string, unknown>, key: string, value: unknown): void {
  if (value == null) return;
  if (Array.isArray(value) && value.length === 0) return;
  if (typeof value === "string" && value.trim() === "") return;
  target[key] = value;
}

function numberParamValue(params: LanguageModelParams | null, key: keyof LanguageModelParams): number | undefined {
  const value = params?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function optionalCreateNumber(value: string, label: string, options: { integer?: boolean; min?: number } = {}): number | undefined {
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) throw new Error(`${label} must be a number.`);
  if (options.integer && !Number.isInteger(parsed)) throw new Error(`${label} must be a whole number.`);
  if (options.min != null && parsed < options.min) throw new Error(`${label} must be ${options.min} or greater.`);
  return parsed;
}

function isoCalendarWeek(date: Date): { week: number; weekYear: number } {
  const localDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const day = localDate.getDay() || 7;
  localDate.setDate(localDate.getDate() + 4 - day);
  const weekYear = localDate.getFullYear();
  const yearStart = new Date(weekYear, 0, 1);
  const week = Math.ceil(((localDate.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  return { week, weekYear };
}

function renderSystemPromptTemplate(template: string, date = new Date()): string {
  const locale = navigator.language || undefined;
  const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone || "local time";
  const { week, weekYear } = isoCalendarWeek(date);
  const values: Record<string, string> = {
    date: new Intl.DateTimeFormat(locale, { year: "numeric", month: "long", day: "numeric" }).format(date),
    time: new Intl.DateTimeFormat(locale, { hour: "numeric", minute: "2-digit", second: "2-digit", timeZoneName: "short" }).format(date),
    weekday: new Intl.DateTimeFormat(locale, { weekday: "long" }).format(date),
    calendar_week: `${weekYear}-W${String(week).padStart(2, "0")}`,
    timezone: timeZone,
  };
  return String(template || "").replace(/\{(date|time|weekday|calendar_week|timezone)\}/g, (_match, key: string) => values[key] || "");
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function maxVoiceSeconds(options: BrowserAiOptions): number {
  const value = Number(options.voiceMaxSeconds);
  return Number.isFinite(value) ? Math.max(3, Math.min(60, Math.round(value))) : 15;
}

async function* iterateStream(stream: AsyncIterable<unknown> | ReadableStream<unknown>): AsyncIterable<unknown> {
  if (Symbol.asyncIterator in stream) {
    yield* stream as AsyncIterable<unknown>;
    return;
  }
  const reader = (stream as ReadableStream<unknown>).getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      yield value;
    }
  } finally {
    reader.releaseLock();
  }
}

function mergeStreamText(previous: string, chunk: unknown): string {
  const text = stringify(chunk);
  if (!text) return previous;
  if (!previous) return text;
  if (text.startsWith(previous)) return text;
  return previous + text;
}

function selectedApiMethodLabel(apiId: ApiId): string {
  if (apiId === "summarizer") return "summarize";
  if (apiId === "writer") return "write";
  if (apiId === "rewriter") return "rewrite";
  if (apiId === "proofreader") return "proofread";
  return "prompt";
}

function proofreadResultText(result: unknown, originalText: string): string {
  const object = result && typeof result === "object" ? (result as Record<string, unknown>) : {};
  const corrected = typeof object.correctedInput === "string" ? object.correctedInput : stringify(result);
  const corrections = Array.isArray(object.corrections) ? object.corrections : [];
  const lines = ["Corrected input", "", corrected, "", "Corrections"];
  if (!corrections.length) {
    lines.push("- No corrections returned.");
    return lines.join("\n");
  }
  for (const entry of corrections) {
    const correction = entry && typeof entry === "object" ? (entry as Record<string, unknown>) : {};
    const start = Number(correction.startIndex);
    const end = Number(correction.endIndex);
    const original = Number.isFinite(start) && Number.isFinite(end) ? originalText.slice(start, end) : "Correction";
    const replacement = String(correction.correction || correction.replacement || correction.corrected || "-");
    const type = Array.isArray(correction.types) ? correction.types.join(", ") : String(correction.type || correction.correctionType || "");
    const explanation = String(correction.explanation || correction.correctionExplanation || "");
    lines.push(`- ${original || "Correction"} -> ${replacement}${type ? ` (${type})` : ""}${explanation ? `: ${explanation}` : ""}`);
  }
  return lines.join("\n");
}

function renderProofreadHtml(result: unknown, originalText: string): string {
  const object = result && typeof result === "object" ? (result as Record<string, unknown>) : {};
  const corrected = typeof object.correctedInput === "string" ? object.correctedInput : stringify(result);
  const corrections = Array.isArray(object.corrections) ? object.corrections : [];
  const correctionHtml = corrections.length
    ? corrections
        .map((entry, index) => {
          const correction = entry && typeof entry === "object" ? (entry as Record<string, unknown>) : {};
          const start = Number(correction.startIndex);
          const end = Number(correction.endIndex);
          const original = Number.isFinite(start) && Number.isFinite(end) ? originalText.slice(start, end) : "";
          const replacement = String(correction.correction || correction.replacement || correction.corrected || "");
          const type = Array.isArray(correction.types) ? correction.types.join(", ") : String(correction.type || correction.correctionType || "");
          const explanation = String(correction.explanation || correction.correctionExplanation || "");
          return `<div class="correction"><div><strong>${escapeHtml(original || `Correction ${index + 1}`)}</strong> -> ${escapeHtml(replacement || "-")}</div><div>${escapeHtml(type || "type unavailable")}</div>${explanation ? `<div>${escapeHtml(explanation)}</div>` : ""}</div>`;
        })
        .join("")
    : '<div class="small">No corrections returned.</div>';
  return `<div class="proof-grid"><div><strong>Corrected input</strong></div><div>${escapeHtml(corrected)}</div><div><strong>Corrections</strong></div>${correctionHtml}<details><summary>Raw result</summary><pre>${escapeHtml(stringify(result))}</pre></details></div>`;
}

function VoiceMessage({ media }: { media: Extract<StoredMedia, { type: "voice" }> }) {
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let objectUrl = "";
    let cancelled = false;
    async function load() {
      if (!media.audioId) {
        setError("Saved audio unavailable.");
        return;
      }
      try {
        const record = await loadVoiceAudio(media.audioId);
        if (!record?.blob || cancelled) return;
        objectUrl = URL.createObjectURL(record.blob);
        setUrl(objectUrl);
      } catch {
        if (!cancelled) setError("Saved audio unavailable.");
      }
    }
    load();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [media.audioId]);

  return (
    <div className="voice-message">
      <div>
        <strong>Voice message</strong>
        <span>{formatDuration(media.duration)}</span>
      </div>
      {url ? <audio controls src={url} preload="metadata" /> : <span>{error || "Loading saved audio..."}</span>}
    </div>
  );
}

export default function BrowserAiApp() {
  const [hydrated, setHydrated] = useState(false);
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState("");
  const [selectedApiId, setSelectedApiId] = useState<ApiId>("prompt");
  const [options, setOptions] = useState<BrowserAiOptions>(DEFAULT_OPTIONS);
  const [statuses, setStatuses] = useState<Record<ApiId, ApiStatus>>(() => Object.fromEntries(API_CONFIGS.map((api) => [api.id, { exists: false, availability: "checking", error: "" }])) as Record<ApiId, ApiStatus>);
  const [progress, setProgress] = useState<Partial<Record<ApiId, ProgressState>>>({});
  const [running, setRunning] = useState(false);
  const [promptInput, setPromptInput] = useState("");
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [imagePreviews, setImagePreviews] = useState<string[]>([]);
  const [toast, setToast] = useState<{ id: string; message: string; tone: "default" | "error" | "success" }[]>([]);
  const [leftClosed, setLeftClosed] = useState(false);
  const [rightClosed, setRightClosed] = useState(false);
  const [mobilePane, setMobilePane] = useState<"left" | "right" | "">("");
  const [schemaStatus, setSchemaStatus] = useState("");
  const [promptUsage, setPromptUsage] = useState("-");

  const [languageModelParams, setLanguageModelParams] = useState<LanguageModelParams | null>(null);
  const [speechAvailability, setSpeechAvailability] = useState<Availability>("checking");
  const [speechError, setSpeechError] = useState("");
  const [speechMode, setSpeechMode] = useState<SpeechMode>("");
  const [speechProcessing, setSpeechProcessing] = useState(false);
  const [recordingStartedAt, setRecordingStartedAt] = useState(0);
  const [speechStatus, setSpeechStatus] = useState("");
  const [speakingMessageId, setSpeakingMessageId] = useState("");
  const [outputVoices, setOutputVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [outputVoicesLoaded, setOutputVoicesLoaded] = useState(false);
  const [welcome, setWelcome] = useState<{ chatId: string; content: string } | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState("");

  const instancesRef = useRef(new Map<string, LanguageModelSession | UtilitySession>());
  const signaturesRef = useRef(new Map<string, string>());
  const activeControllerRef = useRef<AbortController | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const speechModeRef = useRef<SpeechMode>("");
  const maxTimerRef = useRef<number | null>(null);
  const tickTimerRef = useRef<number | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);

  const activeChat = useMemo(() => chats.find((chat) => chat.id === activeChatId) || null, [activeChatId, chats]);
  const selectedApi = apiById(selectedApiId);
  const selectedStatus = statuses[selectedApiId] || { exists: false, availability: "checking", error: "" };
  const selectedProgress = progress[selectedApiId];
  const selectedInstance = instancesRef.current.get(instanceKey(selectedApiId, activeChatId));

  const showToast = useCallback((message: string, tone: "default" | "error" | "success" = "default") => {
    const id = generateId();
    setToast((items) => [...items, { id, message, tone }]);
    window.setTimeout(() => setToast((items) => items.filter((item) => item.id !== id)), 5200);
  }, []);

  useEffect(() => {
    function activateInitialChat(chat: Chat) {
      setChats([chat]);
      setActiveChatId(chat.id);
      setSelectedApiId(chat.apiId);
      if (!chat.messages.length && !localStorage.getItem(WELCOME_STORAGE_KEY)) {
        localStorage.setItem(WELCOME_STORAGE_KEY, "true");
        setWelcome({ chatId: chat.id, content: initialWelcomeMessage() });
      }
    }

    try {
      const storedOptions = localStorage.getItem(OPTIONS_STORAGE_KEY);
      setOptions(normalizeStoredOptions(storedOptions ? JSON.parse(storedOptions) : null));
    } catch {
      setOptions(DEFAULT_OPTIONS);
    }
    try {
      const storedChats = localStorage.getItem(CHAT_STORAGE_KEY);
      const storedChatList = normalizeChats(storedChats ? JSON.parse(storedChats) : null);
      const parsedChats = storedChatList.slice(0, 50);
      deleteOrphanedVoiceAudioIds(storedChatList.slice(50).flatMap(voiceAudioIdsForChat), parsedChats);
      if (parsedChats.length) {
        setChats(parsedChats);
        setActiveChatId(parsedChats[0].id);
        setSelectedApiId(parsedChats[0].apiId);
        if (!parsedChats[0].messages.length && !localStorage.getItem(WELCOME_STORAGE_KEY)) {
          localStorage.setItem(WELCOME_STORAGE_KEY, "true");
          setWelcome({ chatId: parsedChats[0].id, content: initialWelcomeMessage() });
        }
      } else {
        activateInitialChat(createBlankChat("prompt"));
      }
    } catch {
      activateInitialChat(createBlankChat("prompt"));
    }
    setLeftClosed(localStorage.getItem(LEFT_STORAGE_KEY) === "true");
    setRightClosed(localStorage.getItem(RIGHT_STORAGE_KEY) === "true");
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chats));
  }, [chats, hydrated]);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(OPTIONS_STORAGE_KEY, JSON.stringify(options));
  }, [hydrated, options]);

  useEffect(() => {
    if (!hydrated) return;
    if (!supportsSpeechSynthesisOutput()) {
      setOutputVoices([]);
      setOutputVoicesLoaded(true);
      return;
    }
    const refresh = () => {
      const voices = window.speechSynthesis.getVoices();
      setOutputVoices(voices);
      setOutputVoicesLoaded(voices.length > 0);
    };
    refresh();
    window.speechSynthesis.addEventListener?.("voiceschanged", refresh);
    const timer = window.setTimeout(refresh, 0);
    return () => {
      window.clearTimeout(timer);
      window.speechSynthesis.removeEventListener?.("voiceschanged", refresh);
    };
  }, [hydrated]);

  useEffect(() => {
    const urls = imageFiles.map((file) => URL.createObjectURL(file));
    setImagePreviews(urls);
    return () => urls.forEach((url) => URL.revokeObjectURL(url));
  }, [imageFiles]);

  useEffect(() => {
    if (!hydrated) return;
    refreshStatuses();
    refreshSpeechAvailability();
    refreshPromptParams();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, options, selectedApiId]);

  useEffect(() => {
    if (!hydrated) return;
    const timer = window.setTimeout(() => {
      measurePromptUsage(promptInput).catch(() => undefined);
    }, 300);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, promptInput, selectedApiId, activeChatId]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight });
  }, [activeChat?.messages, running]);

  useEffect(() => {
    const valid = validateJsonSchema(options.promptJsonSchema);
    setSchemaStatus(valid);
  }, [options.promptJsonSchema]);

  function instanceKey(apiId: ApiId, chatId = activeChatId): string {
    return apiId === "prompt" ? `prompt:${chatId || "none"}` : apiId;
  }

  function selectedConstructor(api: ApiConfig): LanguageModelConstructor | UtilityConstructor | undefined {
    return window[api.globalName] as LanguageModelConstructor | UtilityConstructor | undefined;
  }

  function monitorFor(apiId: ApiId) {
    return (target: CreateMonitor) => {
      target.addEventListener("downloadprogress", (event) => {
        const progressEvent = event as ProgressEvent;
        const loaded = Number(progressEvent.loaded);
        const total = Number(progressEvent.total);
        const ratio = Number.isFinite(total) && total > 0 ? loaded / total : loaded;
        setProgress((current) => {
          const prev = current[apiId];
          const startedAt = prev?.kind === "download" && prev.startedAt ? prev.startedAt : Date.now();
          const elapsed = Date.now() - startedAt;
          return {
            ...current,
            [apiId]: {
              kind: "download",
              text: "Chrome is installing the model for local inference.",
              ratio,
              showBanner: Number.isFinite(ratio) && ratio < 1 && elapsed > 500,
              startedAt,
            },
          };
        });
        if (ratio >= 1) {
          setStatuses((current) => ({ ...current, [apiId]: { ...(current[apiId] || {}), exists: true, availability: "available", error: "" } }));
        }
      });
    };
  }

  function buildCommonCreateOptions(apiId: ApiId): CreateOptions {
    return { monitor: monitorFor(apiId) };
  }

  function buildExpectedLanguageOptions() {
    const expectedInputLanguages = languageList(options.inputLanguage);
    const expectedContextLanguages = languageList(options.contextLanguage);
    const outputLanguage = options.outputLanguage === "auto" ? "" : options.outputLanguage;
    return { expectedInputLanguages, expectedContextLanguages, outputLanguage };
  }

  function buildCreateOptions(apiId: ApiId, config: { forceIncludeAudio?: boolean; renderPromptTemplate?: boolean } = {}): CreateOptions {
    const createOptions = buildCommonCreateOptions(apiId);
    const expected = buildExpectedLanguageOptions();
    if (apiId === "prompt") {
      const expectedInputs: Record<string, unknown>[] = [];
      const inputLanguages = languageList(options.inputLanguage);
      expectedInputs.push(inputLanguages.length ? { type: "text", languages: inputLanguages } : { type: "text" });
      if (options.includeImage) expectedInputs.push({ type: "image" });
      if (options.includeAudio || config.forceIncludeAudio) expectedInputs.push({ type: "audio" });
      createOptions.expectedInputs = expectedInputs;
      const outputLanguages = languageList(options.outputLanguage);
      createOptions.expectedOutputs = outputLanguages.length ? [{ type: "text", languages: outputLanguages }] : [{ type: "text" }];
      assignIfValue(createOptions, "temperature", optionalCreateNumber(options.promptTemperature, "Temperature", { min: 0 }));
      assignIfValue(createOptions, "topK", optionalCreateNumber(options.promptTopK, "Top K", { integer: true, min: 1 }));
      const systemPrompt = options.systemPrompt.trim();
      if (systemPrompt) {
        createOptions.initialPrompts = [{ role: "system", content: config.renderPromptTemplate ? renderSystemPromptTemplate(systemPrompt) : systemPrompt }];
      }
      return createOptions;
    }
    if (apiId === "summarizer") {
      createOptions.type = options.summarizerType;
      createOptions.format = options.summarizerFormat;
      createOptions.length = options.summarizerLength;
      createOptions.preference = options.summarizerPreference;
      assignIfValue(createOptions, "sharedContext", options.sharedContext.trim());
    }
    if (apiId === "writer") {
      createOptions.tone = options.writerTone;
      createOptions.format = options.writerFormat;
      createOptions.length = options.writerLength;
      assignIfValue(createOptions, "sharedContext", options.sharedContext.trim());
    }
    if (apiId === "rewriter") {
      createOptions.tone = options.rewriterTone;
      createOptions.format = options.rewriterFormat;
      createOptions.length = options.rewriterLength;
      assignIfValue(createOptions, "sharedContext", options.sharedContext.trim());
    }
    if (apiId === "proofreader") {
      assignIfValue(createOptions, "expectedInputLanguages", languageList(options.proofreaderInputLanguage));
      return createOptions;
    }
    assignIfValue(createOptions, "expectedInputLanguages", expected.expectedInputLanguages);
    assignIfValue(createOptions, "expectedContextLanguages", expected.expectedContextLanguages);
    assignIfValue(createOptions, "outputLanguage", expected.outputLanguage);
    return createOptions;
  }

  function buildAvailabilityOptions(apiId: ApiId, config: { forceIncludeAudio?: boolean } = {}): CreateOptions {
    const createOptions = buildCreateOptions(apiId, { renderPromptTemplate: true, ...config });
    delete createOptions.monitor;
    return createOptions;
  }

  function buildCallOptions(apiId: ApiId): CallOptions {
    const callOptions: CallOptions = {};
    if (activeControllerRef.current) callOptions.signal = activeControllerRef.current.signal;
    const context = options.requestContext.trim();
    if (apiId === "summarizer" || apiId === "writer" || apiId === "rewriter") {
      assignIfValue(callOptions, "context", context);
    }
    if (apiId === "prompt") {
      const schemaText = options.promptJsonSchema.trim();
      if (schemaText) {
        callOptions.responseConstraint = JSON.parse(schemaText);
        if (options.promptOmitResponseConstraintInput) callOptions.omitResponseConstraintInput = true;
      }
    }
    return callOptions;
  }

  function optionsSignature(apiId: ApiId, config: { forceIncludeAudio?: boolean } = {}): string {
    const createOptions = buildCreateOptions(apiId, config);
    delete createOptions.monitor;
    return JSON.stringify(createOptions);
  }

  async function availabilityFor(api: ApiConfig): Promise<ApiStatus> {
    const Ctor = selectedConstructor(api);
    if (!Ctor || typeof Ctor.availability !== "function") return { exists: false, availability: "missing", error: "" };
    try {
      const availability = await Ctor.availability(buildAvailabilityOptions(api.id));
      return { exists: true, availability: availability || "available", error: "" };
    } catch (error) {
      return { exists: true, availability: "error", error: stringify(error) };
    }
  }

  async function refreshStatuses() {
    const next: Record<ApiId, ApiStatus> = { ...statuses };
    const pairs = await Promise.all(API_CONFIGS.map(async (api) => [api.id, await availabilityFor(api)] as const));
    for (const [apiId, status] of pairs) next[apiId] = status;
    setStatuses(next);
  }

  async function refreshPromptParams() {
    if (typeof window === "undefined" || !window.LanguageModel?.params) {
      setLanguageModelParams(null);
      return;
    }
    try {
      const params = await window.LanguageModel.params();
      setLanguageModelParams(params || null);
    } catch (error) {
      setLanguageModelParams(null);
    }
  }

  async function safeDestroy(apiId: ApiId, chatId = activeChatId) {
    const key = instanceKey(apiId, chatId);
    const instance = instancesRef.current.get(key);
    if (instance?.destroy) {
      try {
        instance.destroy();
      } catch {
        // Ignore Chrome implementation cleanup errors.
      }
    }
    instancesRef.current.delete(key);
    signaturesRef.current.delete(key);
  }

  async function getOrCreateInstance(apiId: ApiId, forceNew = false, config: { forceIncludeAudio?: boolean } = {}): Promise<LanguageModelSession | UtilitySession> {
    const api = apiById(apiId);
    const Ctor = selectedConstructor(api);
    if (!Ctor || typeof Ctor.create !== "function") throw new Error(`${api.globalName} is not exposed in this browser context.`);
    const key = instanceKey(apiId);
    const signature = optionsSignature(apiId, config);
    const existing = instancesRef.current.get(key);
    const mayReuse = apiId === "prompt" ? options.keepPromptSession : true;
    if (!forceNew && mayReuse && existing && signaturesRef.current.get(key) === signature) return existing;
    if (existing) await safeDestroy(apiId);
    setProgress((current) => ({ ...current, [apiId]: { kind: "session", text: "creating session" } }));
    const instance = await Ctor.create(buildCreateOptions(apiId, { renderPromptTemplate: true, ...config }));
    instancesRef.current.set(key, instance);
    signaturesRef.current.set(key, signature);
    setStatuses((current) => ({ ...current, [apiId]: { exists: true, availability: "available", error: "" } }));
    setProgress((current) => ({ ...current, [apiId]: { kind: "session", text: "ready" } }));
    return instance;
  }

  function updateOptions<K extends keyof BrowserAiOptions>(key: K, value: BrowserAiOptions[K]) {
    setOptions((current) => ({ ...current, [key]: value }));
  }

  function validateJsonSchema(text: string): string {
    const trimmed = text.trim();
    if (!trimmed) return "";
    try {
      JSON.parse(trimmed);
      return "valid JSON";
    } catch {
      return "invalid JSON";
    }
  }

  function addMessage(chatId: string, message: ChatMessage) {
    setChats((current) =>
      current.map((chat) => {
        if (chat.id !== chatId) return chat;
        const messages = [...chat.messages, message];
        const firstUser = message.role === "user" && chat.messages.filter((item) => item.role === "user").length === 0;
        return {
          ...chat,
          messages,
          timestamp: Date.now(),
          title: firstUser ? message.content.slice(0, 45) + (message.content.length > 45 ? "..." : "") : chat.title,
        };
      }),
    );
  }

  function updateMessage(chatId: string, messageId: string, patch: Partial<ChatMessage>) {
    setChats((current) =>
      current.map((chat) =>
        chat.id === chatId
          ? {
              ...chat,
              messages: chat.messages.map((message) => (message.id === messageId ? { ...message, ...patch } : message)),
              timestamp: Date.now(),
            }
          : chat,
      ),
    );
  }

  function buildPromptInput(text: string, imageParts: PromptPart[], audioParts: PromptPart[]): PromptMessage[] {
    const mediaParts = [...imageParts, ...audioParts];
    if (!mediaParts.length) return [{ role: "user", content: text }];
    const content: PromptPart[] = [];
    if (text.trim()) content.push({ type: "text", value: text });
    content.push(...mediaParts);
    return [{ role: "user", content }];
  }

  function pendingImageParts(): PromptPart[] {
    if (!options.includeImage) return [];
    return imageFiles.map((file) => ({ type: "image", value: file }) as PromptPart);
  }

  async function writeStreamToMessage(stream: AsyncIterable<unknown> | ReadableStream<unknown>, chatId: string, messageId: string): Promise<string> {
    let output = "";
    for await (const chunk of iterateStream(stream)) {
      output = mergeStreamText(output, chunk);
      updateMessage(chatId, messageId, { content: output });
    }
    return output;
  }

  async function runSelected(text: string, media: RunMedia = {}) {
    const api = selectedApi;
    const apiId = api.id;
    const userText = String(text || "");
    const imageParts = pendingImageParts();
    const audioParts = media.audioParts || [];
    if (running || (!userText.trim() && !imageParts.length && !audioParts.length)) return;
    const chatId = activeChatId;
    const userMessage: ChatMessage = {
      id: generateId(),
      role: "user",
      content: media.userDisplayText || userText || (imageParts.length ? `Image message (${imageParts.length})` : ""),
      apiLabel: api.label,
      timestamp: Date.now(),
      media: media.isVoiceMessage && media.userAudioId
        ? { type: "voice", audioId: media.userAudioId, duration: Number(media.userAudioDuration) || 0, mimeType: media.userAudioMimeType || "", size: Number(media.userAudioSize) || 0 }
        : imageParts.length
          ? { type: "image", count: imageParts.length }
          : undefined,
    };
    const assistantMessage: ChatMessage = {
      id: generateId(),
      role: "assistant",
      content: media.isVoiceMessage ? "Processing voice message..." : "Starting...",
      apiLabel: api.label,
      timestamp: Date.now(),
    };
    addMessage(chatId, userMessage);
    addMessage(chatId, assistantMessage);
    setWelcome((current) => (current?.chatId === chatId ? null : current));
    setPromptInput("");
    setImageFiles([]);
    setRunning(true);
    activeControllerRef.current = new AbortController();

    try {
      const instance = await getOrCreateInstance(apiId, false, { forceIncludeAudio: audioParts.length > 0 });
      const callOptions = buildCallOptions(apiId);
      let finalContent = "";
      if (apiId === "prompt") {
        const promptInstance = instance as LanguageModelSession;
        const input = buildPromptInput(userText, imageParts, audioParts);
        if (options.stream && promptInstance.promptStreaming) {
          updateMessage(chatId, assistantMessage.id, { content: "" });
          finalContent = await writeStreamToMessage(promptInstance.promptStreaming(input, callOptions), chatId, assistantMessage.id);
        } else {
          finalContent = stringify(await promptInstance.prompt(input, callOptions));
          updateMessage(chatId, assistantMessage.id, { content: finalContent });
        }
      }
      if (apiId === "summarizer") {
        const summarizer = instance as UtilitySession;
        if (options.stream && summarizer.summarizeStreaming) {
          updateMessage(chatId, assistantMessage.id, { content: "" });
          finalContent = await writeStreamToMessage(summarizer.summarizeStreaming(userText, callOptions), chatId, assistantMessage.id);
        } else {
          finalContent = stringify(await summarizer.summarize?.(userText, callOptions));
          updateMessage(chatId, assistantMessage.id, { content: finalContent });
        }
      }
      if (apiId === "writer") {
        const writer = instance as UtilitySession;
        if (options.stream && writer.writeStreaming) {
          updateMessage(chatId, assistantMessage.id, { content: "" });
          finalContent = await writeStreamToMessage(writer.writeStreaming(userText, callOptions), chatId, assistantMessage.id);
        } else {
          finalContent = stringify(await writer.write?.(userText, callOptions));
          updateMessage(chatId, assistantMessage.id, { content: finalContent });
        }
      }
      if (apiId === "rewriter") {
        const rewriter = instance as UtilitySession;
        if (options.stream && rewriter.rewriteStreaming) {
          updateMessage(chatId, assistantMessage.id, { content: "" });
          finalContent = await writeStreamToMessage(rewriter.rewriteStreaming(userText, callOptions), chatId, assistantMessage.id);
        } else {
          finalContent = stringify(await rewriter.rewrite?.(userText, callOptions));
          updateMessage(chatId, assistantMessage.id, { content: finalContent });
        }
      }
      if (apiId === "proofreader") {
        const proofreader = instance as UtilitySession;
        const result = await proofreader.proofread?.(userText, callOptions);
        finalContent = proofreadResultText(result, userText);
        updateMessage(chatId, assistantMessage.id, { content: finalContent });
      }
      if (media.isVoiceMessage && finalContent && options.voiceAutoSpeak) {
        speakText(finalContent, assistantMessage.id).catch((error) => showToast(stringify(error), "error"));
      }
      await refreshStatuses();
    } catch (error) {
      const message = stringify(error);
      updateMessage(chatId, assistantMessage.id, { content: message, error: true });
      setStatuses((current) => ({ ...current, [apiId]: { exists: true, availability: "error", error: message } }));
      showToast(message, "error");
    } finally {
      setRunning(false);
      activeControllerRef.current = null;
      setSpeechStatus("");
      measurePromptUsage("").catch(() => undefined);
    }
  }

  async function createNewChat() {
    const chat = createBlankChat(selectedApiId);
    const next = [chat, ...chats].slice(0, 50);
    const keptIds = new Set(next.map((item) => item.id));
    const removed = chats.filter((item) => !keptIds.has(item.id));
    deleteOrphanedVoiceAudioIds(removed.flatMap(voiceAudioIdsForChat), next);
    setChats(next);
    setActiveChatId(chat.id);
    setSelectedApiId(chat.apiId);
    setWelcome(null);
    setMobilePane("");
  }

  function selectChat(chat: Chat) {
    setActiveChatId(chat.id);
    setSelectedApiId(chat.apiId);
    setWelcome(null);
    setMobilePane("");
  }

  async function deleteChat(chatId: string) {
    const chat = chats.find((item) => item.id === chatId);
    await safeDestroy("prompt", chatId);
    const remaining = chats.filter((item) => item.id !== chatId);
    const next = remaining.length ? remaining : [createBlankChat(selectedApiId)];
    deleteOrphanedVoiceAudioIds(voiceAudioIdsForChat(chat), next);
    if (activeChatId === chatId) {
      setActiveChatId(next[0].id);
      setSelectedApiId(next[0].apiId);
    }
    setChats(next);
  }

  function onApiChange(apiId: ApiId) {
    setSelectedApiId(apiId);
    setChats((current) => current.map((chat) => (chat.id === activeChatId ? { ...chat, apiId } : chat)));
    if (apiId !== "prompt") setImageFiles([]);
  }

  function stopRun() {
    activeControllerRef.current?.abort();
  }

  function supportsSpeechCapture(): boolean {
    return !!navigator.mediaDevices?.getUserMedia && "MediaRecorder" in window;
  }

  function supportsSpeechSynthesisOutput(): boolean {
    return "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
  }

  async function refreshSpeechAvailability() {
    if (!supportsSpeechCapture()) {
      setSpeechAvailability("unavailable");
      setSpeechError("MediaRecorder or microphone capture is not available.");
      return;
    }
    if (!window.LanguageModel) {
      setSpeechAvailability("missing");
      setSpeechError("LanguageModel is not exposed in this browser.");
      return;
    }
    setSpeechAvailability("checking");
    setSpeechError("");
    try {
      const availability = await window.LanguageModel.availability(buildAvailabilityOptions("prompt", { forceIncludeAudio: true }));
      setSpeechAvailability(availability || "available");
    } catch (error) {
      setSpeechAvailability("error");
      setSpeechError(stringify(error));
    }
  }

  function speechAvailabilityLabel(): string {
    if (!supportsSpeechCapture()) return "microphone capture unavailable";
    if (!window.LanguageModel) return "LanguageModel missing";
    if (speechError) return `${speechAvailability}: ${speechError}`;
    return speechAvailability || "checking";
  }

  function canStartSpeechInput(): boolean {
    return supportsSpeechCapture() && !!window.LanguageModel && ["available", "downloadable"].includes(speechAvailability);
  }

  function preferredAudioMimeType(): string {
    const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
    return types.find((type) => MediaRecorder.isTypeSupported(type)) || "";
  }

  async function toggleSpeechRecording(mode: Exclude<SpeechMode, "">) {
    if (selectedApiId !== "prompt") {
      showToast("Speech input is available for the Prompt API.", "error");
      return;
    }
    if (recorderRef.current) {
      if (speechModeRef.current === mode) await stopSpeechRecording();
      else showToast("Stop the current recording first.", "error");
      return;
    }
    await startSpeechRecording(mode);
  }

  async function startSpeechRecording(mode: Exclude<SpeechMode, "">) {
    if (!canStartSpeechInput()) {
      showToast("Speech input is not available: " + speechAvailabilityLabel(), "error");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = preferredAudioMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;
      streamRef.current = stream;
      chunksRef.current = [];
      speechModeRef.current = mode;
      setSpeechMode(mode);
      setSpeechProcessing(false);
      const started = Date.now();
      setRecordingStartedAt(started);
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data?.size) chunksRef.current.push(event.data);
      });
      recorder.start();
      setSpeechStatus(recordingStatus(mode, started));
      maxTimerRef.current = window.setTimeout(() => stopSpeechRecording(), maxVoiceSeconds(options) * 1000);
      tickTimerRef.current = window.setInterval(() => setSpeechStatus(recordingStatus(mode, started)), 500);
    } catch (error) {
      cleanupSpeechRecorder();
      showToast("Microphone access failed: " + stringify(error), "error");
    }
  }

  async function stopSpeechRecording() {
    const recorder = recorderRef.current;
    if (!recorder) return;
    const mode = speechModeRef.current;
    const startedAt = recordingStartedAt;
    const mimeType = recorder.mimeType || preferredAudioMimeType() || "audio/webm";
    await new Promise<void>((resolve) => {
      recorder.addEventListener("stop", () => resolve(), { once: true });
      if (recorder.state !== "inactive") recorder.stop();
      else resolve();
    });
    const chunks = chunksRef.current.slice();
    const duration = Math.max(0, Date.now() - startedAt);
    cleanupSpeechRecorder();
    if (!chunks.length) {
      setSpeechStatus("");
      showToast("No audio was captured.", "error");
      return;
    }
    const blob = new Blob(chunks, { type: mimeType });
    setSpeechProcessing(true);
    setSpeechMode(mode);
    try {
      if (mode === "dictation") await transcribeIntoComposer(blob);
      else await sendVoiceMessage(blob, duration);
    } finally {
      setSpeechMode("");
      speechModeRef.current = "";
      setSpeechProcessing(false);
    }
  }

  function cleanupSpeechRecorder() {
    if (maxTimerRef.current) window.clearTimeout(maxTimerRef.current);
    if (tickTimerRef.current) window.clearInterval(tickTimerRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    recorderRef.current = null;
    streamRef.current = null;
    chunksRef.current = [];
    maxTimerRef.current = null;
    tickTimerRef.current = null;
    setRecordingStartedAt(0);
  }

  function recordingStatus(mode: SpeechMode, startedAt: number): string {
    const label = mode === "dictation" ? "Dictation recording" : "Voice recording";
    return `${label}. ${formatDuration(Date.now() - startedAt)} elapsed. Max ${maxVoiceSeconds(options)}s.`;
  }

  async function createTemporaryAudioSession(systemPrompt: string): Promise<LanguageModelSession> {
    if (!window.LanguageModel) throw new Error("LanguageModel is not exposed in this browser context.");
    const inputLanguages = languageList(options.inputLanguage);
    const outputLanguages = languageList(options.outputLanguage);
    const createOptions = buildCommonCreateOptions("prompt");
    createOptions.expectedInputs = inputLanguages.length ? [{ type: "text", languages: inputLanguages }, { type: "audio" }] : [{ type: "text" }, { type: "audio" }];
    createOptions.expectedOutputs = outputLanguages.length ? [{ type: "text", languages: outputLanguages }] : [{ type: "text" }];
    createOptions.initialPrompts = [{ role: "system", content: systemPrompt }];
    return window.LanguageModel.create(createOptions);
  }

  async function transcribeAudio(blob: Blob): Promise<string> {
    let session: LanguageModelSession | null = null;
    try {
      setSpeechStatus("Transcribing. Converting speech into editable prompt text...");
      session = await createTemporaryAudioSession("You are a speech transcription helper. Return only the words spoken in the audio. Do not answer the speaker. If there is no intelligible speech, return exactly NO_SPEECH.");
      const response = await session.prompt([{ role: "user", content: [{ type: "text", value: "Transcribe this audio exactly." }, { type: "audio", value: blob }] }]);
      return stringify(response).trim().replace(/^["']|["']$/g, "").trim();
    } finally {
      session?.destroy?.();
    }
  }

  async function transcribeIntoComposer(blob: Blob) {
    try {
      const transcript = await transcribeAudio(blob);
      if (!transcript || transcript === "NO_SPEECH") {
        setSpeechStatus("");
        showToast("No intelligible speech was detected.", "error");
        return;
      }
      setPromptInput((current) => [current.trim(), transcript].filter(Boolean).join(" "));
      setSpeechStatus("Dictation ready. Review the inserted text, then send when ready.");
    } catch (error) {
      setSpeechStatus("");
      showToast("Dictation failed: " + stringify(error), "error");
    }
  }

  async function ensureAudioPromptSessionEnabled() {
    if (options.includeAudio) return;
    setOptions((current) => ({ ...current, includeAudio: true }));
    await safeDestroy("prompt");
  }

  async function sendVoiceMessage(blob: Blob, duration: number) {
    try {
      await ensureAudioPromptSessionEnabled();
      const audioId = generateId();
      const record: VoiceRecord = {
        id: audioId,
        blob,
        duration,
        mimeType: blob.type || "audio/webm",
        size: blob.size,
        createdAt: Date.now(),
      };
      await saveVoiceAudio(record);
      setSpeechStatus("");
      await runSelected("Respond to this spoken voice message. If the audio is silent or unintelligible, say so briefly.", {
        audioParts: [{ type: "audio", value: blob }],
        userDisplayText: `Voice message (${Math.max(0.1, duration / 1000).toFixed(1)}s)`,
        userAudioId: audioId,
        userAudioDuration: duration,
        userAudioMimeType: record.mimeType,
        userAudioSize: record.size,
        isVoiceMessage: true,
      });
    } catch (error) {
      setSpeechStatus("");
      showToast("Voice message failed: " + stringify(error), "error");
    }
  }

  function outputVoiceId(voice: SpeechSynthesisVoice): string {
    return [voice.voiceURI || "", voice.name || "", voice.lang || ""].join("||");
  }

  function outputVoiceLabel(voice: SpeechSynthesisVoice): string {
    const lang = voice.lang || "unknown";
    const suffix = voice.default ? " - default" : "";
    return `${voice.name || voice.voiceURI || "Unnamed voice"} (${lang})${suffix}`;
  }

  function selectedOutputLanguage(): string {
    return options.outputLanguage === "auto" ? navigator.language || "en" : options.outputLanguage;
  }

  function voiceMatchesSelectedLanguage(voice: SpeechSynthesisVoice): boolean {
    const base = String(selectedOutputLanguage() || "").toLowerCase().split("-")[0];
    return !!(base && voice.lang && voice.lang.toLowerCase().split("-")[0] === base);
  }

  function selectedOutputVoice(voices = outputVoices): SpeechSynthesisVoice | null {
    const id = options.outputVoiceId;
    if (!id || id === "auto") return null;
    return voices.find((voice) => outputVoiceId(voice) === id) || null;
  }

  function voiceNumberOption(name: "outputVoiceRate" | "outputVoicePitch" | "outputVoiceVolume", fallback: number, min: number, max: number): number {
    const value = Number(options[name]);
    if (!Number.isFinite(value)) return fallback;
    return Math.max(min, Math.min(max, value));
  }

  async function waitForOutputVoices(): Promise<SpeechSynthesisVoice[]> {
    if (!supportsSpeechSynthesisOutput()) return [];
    const current = window.speechSynthesis.getVoices();
    if (current.length) {
      setOutputVoices(current);
      setOutputVoicesLoaded(true);
      return current;
    }
    return new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        window.speechSynthesis.removeEventListener?.("voiceschanged", finish);
        const voices = window.speechSynthesis.getVoices();
        setOutputVoices(voices);
        setOutputVoicesLoaded(voices.length > 0);
        resolve(voices);
      };
      window.speechSynthesis.addEventListener?.("voiceschanged", finish, { once: true });
      const timer = window.setTimeout(finish, 1500);
    });
  }

  async function speakText(text: string, messageId: string) {
    if (!supportsSpeechSynthesisOutput()) {
      showToast("Speech synthesis is not available in this browser.", "error");
      return;
    }
    const speechText = String(text || "").replace(/[`*_#>-]/g, "").trim();
    if (!speechText) {
      showToast("There is no assistant text to speak.", "error");
      return;
    }
    const voices = await waitForOutputVoices();
    const voice = selectedOutputVoice(voices);
    window.speechSynthesis.cancel();
    setSpeakingMessageId(messageId);
    const utterance = new SpeechSynthesisUtterance(speechText);
    utterance.lang = voice?.lang || selectedOutputLanguage();
    if (voice) utterance.voice = voice;
    utterance.rate = voiceNumberOption("outputVoiceRate", 1, 0.5, 2);
    utterance.pitch = voiceNumberOption("outputVoicePitch", 1, 0, 2);
    utterance.volume = voiceNumberOption("outputVoiceVolume", 1, 0, 1);
    utterance.onend = () => setSpeakingMessageId("");
    utterance.onerror = (event) => {
      setSpeakingMessageId("");
      const detail = event.error ? ": " + event.error : "";
      if (!["canceled", "interrupted"].includes(event.error)) showToast("Speech synthesis failed" + detail, "error");
    };
    window.speechSynthesis.speak(utterance);
  }

  function toggleMessageSpeech(message: ChatMessage) {
    if (speakingMessageId === message.id) {
      window.speechSynthesis.cancel();
      setSpeakingMessageId("");
      return;
    }
    speakText(message.content, message.id).catch((error) => showToast(stringify(error), "error"));
  }

  function copyMessageText(message: ChatMessage) {
    navigator.clipboard.writeText(message.content).then(() => {
      setCopiedMessageId(message.id);
      window.setTimeout(() => {
        setCopiedMessageId((current) => (current === message.id ? "" : current));
      }, 2000);
    }).catch((error) => showToast("Copy failed: " + stringify(error), "error"));
  }

  function composerPlaceholder(apiId: ApiId): string {
    if (apiId === "summarizer") return "Summarize text...";
    if (apiId === "writer") return "Writing task...";
    if (apiId === "rewriter") return "Rewrite text...";
    if (apiId === "proofreader") return "Proofread text...";
    return "Ask AI...";
  }

  async function appendCurrentPrompt() {
    if (selectedApiId !== "prompt" || !promptInput.trim()) return;
    try {
      const session = (await getOrCreateInstance("prompt")) as LanguageModelSession;
      if (!session.append) throw new Error("append() is not available on this LanguageModel session.");
      activeControllerRef.current = new AbortController();
      await session.append(promptInput.trim(), buildCallOptions("prompt"));
      addMessage(activeChatId, { id: generateId(), role: "system", content: "Prompt appended to the active session context.", apiLabel: "Session", timestamp: Date.now() });
      setPromptInput("");
      await measurePromptUsage("");
    } catch (error) {
      showToast("Append failed: " + stringify(error), "error");
    } finally {
      activeControllerRef.current = null;
    }
  }

  async function branchCurrentChat() {
    if (!activeChat || selectedApiId !== "prompt") return;
    const key = instanceKey("prompt", activeChat.id);
    const existing = instancesRef.current.get(key) as LanguageModelSession | undefined;
    if (!existing?.clone) {
      showToast("clone() is not available on the active Prompt API session.", "error");
      return;
    }
    try {
      const clone = await existing.clone(buildCreateOptions("prompt", { renderPromptTemplate: true }));
      const newChat: Chat = {
        ...activeChat,
        id: generateId(),
        title: `${activeChat.title} branch`,
        timestamp: Date.now(),
        messages: activeChat.messages.map((message) => ({ ...message, id: generateId() })),
      };
      const next = [newChat, ...chats].slice(0, 50);
      const keptIds = new Set(next.map((item) => item.id));
      const removed = chats.filter((item) => !keptIds.has(item.id));
      deleteOrphanedVoiceAudioIds(removed.flatMap(voiceAudioIdsForChat), next);
      setChats(next);
      setActiveChatId(newChat.id);
      setSelectedApiId("prompt");
      const newKey = instanceKey("prompt", newChat.id);
      instancesRef.current.set(newKey, clone);
      signaturesRef.current.set(newKey, signaturesRef.current.get(key) || optionsSignature("prompt"));
      showToast("Chat branch created from cloned session.", "success");
    } catch (error) {
      showToast("Clone failed: " + stringify(error), "error");
    }
  }

  async function measurePromptUsage(text: string) {
    if (selectedApiId !== "prompt") {
      setPromptUsage("-");
      return;
    }
    const session = instancesRef.current.get(instanceKey("prompt")) as LanguageModelSession | undefined;
    if (!session || (!session.measureContextUsage && !session.measureInputUsage)) {
      const usage = session?.contextUsage ?? session?.inputUsage;
      const quota = session?.contextWindow ?? session?.inputQuota;
      setPromptUsage(usage != null ? `${usage}${quota ? ` / ${quota}` : ""}` : "-");
      return;
    }
    try {
      const value = session.measureContextUsage ? await session.measureContextUsage(text || " ") : await session.measureInputUsage?.(text || " ");
      const quota = session.contextWindow ?? session.inputQuota;
      setPromptUsage(`${value ?? 0}${quota ? ` / ${quota}` : ""}`);
    } catch {
      setPromptUsage("-");
    }
  }

  function renderMessage(message: ChatMessage) {
    const isAssistant = message.role === "assistant";
    const isVoiceUserMessage = message.role === "user" && message.media?.type === "voice";
    const isSpeaking = speakingMessageId === message.id;
    const isCopied = copiedMessageId === message.id;
    const speaker = message.role === "user" ? "You" : isAssistant ? "Chrome AI" : "System";
    const contentHtml = message.error ? `<span class="error-text">${escapeHtml(message.content)}</span>` : selectedApiMethodLabel(selectedApiId) === "proofread" && isAssistant ? parseMarkdown(message.content) : parseMarkdown(message.content);
    return (
      <div key={message.id} className={`message ${message.role} ${isVoiceUserMessage ? "voice-user-message" : ""} ${message.error ? "is-error" : ""} ${isSpeaking ? "is-speaking" : ""}`}>
        <div className="bubble-meta">
          <span>{speaker}</span>
          <span>{message.apiLabel}</span>
        </div>
        <div className="bubble">
          {message.media?.type === "voice" ? <VoiceMessage media={message.media} /> : null}
          {message.media?.type === "image" ? <div className="media-note">{message.media.count} image attachment{message.media.count === 1 ? "" : "s"}</div> : null}
          {isAssistant && message.apiLabel === "Proofreader API" && message.content.includes("Corrected input") ? (
            <div dangerouslySetInnerHTML={{ __html: renderProofreadHtml({ correctedInput: message.content }, "") }} />
          ) : (
            <div dangerouslySetInnerHTML={{ __html: contentHtml }} />
          )}
        </div>
        <div className="message-actions">
          {isAssistant ? (
            <button className={`btn-icon speak-response-btn ${isSpeaking ? "is-speaking" : ""}`} type="button" onClick={() => toggleMessageSpeech(message)} title={isSpeaking ? "Stop speaking" : "Play response"} aria-label={isSpeaking ? "Stop speaking" : "Play response"} disabled={!message.content.trim() || !supportsSpeechSynthesisOutput()}>
              {isSpeaking ? <StopResponseIcon /> : <PlayResponseIcon />}
            </button>
          ) : null}
          <button className="btn-icon copy-btn" type="button" onClick={() => copyMessageText(message)} title="Copy text" aria-label="Copy text">
            {isCopied ? <CheckIcon /> : <CopyIcon />}
          </button>
        </div>
      </div>
    );
  }

  function renderRuntimeRows(): [string, unknown][] {
    const exposedCount = API_CONFIGS.filter((api) => typeof window !== "undefined" && typeof window[api.globalName] === "function").length;
    return [
      ["Secure context", String(window.isSecureContext)],
      ["Protocol", location.protocol],
      ["Origin", location.protocol === "file:" ? "file://" : location.origin],
      ["Chrome", chromeVersion()],
      ["APIs exposed", `${exposedCount}/${API_CONFIGS.length}`],
      ["User activation", navigator.userActivation ? String(navigator.userActivation.isActive) : "unknown"],
    ];
  }

  function renderOptionsPanel() {
    const languageOptions: [string, string][] = [["en", "en"], ["es", "es"], ["ja", "ja"], ["auto", "default"]];
    const defaultTemperature = numberParamValue(languageModelParams, "defaultTemperature");
    const maxTemperature = numberParamValue(languageModelParams, "maxTemperature");
    const defaultTopK = numberParamValue(languageModelParams, "defaultTopK");
    const maxTopK = numberParamValue(languageModelParams, "maxTopK");
    const textLanguageControls = (includeContext = true) => (
      <>
        <div className="two">
          <SelectField label="Input language" value={options.inputLanguage} onChange={(value) => updateOptions("inputLanguage", value)} options={languageOptions} hint="Maps to expectedInputLanguages or text expectedInputs languages." />
          <SelectField label="Output language" value={options.outputLanguage} onChange={(value) => updateOptions("outputLanguage", value)} options={languageOptions} hint="Maps to outputLanguage or expectedOutputs text languages." />
        </div>
        {includeContext ? (
          <SelectField label="Expected context language" value={options.contextLanguage} onChange={(value) => updateOptions("contextLanguage", value)} options={languageOptions} hint="Maps to expectedContextLanguages for shared and per-request context." />
        ) : null}
      </>
    );

    if (selectedApiId === "prompt") {
      return (
        <>
          <Section title="Official API parameters">
            {textLanguageControls(false)}
            <Toggle label="Accept image input" checked={options.includeImage} onChange={(value) => updateOptions("includeImage", value)} hint={'Adds { type: "image" } to expectedInputs. Also enabled automatically when an image is attached.'} />
            <Toggle label="Accept audio input" checked={options.includeAudio} onChange={(value) => updateOptions("includeAudio", value)} hint={'Adds { type: "audio" } to expectedInputs. Required for Voice messages.'} />
            <TextAreaField label="System prompt" value={options.systemPrompt} onChange={(value) => updateOptions("systemPrompt", value)} hint="Passed as a system initialPrompt." />
            <div className="small">Expected output modality: <span className="kv-val">text</span>.</div>
          </Section>
          <Section title="LanguageModel sampling params">
            <div className="two">
              <InputField
                label="Temperature"
                value={options.promptTemperature}
                onChange={(value) => updateOptions("promptTemperature", value)}
                type="number"
                min={0}
                max={maxTemperature}
                step={0.1}
                placeholder={defaultTemperature == null ? "Chrome default" : `default ${defaultTemperature}`}
                hint="Passed to LanguageModel.create() as temperature. Leave blank for Chrome default."
              />
              <InputField
                label="Top K"
                value={options.promptTopK}
                onChange={(value) => updateOptions("promptTopK", value)}
                type="number"
                min={1}
                max={maxTopK}
                step={1}
                placeholder={defaultTopK == null ? "Chrome default" : `default ${defaultTopK}`}
                hint="Passed to LanguageModel.create() as topK. Leave blank for Chrome default."
              />
            </div>
            <div className="small">
              Reported defaults: temperature <span className="kv-val">{defaultTemperature ?? "-"}</span>, topK <span className="kv-val">{defaultTopK ?? "-"}</span>. Max values: temperature <span className="kv-val">{maxTemperature ?? "-"}</span>, topK <span className="kv-val">{maxTopK ?? "-"}</span>.
            </div>
            <div className="small">Chrome documents these sampling controls as available only where this Prompt API build allows them, such as extensions or the sampling origin trial.</div>
          </Section>
          <Section title="Official request options">
            <Toggle label="Use streaming output" checked={options.stream} onChange={(value) => updateOptions("stream", value)} hint="Calls promptStreaming() when enabled, otherwise prompt()." />
            <TextAreaField label="JSON response schema" value={options.promptJsonSchema} onChange={(value) => updateOptions("promptJsonSchema", value)} hint="Passed as responseConstraint to prompt() or promptStreaming()." />
            <span className={`json-status ${schemaStatus === "valid JSON" ? "success-text" : schemaStatus === "invalid JSON" ? "error-text" : ""}`}>{schemaStatus}</span>
            <Toggle label="Omit schema from model input" checked={options.promptOmitResponseConstraintInput} onChange={(value) => updateOptions("promptOmitResponseConstraintInput", value)} hint="Advanced Prompt API option: maps to omitResponseConstraintInput when a schema is provided." />
          </Section>
          <Section title="App behavior">
            <Toggle label="Keep Prompt session between messages" checked={options.keepPromptSession} onChange={(value) => updateOptions("keepPromptSession", value)} hint="Local app behavior. Reuses the same LanguageModel session while creation options match." />
            <div className="speech-options">
              <InputField label="Max recording seconds" value={options.voiceMaxSeconds} onChange={(value) => updateOptions("voiceMaxSeconds", value)} type="number" min={3} max={60} step={1} hint="Local MediaRecorder limit; not a Prompt API parameter." />
              <div className="small">Audio input availability: <span className="kv-val">{speechAvailabilityLabel()}</span></div>
            </div>
          </Section>
          <OutputVoiceControls />
        </>
      );
    }
    if (selectedApiId === "summarizer") {
      return (
        <>
          <Section title="Official API parameters">
            <div className="two">
              <SelectField label="Type" value={options.summarizerType} onChange={(value) => updateOptions("summarizerType", value)} options={[["key-points", "key-points"], ["tldr", "tldr"], ["teaser", "teaser"], ["headline", "headline"]]} />
              <SelectField label="Length" value={options.summarizerLength} onChange={(value) => updateOptions("summarizerLength", value)} options={[["short", "short"], ["medium", "medium"], ["long", "long"]]} />
            </div>
            <div className="two">
              <SelectField label="Format" value={options.summarizerFormat} onChange={(value) => updateOptions("summarizerFormat", value)} options={[["markdown", "markdown"], ["plain-text", "plain-text"]]} />
              <SelectField label="Preference" value={options.summarizerPreference} onChange={(value) => updateOptions("summarizerPreference", value)} options={[["auto", "auto"], ["speed", "speed"], ["capability", "capability"]]} />
            </div>
            {textLanguageControls(true)}
            <TextAreaField label="Shared context" value={options.sharedContext} onChange={(value) => updateOptions("sharedContext", value)} hint="Passed to Summarizer.create() for repeated summarization tasks." />
            <div className="small">Official length mapping: key-points use 3/5/7 bullets; tldr and teaser use 1/3/5 sentences; headline uses 12/17/22 words.</div>
          </Section>
          <RequestOptions method="summarize" />
        </>
      );
    }
    if (selectedApiId === "writer") {
      return (
        <>
          <Section title="Official API parameters">
            <div className="two">
              <SelectField label="Tone" value={options.writerTone} onChange={(value) => updateOptions("writerTone", value)} options={[["formal", "formal"], ["neutral", "neutral"], ["casual", "casual"]]} />
              <SelectField label="Length" value={options.writerLength} onChange={(value) => updateOptions("writerLength", value)} options={[["short", "short"], ["medium", "medium"], ["long", "long"]]} />
            </div>
            <SelectField label="Format" value={options.writerFormat} onChange={(value) => updateOptions("writerFormat", value)} options={[["markdown", "markdown"], ["plain-text", "plain-text"]]} />
            {textLanguageControls(true)}
            <TextAreaField label="Shared context" value={options.sharedContext} onChange={(value) => updateOptions("sharedContext", value)} hint="Passed to Writer.create() for repeated writing tasks." />
          </Section>
          <RequestOptions method="write" />
        </>
      );
    }
    if (selectedApiId === "rewriter") {
      return (
        <>
          <Section title="Official API parameters">
            <div className="two">
              <SelectField label="Tone" value={options.rewriterTone} onChange={(value) => updateOptions("rewriterTone", value)} options={[["more-formal", "more-formal"], ["as-is", "as-is"], ["more-casual", "more-casual"]]} />
              <SelectField label="Length" value={options.rewriterLength} onChange={(value) => updateOptions("rewriterLength", value)} options={[["shorter", "shorter"], ["as-is", "as-is"], ["longer", "longer"]]} />
            </div>
            <SelectField label="Format" value={options.rewriterFormat} onChange={(value) => updateOptions("rewriterFormat", value)} options={[["as-is", "as-is"], ["markdown", "markdown"], ["plain-text", "plain-text"]]} />
            {textLanguageControls(true)}
            <TextAreaField label="Shared context" value={options.sharedContext} onChange={(value) => updateOptions("sharedContext", value)} hint="Passed to Rewriter.create() for repeated rewriting tasks." />
          </Section>
          <RequestOptions method="rewrite" />
        </>
      );
    }
    return (
      <Section title="Official API parameters">
        <SelectField label="Expected input language" value={options.proofreaderInputLanguage} onChange={(value) => updateOptions("proofreaderInputLanguage", value)} options={[["en", "en"]]} hint="Maps to expectedInputLanguages." />
        <div className="small">Proofreader exposes <span className="kv-val">proofread()</span> only. The official docs do not document streaming or correction-type/explanation controls.</div>
      </Section>
    );
  }

  function RequestOptions({ method }: { method: string }) {
    return (
      <Section title="Official request options">
        <Toggle label="Use streaming output" checked={options.stream} onChange={(value) => updateOptions("stream", value)} hint={`Calls ${method}Streaming() when enabled, otherwise ${method}().`} />
        <TextAreaField label="Per-request context" value={options.requestContext} onChange={(value) => updateOptions("requestContext", value)} hint="Passed as the documented context option for this request only." />
      </Section>
    );
  }

  function OutputVoiceControls() {
    const supported = supportsSpeechSynthesisOutput();
    const selectedId = options.outputVoiceId;
    const hasSelectedVoice = selectedId && selectedId !== "auto" && outputVoices.some((voice) => outputVoiceId(voice) === selectedId);
    const preferred = outputVoices.filter(voiceMatchesSelectedLanguage);
    const other = outputVoices.filter((voice) => !voiceMatchesSelectedLanguage(voice));
    const supportText = supported
      ? `${outputVoices.length || "No"} browser voices detected.`
      : "Speech synthesis is not available in this browser.";
    return (
      <Section title="Output voice">
        <Toggle label="Speak responses after Voice messages" checked={options.voiceAutoSpeak} onChange={(value) => updateOptions("voiceAutoSpeak", value)} hint="Uses local browser speech synthesis after a Voice message receives an assistant response." />
        <label className="field">
          Output voice
          <select name="outputVoiceId" value={options.outputVoiceId} disabled={!supported} onChange={(event) => updateOptions("outputVoiceId", event.target.value)}>
            <option value="auto">Auto: Chrome default for output language</option>
            {selectedId && selectedId !== "auto" && !hasSelectedVoice ? <option value={selectedId}>Saved voice unavailable on this device</option> : null}
            {!outputVoices.length ? <option value="" disabled>{supported && !outputVoicesLoaded ? "Loading system voices..." : "No browser voices reported"}</option> : null}
            {preferred.length ? (
              <optgroup label={`Matching ${selectedOutputLanguage()} voices`}>
                {preferred.map((voice) => <option key={outputVoiceId(voice)} value={outputVoiceId(voice)}>{outputVoiceLabel(voice)}</option>)}
              </optgroup>
            ) : null}
            {other.length ? (
              <optgroup label="Other browser voices">
                {other.map((voice) => <option key={outputVoiceId(voice)} value={outputVoiceId(voice)}>{outputVoiceLabel(voice)}</option>)}
              </optgroup>
            ) : null}
          </select>
          <span className="field-hint">Local browser/OS voice used by Web Speech synthesis. Voice availability depends on this device and Chrome profile.</span>
        </label>
        <div className="two">
          <InputField label="Rate" value={options.outputVoiceRate} onChange={(value) => updateOptions("outputVoiceRate", value)} type="number" min={0.5} max={2} step={0.1} hint="1 is normal speed." />
          <InputField label="Pitch" value={options.outputVoicePitch} onChange={(value) => updateOptions("outputVoicePitch", value)} type="number" min={0} max={2} step={0.1} hint="1 is normal pitch." />
        </div>
        <InputField label="Volume" value={options.outputVoiceVolume} onChange={(value) => updateOptions("outputVoiceVolume", value)} type="number" min={0} max={1} step={0.05} hint="1 is full volume." />
        <div className="voice-output-actions">
          <button type="button" className="btn" disabled={!supported} onClick={() => speakText("This is the selected output voice.", "test-output-voice").catch((error) => showToast(stringify(error), "error"))}>Test output voice</button>
        </div>
        <div className="small">{supportText} These controls are local app behavior, not Prompt API inference parameters.</div>
      </Section>
    );
  }

  function submitComposer(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    runSelected(promptInput).catch((error) => showToast(stringify(error), "error"));
  }

  if (!hydrated) {
    return <div className="loading-shell">Loading Browser AI...</div>;
  }

  const exposedCount = API_CONFIGS.filter((api) => typeof window[api.globalName] === "function").length;
  const availableCount = API_CONFIGS.filter((api) => statuses[api.id]?.availability === "available").length;
  const hasComposerContent = promptInput.trim().length > 0 || imageFiles.length > 0;
  const showPromptVoiceAction = selectedApiId === "prompt" && !running && !hasComposerContent;
  const showSendAction = running || selectedApiId !== "prompt" || hasComposerContent;
  const welcomeMessage: ChatMessage | null =
    activeChat && !activeChat.messages.length && welcome?.chatId === activeChat.id
      ? { id: "initial-welcome", role: "assistant", content: welcome.content, apiLabel: "System", timestamp: Date.now() }
      : null;

  return (
    <div className="browser-ai-shell">
      <header className="topbar">
        <div className="top-left">
          <button className="icon-btn" type="button" onClick={() => {
            if (window.innerWidth <= MOBILE_LAYOUT_BREAKPOINT) setMobilePane(mobilePane === "left" ? "" : "left");
            else {
              const next = !leftClosed;
              setLeftClosed(next);
              localStorage.setItem(LEFT_STORAGE_KEY, String(next));
            }
          }} title="Toggle Sidebar"><MenuIcon /></button>
          <div className="brand">
            <div className="mark" aria-hidden="true"><GeminiMark size={28} /></div>
            <h1>Chrome AI</h1>
          </div>
        </div>
        <div className="top-actions">
          <button className="icon-btn" type="button" onClick={() => showToast(`Chrome ${chromeVersion()}`)} title="Environment Info"><InfoIcon size={18} /></button>
          <button className="icon-btn" type="button" onClick={() => {
            if (window.innerWidth <= MOBILE_LAYOUT_BREAKPOINT) setMobilePane(mobilePane === "right" ? "" : "right");
            else {
              const next = !rightClosed;
              setRightClosed(next);
              localStorage.setItem(RIGHT_STORAGE_KEY, String(next));
            }
          }} title="Toggle Inspector"><PanelIcon /></button>
        </div>
      </header>

      <main className={`app-grid ${leftClosed ? "hide-left" : ""} ${rightClosed ? "hide-right" : ""} ${mobilePane === "left" ? "show-left" : ""} ${mobilePane === "right" ? "show-right" : ""}`}>
        <aside className="panel side-panel">
          <div className="side-panel-actions">
            <button className="side-new-chat-btn" type="button" onClick={createNewChat}>
              <PencilIcon />
              <span>New Chat</span>
            </button>
          </div>
          <div className="history-list">
            {chats.map((chat) => (
              <div className={`chat-item ${chat.id === activeChatId ? "active" : ""}`} key={chat.id}>
                <button className="chat-item-content" type="button" onClick={() => selectChat(chat)}>
                  <span>{chat.title}</span>
                  <small>{new Date(chat.timestamp).toLocaleDateString()} {new Date(chat.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small>
                </button>
                <button className="chat-delete" type="button" onClick={() => deleteChat(chat.id)} title="Delete chat">×</button>
              </div>
            ))}
          </div>
        </aside>
        <section className={`panel chat-panel ${activeChat?.messages.length ? "" : "is-empty"}`}>
          <div className="transcript" ref={transcriptRef}>
            <div className="transcript-inner">
              {activeChat?.messages.length ? (
                activeChat.messages.map(renderMessage)
              ) : (
                <>
                  <div className="empty">
                    <div className="empty-mark-wrap"><GeminiMark size={48} /></div>
                    <h2>Chrome Built-in AI Playground</h2>
                    <p>Experiment with Local Gemini Nano APIs directly in your browser.<br />Select an API below and send a prompt to get started.</p>
                  </div>
                  {welcomeMessage ? renderMessage(welcomeMessage) : null}
                </>
              )}
            </div>
          </div>
          {selectedProgress?.kind === "download" && selectedProgress.showBanner ? (
            <div className="download-progress-wrap">
              <div className="download-progress-header">
                <GeminiMark size={16} />
                <span>Downloading on-device model</span>
              </div>
              <div className="download-progress-bar">
                <div className="download-progress-fill" style={{ width: `${Math.round((selectedProgress.ratio || 0) * 100)}%` }} />
              </div>
              <div className="download-progress-text">
                <span>{selectedProgress.text}</span>
                <span className="download-progress-pct">{Math.round((selectedProgress.ratio || 0) * 100)}%</span>
              </div>
            </div>
          ) : null}
          <div className="composer-wrap">
            <div className="composer-container">
              <div className="composer-toolbar">
                <select className="api-selector" name="apiSelector" value={selectedApiId} onChange={(event) => onApiChange(event.target.value as ApiId)} aria-label="API selector">
                  {API_CONFIGS.map((api) => <option key={api.id} value={api.id}>{api.label}</option>)}
                </select>
                <div className="api-tooltip-container">
                  <button type="button" className="icon-btn api-info-btn" title="API Info"><InfoIcon size={14} /></button>
                  <div className="api-tooltip">{selectedApi.description}</div>
                </div>
                <div className="toolbar-spacer" />
                <button className="new-chat-btn" type="button" onClick={createNewChat} title="New Chat" aria-label="New Chat">
                  <PencilIcon />
                  <span>New Chat</span>
                </button>
              </div>
              {imagePreviews.length ? (
                <div className="image-preview">
                  {imagePreviews.map((url, index) => (
                    <div className="image-preview-item" key={url}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={url} alt={`Attachment ${index + 1}`} />
                      <button className="image-preview-remove" type="button" onClick={() => setImageFiles((files) => files.filter((_file, fileIndex) => fileIndex !== index))}>×</button>
                    </div>
                  ))}
                </div>
              ) : null}
              {speechStatus ? <div className="speech-status">{speechStatus}</div> : selectedApiId === "prompt" && speechAvailability !== "available" ? <div className="speech-status muted">{speechAvailabilityLabel()}</div> : null}
              <form className={`composer ${selectedApiId === "prompt" ? "" : "is-text-only"}`} onSubmit={submitComposer}>
                {selectedApiId === "prompt" ? (
                  <>
                    <label className="attach-btn" title="Attach image">
                      <PlusIcon />
                      <input type="file" name="imageInput" accept="image/*" multiple hidden onChange={(event) => {
                        const files = Array.from(event.target.files || []);
                        if (files.length) {
                          setImageFiles(files);
                          updateOptions("includeImage", true);
                        }
                        event.target.value = "";
                      }} />
                    </label>
                    <button className={`speech-btn ${speechMode === "dictation" ? "is-recording" : ""} ${speechProcessing && speechMode === "dictation" ? "is-processing" : ""}`} type="button" disabled={running || speechProcessing || (!recorderRef.current && !canStartSpeechInput())} onClick={() => toggleSpeechRecording("dictation")} title="Start dictation"><MicIcon /></button>
                  </>
                ) : null}
                <textarea name="promptInput" value={promptInput} onInput={(event) => setPromptInput(event.currentTarget.value)} onChange={(event) => setPromptInput(event.target.value)} onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }} placeholder={composerPlaceholder(selectedApiId)} rows={1} />
                {showPromptVoiceAction ? (
                  <button className={`voice-btn ${speechMode === "voice" ? "is-recording" : ""} ${speechProcessing && speechMode === "voice" ? "is-processing" : ""}`} type="button" disabled={running || speechProcessing || (!recorderRef.current && !canStartSpeechInput())} onClick={() => toggleSpeechRecording("voice")} title="Start Voice" aria-label="Start Voice"><WaveIcon /><span>Voice</span></button>
                ) : null}
                {showSendAction ? (
                  <button className={`send-btn ${running ? "is-stop" : hasComposerContent ? "has-content" : ""}`} disabled={!running && !hasComposerContent} type={running ? "button" : "submit"} onClick={running ? stopRun : undefined} title={running ? "Stop generating" : "Send"}>
                    {running ? <StopIcon /> : <ArrowUpIcon />}
                  </button>
                ) : null}
              </form>
            </div>
          </div>
        </section>
        <aside className="panel inspector">
          <div className="panel-head">
            <div className="inspector-title-row">
              <div className="inspector-title-left">
                <h2>Inspector</h2>
                <span className="pill"><span className={`status-dot ${availableCount ? "dot-available" : exposedCount ? "dot-downloadable" : "dot-missing"}`} /><strong>{exposedCount}/{API_CONFIGS.length} exposed</strong></span>
              </div>
              <span className={`badge ${selectedStatus.availability}`}>{selectedStatus.availability}</span>
            </div>
            <div className="small">{selectedApi.model} via {selectedApi.globalName}</div>
          </div>
          <div className="inspector-body">
            <Section title="Parameters">{renderOptionsPanel()}</Section>
            <Section title="Instance status">
              <Kv rows={[
                ["Global", selectedApi.globalName],
                ["Availability", selectedStatus.availability],
                ["Instance", selectedInstance ? "created" : "not created"],
                ["Progress", selectedProgress?.text || "-"],
                ["Prompt usage", promptUsage],
                ["Audio input", selectedApiId === "prompt" ? speechAvailabilityLabel() : "-"],
                ["Error", selectedStatus.error || "-"],
              ]} />
              <div className="action-row">
                <button className="btn" type="button" disabled={running || !selectedStatus.exists} onClick={() => getOrCreateInstance(selectedApiId, true).then(() => showToast(`${selectedApi.label} created.`, "success")).catch((error) => showToast(stringify(error), "error"))}>Create</button>
                <button className="btn" type="button" disabled={running || !selectedInstance} onClick={() => safeDestroy(selectedApiId).then(() => showToast(`${selectedApi.label} destroyed.`))}>Destroy</button>
              </div>
              {selectedApiId === "prompt" ? (
                <div className="action-row">
                  <button className="btn" type="button" disabled={!promptInput.trim() || running} onClick={appendCurrentPrompt}>Append</button>
                  <button className="btn" type="button" disabled={!selectedInstance || running} onClick={branchCurrentChat}>Branch</button>
                </div>
              ) : null}
            </Section>

            <Section title="Environment">
              <Kv rows={renderRuntimeRows()} />
            </Section>
          </div>
        </aside>
      </main>
      <div className="toast">
        {toast.map((item) => <div className={`toast-item ${item.tone}`} key={item.id}>{item.message}</div>)}
      </div>
    </div>
  );
}

function createBlankChat(apiId: ApiId): Chat {
  return { id: generateId(), title: "New Chat", apiId, timestamp: Date.now(), messages: [] };
}

function voiceAudioIdsForChat(chat?: Chat): string[] {
  if (!chat) return [];
  return chat.messages.map((message) => (message.media?.type === "voice" ? message.media.audioId : "")).filter(Boolean);
}

function deleteOrphanedVoiceAudioIds(ids: string[], remainingChats: Chat[]) {
  const stillReferenced = new Set(remainingChats.flatMap(voiceAudioIdsForChat));
  Array.from(new Set(ids))
    .filter((id) => id && !stillReferenced.has(id))
    .forEach((id) => deleteVoiceAudio(id).catch(() => undefined));
}

function GeminiMark({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="url(#gemini-gradient)" stroke="none" aria-hidden="true">
      <defs>
        <linearGradient id="gemini-gradient" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#2563eb" />
          <stop offset="100%" stopColor="#db2777" />
        </linearGradient>
      </defs>
      <path d="M12 2.5C12.5 7.5 16.5 11.5 21.5 12C16.5 12.5 12.5 16.5 12 21.5C11.5 16.5 7.5 12.5 2.5 12C7.5 11.5 11.5 7.5 12 2.5Z" />
    </svg>
  );
}

function MenuIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  );
}

function InfoIcon({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  );
}

function PanelIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <line x1="15" y1="3" x2="15" y2="21" />
    </svg>
  );
}

function PencilIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
      <path d="M19 10v1a7 7 0 0 1-14 0v-1" />
      <line x1="12" y1="18" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
    </svg>
  );
}

function WaveIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M2 10v4" />
      <path d="M6 7v10" />
      <path d="M10 4v16" />
      <path d="M14 7v10" />
      <path d="M18 10v4" />
      <path d="M22 11v2" />
    </svg>
  );
}

function ArrowUpIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="12" y1="19" x2="12" y2="5" />
      <polyline points="5 12 12 5 19 12" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="6" y="6" width="12" height="12" rx="2" ry="2" />
    </svg>
  );
}

function PlayResponseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polygon points="6 3 20 12 6 21 6 3" />
    </svg>
  );
}

function StopResponseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="6" y="6" width="12" height="12" rx="1" />
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function Kv({ rows }: { rows: [string, unknown][] }) {
  return (
    <div className="kv">
      {rows.map(([key, value]) => (
        <div className="kv-row" key={key}>
          <span className="kv-key">{key}</span>
          <span className="kv-val">{value == null || value === "" ? "-" : String(value)}</span>
        </div>
      ))}
    </div>
  );
}

function Toggle({ label, checked, onChange, hint = "" }: { label: string; checked: boolean; onChange: (value: boolean) => void; hint?: string }) {
  return (
    <label className="toggle-row">
      <input type="checkbox" name={fieldName(label)} checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="toggle-copy">
        <span>{label}</span>
        {hint ? <span className="field-hint">{hint}</span> : null}
      </span>
    </label>
  );
}

function SelectField({ label, value, options, onChange, hint = "" }: { label: string; value: string; options: [string, string][]; onChange: (value: string) => void; hint?: string }) {
  return (
    <label className="field">
      {label}
      <select name={fieldName(label)} value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map(([optionValue, text]) => <option key={optionValue} value={optionValue}>{text}</option>)}
      </select>
      {hint ? <span className="field-hint">{hint}</span> : null}
    </label>
  );
}

function InputField({
  label,
  value,
  onChange,
  type = "text",
  placeholder = "",
  min,
  max,
  step,
  hint = "",
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: string;
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
}) {
  return (
    <label className="field">
      {label}
      <input name={fieldName(label)} type={type} value={value} placeholder={placeholder} min={min} max={max} step={step} onChange={(event) => onChange(event.target.value)} />
      {hint ? <span className="field-hint">{hint}</span> : null}
    </label>
  );
}

function TextAreaField({ label, value, onChange, hint = "" }: { label: string; value: string; onChange: (value: string) => void; hint?: string }) {
  return (
    <label className="field">
      {label}
      <textarea name={fieldName(label)} value={value} onChange={(event) => onChange(event.target.value)} />
      {hint ? <span className="field-hint">{hint}</span> : null}
    </label>
  );
}

function fieldName(label: string): string {
  return label.replace(/[^a-z0-9]+/gi, "_").replace(/^_+|_+$/g, "").toLowerCase();
}
