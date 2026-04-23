import { useEffect, useRef, useState } from 'react';
import { Loader2, MessageCircle, Mic, Send, X } from 'lucide-react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';

interface Message {
  role: 'user' | 'bot';
  text: string;
  isAudio?: boolean;
}

type LeadStage = 'chat';


const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const FLOATING_BOT_IMAGE_URL = import.meta.env.VITE_FLOATING_BOT_IMAGE_URL || '';

const SESSION_STORAGE_KEY = 'vtl_session_id';

const decodeHeaderValue = (value: string | null): string => {
  if (!value) {
    return '';
  }

  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
};

const normalizeMarkdownText = (text: string): string => {
  const normalized = (text || '').replace(/\r\n?/g, '\n');
  const fenceCount = (normalized.match(/(^|\n)```/g) || []).length;
  if (fenceCount % 2 === 1) {
    return `${normalized}\n\n\`\`\``;
  }
  return normalized;
};

function MarkdownMessage({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeSanitize]}
      className="vtl-markdown"
      components={{
        a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
      }}
    >
      {normalizeMarkdownText(text)}
    </ReactMarkdown>
  );
}

const isValidEmail = (email: string): boolean => {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
};

const createSessionId = (): string => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID().replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 64);
  }
  return `session_${Date.now()}`;
};

function App() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreamingResponse, setIsStreamingResponse] = useState(false);
  const [isWaitingForFirstToken, setIsWaitingForFirstToken] = useState(false);
  const [floatingImageError, setFloatingImageError] = useState(false);

  const [sessionId, setSessionId] = useState('');
  const [leadStage, setLeadStage] = useState<LeadStage>('chat');

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const storedSessionId = localStorage.getItem(SESSION_STORAGE_KEY)?.trim();

    const resolvedSessionId = storedSessionId || createSessionId();
    setSessionId(resolvedSessionId);
    localStorage.setItem(SESSION_STORAGE_KEY, resolvedSessionId);

    setLeadStage('chat');
    setMessages([
      {
        role: 'bot',
        text: `Welcome! How can I help you today?`,
      },
    ]);
  }, []);

  // Storage logic for email/name removed


  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  const appendToLatestBotMessage = (chunk: string) => {
    if (!chunk) {
      return;
    }

    setMessages((prev) => {
      const next = [...prev];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === 'bot') {
          next[index] = { ...next[index], text: `${next[index].text}${chunk}` };
          return next;
        }
      }
      return [...next, { role: 'bot', text: chunk }];
    });
  };

  const setLatestBotMessageText = (text: string) => {
    setMessages((prev) => {
      const next = [...prev];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === 'bot') {
          next[index] = { ...next[index], text };
          return next;
        }
      }
      return [...next, { role: 'bot', text }];
    });
  };

  const submitUserMessage = async (rawMessage: string) => {
    if (!rawMessage.trim()) {
      return;
    }

    const userMsg = rawMessage.trim();
    setMessages((prev) => [...prev, { role: 'user', text: userMsg }]);

    // Email and Name stage logic removed


    setIsLoading(true);
    setIsStreamingResponse(true);
    setIsWaitingForFirstToken(true);
    setMessages((prev) => [...prev, { role: 'bot', text: '' }]);

    try {
      const formData = new FormData();
      formData.append('query', userMsg);
      formData.append('session_id', sessionId);

      const response = await fetch(`${API_BASE_URL}/api/chat/text/stream`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok || !response.body) {
        throw new Error('Streaming API failed');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let streamedText = '';
      let resolvedSessionId = sessionId;

      const processEvent = (eventType: string, payload: string) => {
        let parsed: unknown;
        try {
          parsed = JSON.parse(payload);
        } catch {
          return;
        }

        if (!parsed || typeof parsed !== 'object') {
          return;
        }

        const data = parsed as {
          token?: unknown;
          reply?: unknown;
          session_id?: unknown;
          message?: unknown;
        };

        if (eventType === 'token') {
          const token = typeof data.token === 'string' ? data.token : '';
          if (token) {
            setIsWaitingForFirstToken(false);
            streamedText += token;
            appendToLatestBotMessage(token);
          }
          return;
        }

        if (eventType === 'done') {
          setIsWaitingForFirstToken(false);
          const doneReply = typeof data.reply === 'string' ? data.reply : '';
          if (!streamedText.trim() && doneReply) {
            streamedText = doneReply;
            setLatestBotMessageText(doneReply);
          }

          if (typeof data.session_id === 'string' && data.session_id.trim()) {
            resolvedSessionId = data.session_id.trim();
          }
          return;
        }

        if (eventType === 'error') {
          const errorMessage = typeof data.message === 'string'
            ? data.message
            : 'Sorry, an error occurred while streaming the response.';
          throw new Error(errorMessage);
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          buffer += decoder.decode();
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        let splitIndex = buffer.indexOf('\n\n');
        while (splitIndex !== -1) {
          const rawEvent = buffer.slice(0, splitIndex);
          buffer = buffer.slice(splitIndex + 2);

          const lines = rawEvent.replace(/\r/g, '').split('\n');
          let eventType = 'message';
          const dataLines: string[] = [];

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
              continue;
            }
            if (line.startsWith('data:')) {
              dataLines.push(line.slice(5).trim());
            }
          }

          if (dataLines.length > 0) {
            processEvent(eventType, dataLines.join('\n'));
          }

          splitIndex = buffer.indexOf('\n\n');
        }
      }

      // Handle a final event block if stream closes without trailing delimiter.
      const trailingEvent = buffer.trim();
      if (trailingEvent) {
        const lines = trailingEvent.replace(/\r/g, '').split('\n');
        let eventType = 'message';
        const dataLines: string[] = [];

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim();
            continue;
          }
          if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim());
          }
        }

        if (dataLines.length > 0) {
          processEvent(eventType, dataLines.join('\n'));
        }
      }

      if (!streamedText.trim()) {
        throw new Error('Empty streamed response');
      }

      if (resolvedSessionId !== sessionId) {
        setSessionId(resolvedSessionId);
        localStorage.setItem(SESSION_STORAGE_KEY, resolvedSessionId);
      }
    } catch {
      setLatestBotMessageText('Sorry, an error occurred while streaming the response.');
    } finally {
      setIsWaitingForFirstToken(false);
      setIsStreamingResponse(false);
      setIsLoading(false);
    }
  };

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) {
      return;
    }

    const messageToSend = inputText;
    setInputText('');
    await submitUserMessage(messageToSend);
  };

  const startRecording = async () => {
    if (isLoading || isRecording || leadStage !== 'chat') {
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = handleAudioStop;
      mediaRecorder.start();
      setIsRecording(true);
    } catch {
      alert('Please allow microphone access to use voice features.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
    }
  };

  const handleAudioStop = async () => {
    setIsLoading(true);
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });

    const formData = new FormData();
    formData.append('audio', audioFile);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/voice`, {
        method: 'POST',
        headers: {
          'X-Session-Id': sessionId,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Voice API failed');
      }

      let userQuery = decodeHeaderValue(response.headers.get('X-User-Query-Encoded'))
        || response.headers.get('X-User-Query')
        || 'Voice Message';
      let botReply = decodeHeaderValue(response.headers.get('X-Bot-Reply-Encoded'))
        || response.headers.get('X-Bot-Reply')
        || 'Audio Reply';

      try {
        const lastTurnResponse = await axios.get(`${API_BASE_URL}/api/chat/last`, {
          params: { session_id: sessionId },
        });
        userQuery = lastTurnResponse.data.user_query || userQuery;
        botReply = lastTurnResponse.data.reply || botReply;
      } catch {
        // Keep header-based fallbacks when last-turn lookup is unavailable.
      }

      setMessages((prev) => [
        ...prev,
        { role: 'user', text: userQuery, isAudio: true },
        { role: 'bot', text: botReply, isAudio: true },
      ]);

      const audioResponseBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioResponseBlob);
      const audio = new Audio(audioUrl);
      await audio.play();
      audio.onended = () => URL.revokeObjectURL(audioUrl);
    } catch {
      setMessages((prev) => [...prev, { role: 'bot', text: 'Sorry, failed to process audio.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const voiceHintText = isRecording
    ? '🔴 Recording... release to send'
    : 'Hold mic to record • Release to send';

  const showFloatingImage = !isOpen && !!FLOATING_BOT_IMAGE_URL && !floatingImageError;

  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`fixed z-50 transition-transform hover:scale-105 flex items-center justify-center ${
          isOpen ? 'top-3 right-3 sm:top-auto sm:bottom-6 sm:right-6' : 'bottom-4 right-4 sm:bottom-6 sm:right-6'
        } ${
          showFloatingImage
            ? 'w-16 h-16 sm:w-[110px] sm:h-[110px] rounded-full bg-transparent shadow-none overflow-hidden p-0'
            : 'p-3 sm:p-4 vtl-brand-gradient text-white rounded-full shadow-2xl hover:brightness-95'
        }`}
      >
        {isOpen ? (
          <X className="w-5 h-5 sm:w-6 sm:h-6" />
        ) : showFloatingImage ? (
          <img
            src={FLOATING_BOT_IMAGE_URL}
            alt="Assistant"
            className="w-full h-full object-cover object-center rounded-full"
            onError={() => setFloatingImageError(true)}
          />
        ) : (
          <MessageCircle className="w-5 h-5 sm:w-6 sm:h-6" />
        )}
      </button>

      {isOpen && (
        <div className="fixed inset-x-0 top-0 bottom-0 sm:inset-auto sm:bottom-24 sm:right-6 z-40 w-full sm:w-[min(540px,94vw)] h-full sm:h-[760px] sm:max-h-[85vh] bg-[var(--vtl-panel)] rounded-none sm:rounded-2xl shadow-2xl flex flex-col border border-[var(--vtl-border)] overflow-hidden">
          <div className="vtl-brand-gradient p-3 sm:p-4 text-white font-bold text-base sm:text-lg flex justify-between items-center shadow-md z-10">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-[var(--vtl-accent)] rounded-full animate-pulse"></div>
              <span>DIGICoCo Assistant</span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-4 bg-[var(--vtl-surface)]">
            {messages.map((msg, idx) => (
              <div key={`${msg.role}-${idx}`} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`p-3 rounded-2xl text-sm max-w-[90%] sm:max-w-[85%] shadow-sm bg-[var(--vtl-panel)] border border-[var(--vtl-border)] text-[var(--vtl-text)] ${
                    msg.role === 'user' ? 'rounded-br-none' : 'rounded-bl-none'
                  }`}
                >
                  {msg.isAudio && <span className="text-xs opacity-75 block mb-1">🎤 Voice</span>}
                  {msg.role === 'bot' && msg.text.trim().length === 0 && isStreamingResponse && isLoading && isWaitingForFirstToken && idx === messages.length - 1 ? (
                    <div className="flex items-center gap-2 text-[var(--vtl-muted)]">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Thinking...</span>
                    </div>
                  ) : msg.role === 'bot' ? (
                    <MarkdownMessage text={msg.text} />
                  ) : (
                    <span className="whitespace-pre-wrap break-words">{msg.text}</span>
                  )}
                </div>
              </div>
            ))}

            {isLoading && !isStreamingResponse && (
              <div className="flex justify-start">
                <div className="bg-[var(--vtl-panel)] border border-[var(--vtl-border)] p-3 rounded-2xl rounded-bl-none flex items-center gap-2 text-[var(--vtl-muted)] text-sm">
                  <Loader2 className="w-4 h-4 animate-spin" /> Thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-3 bg-[var(--vtl-panel)] border-t border-[var(--vtl-border)]">
            <div className={`text-xs mb-2 ${isRecording ? 'text-red-500 font-medium' : 'text-[var(--vtl-muted)]'}`}>
              {voiceHintText}
            </div>

            <div className="flex items-center gap-2">
            <button
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onMouseLeave={stopRecording}
              onTouchStart={startRecording}
              onTouchEnd={stopRecording}
              title="Hold to record voice message, release to send"
              className={`p-2 sm:p-2.5 rounded-full flex-shrink-0 ${
                isRecording
                  ? 'bg-red-500 text-white animate-pulse'
                  : 'bg-[var(--vtl-chip-bg)] text-[var(--vtl-primary)] hover:bg-[var(--vtl-chip-hover)] disabled:opacity-50 disabled:cursor-not-allowed'
              }`}
              disabled={leadStage !== 'chat' || isLoading}
            >
              <Mic className="w-4 h-4 sm:w-5 sm:h-5" />
            </button>

            <form onSubmit={handleTextSubmit} className="flex-1 flex gap-2">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Message..."
                className="flex-1 min-w-0 px-3 sm:px-4 py-2 text-sm rounded-full bg-[var(--vtl-chip-bg)] text-[var(--vtl-text)] border border-transparent focus:bg-white focus:border-[var(--vtl-secondary)] outline-none"
                disabled={isRecording || isLoading}
              />
              <button
                type="submit"
                title="Send message"
                disabled={!inputText.trim() || isRecording || isLoading}
                className="p-2 sm:p-2.5 vtl-brand-gradient text-white rounded-full hover:brightness-95 disabled:opacity-50"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
