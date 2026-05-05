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

type LeadStep = 'email' | 'name' | 'chat';

type BrowserSpeechRecognition = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: {
    resultIndex: number;
    results: ArrayLike<ArrayLike<{ transcript: string }>>;
  }) => void) | null;
  onerror: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type BrowserSpeechRecognitionCtor = new () => BrowserSpeechRecognition;


const resolveApiBaseUrl = (): string => {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configured) {
    return configured.replace(/\/+$/, '');
  }

  if (typeof window !== 'undefined') {
    return `${window.location.origin}/backend`;
  }

  return 'http://localhost:8000';
};

const API_BASE_URL = resolveApiBaseUrl();
const SESSION_STORAGE_KEY = 'chatbot_session_id';

const ChatBotIcon = ({ className }: { className?: string }) => (
  <img src="/digicoco.png" alt="DIGICoCo ChatBot" className={className} />
);

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

const isValidEmail = (email: string): boolean => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());

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
  const [isVoiceRequestInFlight, setIsVoiceRequestInFlight] = useState(false);
  const [floatingImageError, setFloatingImageError] = useState(false);

  const [sessionId, setSessionId] = useState('');
  const [leadStep, setLeadStep] = useState<LeadStep>('email');
  const [userEmail, setUserEmail] = useState('');
  const [userName, setUserName] = useState('');

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingStartedAtRef = useRef<number>(0);
  const isStoppingRecordingRef = useRef(false);
  const pendingVoiceTurnRef = useRef<{ userIndex: number; botIndex: number } | null>(null);
  const speechRecognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const liveVoiceTranscriptRef = useRef('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

  const stopAudio = () => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      currentAudioRef.current = null;
    }
  };

  useEffect(() => {
    const storedSessionId = localStorage.getItem(SESSION_STORAGE_KEY)?.trim();
    if (storedSessionId) {
      // try to restore session from backend
      (async () => {
        try {
          const res = await fetch(`${API_BASE_URL}/api/session/${encodeURIComponent(storedSessionId)}`);
          if (res.ok) {
            const data = await res.json();
            setSessionId(storedSessionId);
            setUserEmail(data.email || '');
            setUserName(data.name || '');
            setLeadStep('chat');
            setMessages([{ role: 'bot', text: `Welcome back, ${data.name || 'friend'}! How can I help you today?` }]);
            return;
          }
        } catch (e) {
          // fall back to fresh lead flow
          // console.warn('Session restore failed', e);
        }
        // no valid session
        setSessionId(createSessionId());
        setLeadStep('email');
        setUserEmail('');
        setUserName('');
        setMessages([{ role: 'bot', text: 'Hi! Before we begin, please share your email address.' }]);
      })();
      return;
    }

    setSessionId(createSessionId());
    setLeadStep('email');
    setUserEmail('');
    setUserName('');
    setMessages([{ role: 'bot', text: 'Hi! Before we begin, please share your email address.' }]);
  }, []);


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
    stopAudio();
    if (!rawMessage.trim()) {
      return;
    }

    if (leadStep !== 'chat') {
      const leadInput = rawMessage.trim();

      if (leadStep === 'email') {
        if (!isValidEmail(leadInput)) {
          setMessages((prev) => [...prev, { role: 'bot', text: 'Please enter a valid email address.' }]);
          return;
        }

        setUserEmail(leadInput);
        setMessages((prev) => [
          ...prev,
          { role: 'user', text: leadInput },
          { role: 'bot', text: 'Thanks! Now please share your name.' },
        ]);
        setLeadStep('name');
        return;
      }

      if (leadStep === 'name') {
        setUserName(leadInput);
        // register session on backend (creates session row and user)
        try {
          const res = await fetch(`${API_BASE_URL}/api/session/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: userEmail, name: leadInput, session_id: sessionId || undefined }),
          });
          if (res.ok) {
            const data = await res.json();
            const sid = data.session_id || sessionId || `s_${crypto?.randomUUID?.() || Math.random().toString(36).slice(2)}`;
            localStorage.setItem(SESSION_STORAGE_KEY, sid);
            setSessionId(sid);
          }
        } catch (e) {
          // ignore — session may still work in-memory
          console.error('Session register failed', e);
        }

        setMessages((prev) => [
          ...prev,
          { role: 'user', text: leadInput },
          { role: 'bot', text: `Nice to meet you, ${leadInput}. How can I help you today?` },
        ]);
        setLeadStep('chat');
        return;
      }
    }

    const userMsg = rawMessage.trim();
    setMessages((prev) => [...prev, { role: 'user', text: userMsg }]);


    setIsLoading(true);
    setIsStreamingResponse(true);
    setIsWaitingForFirstToken(true);
    setMessages((prev) => [...prev, { role: 'bot', text: '' }]);

    try {
      const formData = new FormData();
      formData.append('query', userMsg);
      formData.append('session_id', sessionId);

      const headers: Record<string, string> = {};
      if (userEmail) headers['X-User-Email'] = userEmail;
      if (userName) headers['X-User-Name'] = userName;

      const response = await fetch(`${API_BASE_URL}/api/chat/text/stream`, {
        method: 'POST',
        body: formData,
        headers,
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
    stopAudio();
    if (isLoading || isRecording || leadStep !== 'chat') {
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');
      const mediaRecorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      recordingStartedAtRef.current = Date.now();
      isStoppingRecordingRef.current = false;

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = handleAudioStop;
      mediaRecorder.start();

      const speechCtor = (
        window as Window & {
          SpeechRecognition?: BrowserSpeechRecognitionCtor;
          webkitSpeechRecognition?: BrowserSpeechRecognitionCtor;
        }
      ).SpeechRecognition || (
        window as Window & {
          SpeechRecognition?: BrowserSpeechRecognitionCtor;
          webkitSpeechRecognition?: BrowserSpeechRecognitionCtor;
        }
      ).webkitSpeechRecognition;

      if (speechCtor) {
        const recognizer = new speechCtor();
        recognizer.continuous = true;
        recognizer.interimResults = true;
        recognizer.lang = 'en-US';
        recognizer.onresult = (event) => {
          let transcript = '';
          for (let i = event.resultIndex; i < event.results.length; i += 1) {
            transcript += event.results[i][0]?.transcript || '';
          }
          liveVoiceTranscriptRef.current = transcript.trim() || liveVoiceTranscriptRef.current;
        };
        recognizer.onerror = () => {
          speechRecognitionRef.current = null;
        };

        try {
          recognizer.start();
          speechRecognitionRef.current = recognizer;
        } catch {
          speechRecognitionRef.current = null;
        }
      }

      setIsRecording(true);
    } catch (error) {
      console.error('Microphone access error:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          text: 'Microphone access is blocked. If you are viewing this on a website, the website needs to allow microphone permissions in its iframe settings.'
        }
      ]);
    }
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current || !isRecording || isStoppingRecordingRef.current) {
      return;
    }

    if (mediaRecorderRef.current.state !== 'inactive') {
      isStoppingRecordingRef.current = true;
      mediaRecorderRef.current.stop();
      speechRecognitionRef.current?.stop();
      speechRecognitionRef.current = null;
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
    }
  };

  const updatePendingVoiceTurn = (userText: string, botText: string) => {
    setMessages((prev) => {
      const turn = pendingVoiceTurnRef.current;
      if (!turn) {
        return [...prev, { role: 'user', text: userText, isAudio: true }, { role: 'bot', text: botText, isAudio: true }];
      }

      if (
        turn.userIndex >= 0 &&
        turn.botIndex >= 0 &&
        prev[turn.userIndex]?.role === 'user' &&
        prev[turn.userIndex]?.isAudio &&
        prev[turn.botIndex]?.role === 'bot' &&
        prev[turn.botIndex]?.isAudio
      ) {
        const next = [...prev];
        next[turn.userIndex] = { ...next[turn.userIndex], text: userText };
        next[turn.botIndex] = { ...next[turn.botIndex], text: botText };
        return next;
      }

      return [...prev, { role: 'user', text: userText, isAudio: true }, { role: 'bot', text: botText, isAudio: true }];
    });
  };

  const handleAudioStop = async () => {
    const recordingDurationMs = Date.now() - recordingStartedAtRef.current;
    const chunkType = audioChunksRef.current[0]?.type || 'audio/webm';
    const audioBlob = new Blob(audioChunksRef.current, { type: chunkType });
    if (audioBlob.size === 0 || recordingDurationMs < 300) {
      isStoppingRecordingRef.current = false;
      setMessages((prev) => [...prev, { role: 'bot', text: 'Recording is too short. Please hold the mic and try again.' }]);
      return;
    }

    setIsLoading(true);
    setIsVoiceRequestInFlight(true);

    const audioFile = new File([audioBlob], 'recording.webm', { type: chunkType });
    const formData = new FormData();
    formData.append('audio', audioFile);

    let finalUserText = liveVoiceTranscriptRef.current.trim();

    if (!finalUserText) {
      try {
        const transcribeRes = await fetch(`${API_BASE_URL}/api/chat/transcribe`, {
          method: 'POST',
          body: formData,
        });
        if (transcribeRes.ok) {
          const transcribeData = await transcribeRes.json();
          finalUserText = transcribeData.text;
        }
      } catch (e) {
        console.error("Transcription failed", e);
      }
    }

    finalUserText = finalUserText || "Voice Message";

    setMessages((prev) => {
      const userIndex = prev.length;
      const botIndex = prev.length + 1;
      pendingVoiceTurnRef.current = { userIndex, botIndex };
      return [
        ...prev,
        { role: 'user', text: finalUserText, isAudio: true },
        { role: 'bot', text: '', isAudio: true },
      ];
    });

    try {
      const voiceFormData = new FormData();
      if (finalUserText && finalUserText !== "Voice Message") {
        voiceFormData.append('query', finalUserText);
      } else {
        voiceFormData.append('audio', audioFile);
      }

      const response = await fetch(`${API_BASE_URL}/api/chat/voice`, {
        method: 'POST',
        headers: {
          'X-Session-Id': sessionId,
          ...(userEmail ? { 'X-User-Email': userEmail } : {}),
          ...(userName ? { 'X-User-Name': userName } : {}),
        },
        body: voiceFormData,
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

      if (!pendingVoiceTurnRef.current) {
        setMessages((prev) => {
          const userIndex = prev.length;
          const botIndex = prev.length + 1;
          pendingVoiceTurnRef.current = { userIndex, botIndex };
          return [
            ...prev,
            { role: 'user', text: userQuery, isAudio: true },
            { role: 'bot', text: '', isAudio: true },
          ];
        });
      }

      try {
        const lastTurnResponse = await axios.get(`${API_BASE_URL}/api/chat/last`, {
          params: { session_id: sessionId },
        });
        userQuery = lastTurnResponse.data.user_query || userQuery;
        botReply = lastTurnResponse.data.reply || botReply;
      } catch {
        // Keep header-based fallbacks when last-turn lookup is unavailable.
      }

      updatePendingVoiceTurn(userQuery || 'Voice Message', botReply || 'Audio Reply');

      const audioResponseBlob = await response.blob();
      if (audioResponseBlob.size > 0) {
        const audioUrl = URL.createObjectURL(audioResponseBlob);
        const audio = new Audio(audioUrl);
        currentAudioRef.current = audio;
        try {
          await audio.play();
        } catch {
          // Keep text response visible even when autoplay is blocked.
        }
        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          if (currentAudioRef.current === audio) {
            currentAudioRef.current = null;
          }
        };
      }
    } catch {
      updatePendingVoiceTurn('Voice Message', 'Sorry, failed to process audio.');
    } finally {
      audioChunksRef.current = [];
      isStoppingRecordingRef.current = false;
      pendingVoiceTurnRef.current = null;
      liveVoiceTranscriptRef.current = '';
      setIsVoiceRequestInFlight(false);
      setIsLoading(false);
    }
  };

  const voiceHintText = isRecording
    ? '🔴 Recording... tap again to send'
    : 'Tap mic to record • Tap again to send';

  const handleVoiceButtonClick = () => {
    if (isRecording) {
      stopRecording();
      return;
    }
    void startRecording();
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`fixed z-50 transition-transform hover:scale-105 flex items-center justify-center ${isOpen ? 'top-3 right-3 sm:top-auto sm:bottom-6 sm:right-6' : 'bottom-4 right-4 sm:bottom-6 sm:right-6'
          } ${!isOpen
            ? 'w-16 h-16 sm:w-[110px] sm:h-[110px] bg-transparent shadow-none p-0'
            : 'p-3 sm:p-4 bg-[var(--vtl-primary)] text-white rounded-full shadow-2xl hover:brightness-95'
          }`}
      >
        {isOpen ? (
          <X className="w-5 h-5 sm:w-6 sm:h-6" />
        ) : (
          <ChatBotIcon className="w-full h-full object-contain drop-shadow-lg" />
        )}
      </button>

      {isOpen && (
        <div className="fixed inset-x-0 top-0 bottom-0 sm:inset-auto sm:bottom-24 sm:right-6 z-40 w-full sm:w-[min(540px,94vw)] h-full sm:h-[760px] sm:max-h-[85vh] bg-[var(--vtl-panel)] rounded-none sm:rounded-2xl shadow-2xl flex flex-col border border-[var(--vtl-border)] overflow-hidden">
          <div className="bg-[var(--vtl-primary)] p-3 sm:p-4 text-white font-bold text-base sm:text-lg flex justify-between items-center shadow-md z-10">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-[var(--vtl-accent)] rounded-full animate-pulse"></div>
              <span>DIGIC Assistant</span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-4 bg-[var(--vtl-surface)]">
            {messages.map((msg, idx) => (
              <div key={`${msg.role}-${idx}`} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`p-3 rounded-2xl text-sm max-w-[90%] sm:max-w-[85%] shadow-sm bg-[var(--vtl-panel)] border border-[var(--vtl-border)] text-[var(--vtl-text)] ${msg.role === 'user' ? 'rounded-br-none' : 'rounded-bl-none'
                    }`}
                >
                  {msg.isAudio && <span className="text-xs opacity-75 block mb-1">🎤 Voice</span>}
                  {msg.role === 'bot' && msg.text.trim().length === 0 && isLoading && idx === messages.length - 1 ? (
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

            {isLoading && !isStreamingResponse && !isVoiceRequestInFlight && !(messages.length > 0 && messages[messages.length - 1]?.role === 'bot' && messages[messages.length - 1]?.text.trim().length === 0) && (
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
                onClick={handleVoiceButtonClick}
                title={isRecording ? 'Tap to stop and send voice message' : 'Tap to start recording voice message'}
                className={`p-2 sm:p-2.5 rounded-full flex-shrink-0 ${isRecording
                  ? 'bg-red-500 text-white animate-pulse'
                  : 'bg-[var(--vtl-chip-bg)] text-[var(--vtl-primary)] hover:bg-[var(--vtl-chip-hover)] disabled:opacity-50 disabled:cursor-not-allowed'
                  }`}
                disabled={isLoading || leadStep !== 'chat'}
              >
                <Mic className="w-4 h-4 sm:w-5 sm:h-5" />
              </button>

              <form onSubmit={handleTextSubmit} className="flex-1 flex gap-2">
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder={leadStep === 'email' ? 'Email address' : leadStep === 'name' ? 'Your name' : 'Message...'}
                  className="flex-1 min-w-0 px-3 sm:px-4 py-2 text-sm rounded-full bg-[var(--vtl-chip-bg)] text-[var(--vtl-text)] border border-transparent focus:bg-white focus:border-[var(--vtl-secondary)] outline-none"
                  disabled={isRecording || isLoading}
                />
                <button
                  type="submit"
                  title={leadStep === 'chat' ? 'Send message' : 'Continue'}
                  disabled={!inputText.trim() || isRecording || isLoading}
                  className="p-2 sm:p-2.5 bg-[var(--vtl-primary)] text-white rounded-full hover:brightness-95 disabled:opacity-50"
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
