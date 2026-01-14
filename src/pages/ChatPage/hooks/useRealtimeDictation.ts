import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

declare global {
  interface Window {
    webkitSpeechRecognition?: typeof SpeechRecognition;
  }
}

export type DictationStatus =
  | 'idle'
  | 'requesting-permission'
  | 'fetching-session'
  | 'connecting'
  | 'listening'
  | 'stopping'
  | 'error';

interface UseRealtimeDictationOptions {
  onTranscriptChange?: (transcript: string, options: { isFinal: boolean }) => void;
  model?: string;
}

interface RealtimeSessionResponse {
  client_secret?: {
    value: string;
    expires_at?: number;
  };
  id?: string;
  ice_servers?: RTCIceServer[];
  expires_at?: number;
}

interface UseRealtimeDictationResult {
  isActive: boolean;
  status: DictationStatus;
  error: string | null;
  transcript: string;
  startDictation: () => Promise<void>;
  stopDictation: (options?: { silent?: boolean }) => void;
}

const DEFAULT_MODEL = 'gpt-realtime-mini';

export const useRealtimeDictation = ({
  onTranscriptChange,
  model = DEFAULT_MODEL,
}: UseRealtimeDictationOptions = {}): UseRealtimeDictationResult => {
  const [isActive, setIsActive] = useState(false);
  const [status, setStatus] = useState<DictationStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [transcript, setTranscript] = useState('');

  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const dataChannelRef = useRef<RTCDataChannel | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const transcriptRef = useRef('');
  const isStoppingRef = useRef(false);
  const connectionTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const speechRecognitionRef = useRef<SpeechRecognition | null>(null);
  const isFallbackActiveRef = useRef(false);
  const shouldRestartFallbackRef = useRef(false);

  const instructions = useMemo(
    () =>
      'You are assisting with live dictation. Transcribe the incoming microphone audio into plain text. ' +
      'Return partial results quickly, refine them as confidence improves, and include punctuation where helpful. ' +
      'Do not add speaker labels or commentary.',
    [],
  );

  const speechRecognitionConstructor = useMemo(() => {
    if (typeof window === 'undefined') {
      return null;
    }
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }, []);

  const clearConnectionTimeout = useCallback(() => {
    if (connectionTimeoutRef.current !== null) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = null;
    }
  }, []);

  const teardown = useCallback(() => {
    isStoppingRef.current = true;
    clearConnectionTimeout();
    dataChannelRef.current?.close();
    dataChannelRef.current = null;
    peerConnectionRef.current?.close();
    peerConnectionRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
    transcriptRef.current = '';
    setTranscript('');
    isStoppingRef.current = false;
  }, [clearConnectionTimeout]);

  const notifyTranscript = useCallback(
    (value: string, isFinal: boolean) => {
      onTranscriptChange?.(value, { isFinal });
    },
    [onTranscriptChange],
  );

  const sendCreateResponseEvent = useCallback(() => {
    const channel = dataChannelRef.current;
    if (!channel || channel.readyState !== 'open') {
      return;
    }

    const payload = {
      type: 'response.create',
      response: {
        conversation: 'none',
        modalities: ['text'],
        instructions,
        metadata: {
          intent: 'dictation',
        },
      },
    };

    channel.send(JSON.stringify(payload));
  }, [instructions]);

  const stopDictation = useCallback(
    ({ silent = false }: { silent?: boolean } = {}) => {
      if (!isActive && !silent) {
        return;
      }
      if (isFallbackActiveRef.current) {
        setStatus('stopping');
        shouldRestartFallbackRef.current = false;
        speechRecognitionRef.current?.stop();
        speechRecognitionRef.current = null;
        isFallbackActiveRef.current = false;
        setIsActive(false);
        if (!silent) {
          setStatus('idle');
        }
        return;
      }
      setStatus('stopping');
      clearConnectionTimeout();
      teardown();
      setIsActive(false);
      if (!silent) {
        setStatus('idle');
      }
    },
    [clearConnectionTimeout, isActive, teardown],
  );

  const startSpeechRecognitionFallback = useCallback(
    (fallbackReason?: string) => {
      if (!speechRecognitionConstructor) {
        return false;
      }

      if (isFallbackActiveRef.current && speechRecognitionRef.current) {
        return true;
      }

      try {
        shouldRestartFallbackRef.current = true;
        const SpeechRecognitionCtor = speechRecognitionConstructor as new () => SpeechRecognition;
        const recognition = new SpeechRecognitionCtor();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = navigator.language || 'en-US';

        recognition.onstart = () => {
          if (fallbackReason) {
            console.warn(
              'Realtime dictation falling back to browser speech recognition:',
              fallbackReason,
            );
          }
          setError(null);
          setStatus('listening');
          setIsActive(true);
        };

        recognition.onresult = (event: SpeechRecognitionEvent) => {
          let interimTranscript = '';
          let finalTranscript = '';
          for (let i = event.resultIndex; i < event.results.length; i += 1) {
            const result = event.results[i];
            if (result.isFinal) {
              finalTranscript += result[0]?.transcript ?? '';
            } else {
              interimTranscript += result[0]?.transcript ?? '';
            }
          }
          const combined = `${finalTranscript}${interimTranscript}`.trim();
          transcriptRef.current = combined;
          setTranscript(combined);
          notifyTranscript(combined, finalTranscript.length > 0 && interimTranscript.length === 0);
        };

        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
          if (event.error === 'aborted') {
            // Expected when we manually stop, ignore.
            return;
          }
          const isNoSpeech = event.error === 'no-speech';
          if (!isNoSpeech) {
            shouldRestartFallbackRef.current = false;
          }
          speechRecognitionRef.current = null;
          isFallbackActiveRef.current = false;
          setIsActive(false);
          if (isNoSpeech) {
            // Let onend trigger a restart without surfacing an error.
            return;
          }
          const fallbackError =
            event.error === 'not-allowed'
              ? 'Browser speech recognition permission denied.'
              : `Speech recognition error — ${event.error}`;
          setError(fallbackError);
          setStatus('error');
        };

        recognition.onend = () => {
          const shouldRestart = shouldRestartFallbackRef.current;
          if (shouldRestart) {
            try {
              recognition.start();
              return;
            } catch (restartError) {
              console.error('Failed to restart speech recognition fallback', restartError);
            }
          }
          speechRecognitionRef.current = null;
          isFallbackActiveRef.current = false;
          shouldRestartFallbackRef.current = false;
          setIsActive(false);
          setTranscript((current) => current.trim());
          setStatus('idle');
        };

        recognition.start();
        speechRecognitionRef.current = recognition;
        isFallbackActiveRef.current = true;
        return true;
      } catch (err) {
        console.error('Failed to start speech recognition fallback', err);
        return false;
      }
    },
    [notifyTranscript, speechRecognitionConstructor],
  );

  const handleDictationFailure = useCallback(
    (message: string, { allowFallback = false }: { allowFallback?: boolean } = {}) => {
      clearConnectionTimeout();
      teardown();
      if (allowFallback && startSpeechRecognitionFallback(message)) {
        return;
      }
      setError(message);
      setStatus('error');
      setIsActive(false);
    },
    [clearConnectionTimeout, startSpeechRecognitionFallback, teardown],
  );

  const handleRealtimeMessage = useCallback(
    (event: MessageEvent<string>) => {
      try {
        const message = JSON.parse(event.data);
        switch (message.type) {
          case 'response.output_text.delta': {
            transcriptRef.current = `${transcriptRef.current}${message.delta}`;
            const nextTranscript = transcriptRef.current;
            setTranscript(nextTranscript);
            notifyTranscript(nextTranscript, false);
            break;
          }
          case 'response.output_text.done': {
            const nextTranscript = transcriptRef.current.trim();
            setTranscript(nextTranscript);
            notifyTranscript(nextTranscript, true);
            break;
          }
          case 'response.completed': {
            if (isStoppingRef.current) {
              return;
            }
            // Request another transcription pass to keep the stream alive
            sendCreateResponseEvent();
            break;
          }
          case 'error':
          case 'response.error': {
            if (message.error?.message) {
              handleDictationFailure(message.error.message);
            } else if (message.message) {
              handleDictationFailure(message.message);
            } else {
              handleDictationFailure('Realtime session reported an error.');
            }
            break;
          }
          default:
            break;
        }
      } catch (err) {
        console.error('Failed to parse realtime message', err);
      }
    },
    [handleDictationFailure, notifyTranscript, sendCreateResponseEvent],
  );

  const startDictation = useCallback(async () => {
    if (isActive) {
      return;
    }

    try {
      setError(null);
      setStatus('requesting-permission');

      const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = mediaStream;

      setStatus('fetching-session');
      const sessionResponse = await fetch('/api/v2/realtime/session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model }),
      });

      if (!sessionResponse.ok) {
        const errorText = await sessionResponse.text();
        throw new Error(errorText || 'Failed to fetch realtime session.');
      }

      const sessionData = (await sessionResponse.json()) as RealtimeSessionResponse;
      const clientSecret = sessionData?.client_secret?.value;

      if (!clientSecret) {
        throw new Error('Realtime session missing client secret.');
      }

      const sessionIceServers = sessionData?.ice_servers || [];
      const fallbackIceServers: RTCIceServer[] = [
        {
          urls: [
            'stun:openrelay.metered.ca:80',
            'stun:openrelay.metered.ca:3478',
            'turn:openrelay.metered.ca:80?transport=udp',
            'turn:openrelay.metered.ca:80?transport=tcp',
            'turn:openrelay.metered.ca:443?transport=udp',
            'turn:openrelay.metered.ca:443?transport=tcp',
            'turns:openrelay.metered.ca:443?transport=tcp',
          ],
          username: 'openrelayproject',
          credential: 'openrelayproject',
        },
      ];
      const iceServers =
        sessionIceServers.length > 0
          ? sessionIceServers
          : [...fallbackIceServers, { urls: ['stun:stun.l.google.com:19302'] }];

      setStatus('connecting');
      clearConnectionTimeout();
      connectionTimeoutRef.current = setTimeout(() => {
        handleDictationFailure(
          'Timed out while connecting to the dictation service. Check your connection and try again.',
        );
      }, 15000);
      const peerConnection = new RTCPeerConnection({ iceServers });
      peerConnectionRef.current = peerConnection;

      peerConnection.onconnectionstatechange = () => {
        if (peerConnection.connectionState === 'connected') {
          setStatus('listening');
          clearConnectionTimeout();
        } else if (peerConnection.connectionState === 'failed') {
          handleDictationFailure('Realtime connection failed.', { allowFallback: true });
        } else if (peerConnection.connectionState === 'disconnected') {
          handleDictationFailure('Realtime connection was interrupted.', { allowFallback: true });
        }
      };

      peerConnection.oniceconnectionstatechange = () => {
        const state = peerConnection.iceConnectionState;
        if (state === 'connected' || state === 'completed') {
          clearConnectionTimeout();
        } else if (state === 'failed') {
          handleDictationFailure('Realtime ICE negotiation failed.', { allowFallback: true });
        }
      };

      peerConnection.onicecandidateerror = (event) => {
        const errorMessage =
          typeof event.errorText === 'string' && event.errorText.length > 0
            ? event.errorText
            : 'The network rejected a candidate during realtime negotiation.';
        // ICE candidate errors are expected during negotiation and not necessarily fatal.
        // The connection may still succeed via other candidates.
        // Actual connection failures are handled by oniceconnectionstatechange and onconnectionstatechange.
        console.warn(`Realtime ICE candidate error (non-fatal): ${errorMessage}`);
      };

      mediaStream.getTracks().forEach((track) => {
        peerConnection.addTrack(track, mediaStream);
      });

      const dataChannel = peerConnection.createDataChannel('oai-events');
      dataChannelRef.current = dataChannel;
      dataChannel.onmessage = handleRealtimeMessage;
      dataChannel.onopen = () => {
        transcriptRef.current = '';
        setTranscript('');
        sendCreateResponseEvent();
      };

      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);

      peerConnection.onicecandidate = async (event) => {
        if (event.candidate) {
          return;
        }

        const localDescription = peerConnection.localDescription;
        if (!localDescription) {
          return;
        }

        try {
          const answerResponse = await fetch('/api/v2/realtime/answer', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              model,
              clientSecret,
              sdp: localDescription.sdp,
            }),
          });

          if (!answerResponse.ok) {
            const answerText = await answerResponse.text();
            throw new Error(answerText || 'Failed to establish realtime connection.');
          }

          const answer = await answerResponse.text();
          if (!peerConnectionRef.current) {
            return;
          }
          await peerConnection.setRemoteDescription({ type: 'answer', sdp: answer });
          setIsActive(true);
        } catch (error) {
          const message =
            error instanceof Error ? error.message : 'Failed to establish realtime connection.';
          handleDictationFailure(message, { allowFallback: true });
        }
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start dictation.';
      console.error('Failed to start realtime dictation', err);
      handleDictationFailure(message, { allowFallback: true });
    }
  }, [
    clearConnectionTimeout,
    handleDictationFailure,
    handleRealtimeMessage,
    isActive,
    model,
    sendCreateResponseEvent,
  ]);

  useEffect(() => {
    return () => {
      stopDictation({ silent: true });
    };
  }, [stopDictation]);

  return {
    isActive,
    status,
    error,
    transcript,
    startDictation,
    stopDictation,
  };
};

export default useRealtimeDictation;
