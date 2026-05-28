import { useEffect, useRef, useCallback } from 'react';

const WS_BASE = import.meta.env.VITE_WS_BASE || 'ws://127.0.0.1:8080';

export type WSMessage =
  | { type: 'metric'; task_id: number; data: Record<string, unknown> }
  | { type: 'log'; task_id: number; data: { level: string; message: string } }
  | { type: 'status'; task_id: number; status: string; progress: number };

type Handler = (msg: WSMessage) => void;

export function useTrainingWS(taskId: number, onMessage: Handler) {
  const wsRef = useRef<WebSocket | null>(null);
  const handlerRef = useRef<Handler>(onMessage);
  handlerRef.current = onMessage;

  const connect = useCallback(() => {
    const url = `${WS_BASE}/api/ws/training/${taskId}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'subscribe', task_id: taskId }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WSMessage;
        handlerRef.current(msg);
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      // auto-reconnect after 3s
      setTimeout(() => {
        if (document.visibilityState !== 'hidden') connect();
      }, 3000);
    };
  }, [taskId]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return wsRef;
}
