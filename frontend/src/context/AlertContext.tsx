"use client";

import React, { createContext, useContext, useEffect, useState, ReactNode } from "react";

type Alert = {
  source: string;
  count: number;
  timestamp: number;
  type: string;
  snapshot?: string;
};

interface AlertContextType {
  alerts: Alert[];
  latestAlert: Alert | null;
  isConnected: boolean;
  clearAlerts: () => void;
}

const AlertContext = createContext<AlertContextType | undefined>(undefined);

export const AlertProvider = ({ children }: { children: ReactNode }) => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [latestAlert, setLatestAlert] = useState<Alert | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectInterval: NodeJS.Timeout;

    const connect = () => {
      ws = new WebSocket("ws://localhost:8000/ws");

      ws.onopen = () => {
        setIsConnected(true);
        console.log("Connected to Defense System Brain");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "alert") {
            const newAlert = {
              source: data.source,
              count: data.count,
              timestamp: data.timestamp,
              type: data.type,
            };
            setAlerts((prev) => [newAlert, ...prev].slice(0, 50)); // Keep last 50
            setLatestAlert(newAlert);
          }
        } catch (e) {
          console.error("Error parsing websocket message", e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log("Disconnected from Brain, retrying...");
        // Reconnect logic
        reconnectInterval = setTimeout(connect, 3000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error", err);
        ws.close();
      };
    };

    connect();

    return () => {
      if (ws) ws.close();
      if (reconnectInterval) clearTimeout(reconnectInterval);
    };
  }, []);

  const clearAlerts = () => setAlerts([]);

  return (
    <AlertContext.Provider value={{ alerts, latestAlert, isConnected, clearAlerts }}>
      {children}
    </AlertContext.Provider>
  );
};

export const useAlerts = () => {
  const context = useContext(AlertContext);
  if (context === undefined) {
    throw new Error("useAlerts must be used within an AlertProvider");
  }
  return context;
};
