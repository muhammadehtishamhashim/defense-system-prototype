"use client";

import { useEffect, useState } from "react";
import { Play, Square } from "lucide-react";

export function MonitorControls({ source }: { source: string }) {
  const [isActive, setIsActive] = useState(false);
  const [loading, setLoading] = useState(true);

  // Poll status on mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch("http://localhost:8000/status");
        const data = await res.json();
        // data is { "threat": true, ... }
        if (data && typeof data[source] !== 'undefined') {
            setIsActive(data[source]);
        }
      } catch (e) {
        console.error("Failed to fetch status");
      } finally {
        setLoading(false);
      }
    };
    checkStatus();
  }, [source]);

  const toggleStatus = async (action: "start" | "stop") => {
      setLoading(true);
      try {
          const res = await fetch(`http://localhost:8000/control/${source}/${action}`, {
              method: 'POST'
          });
          const data = await res.json();
          if (data.active !== undefined) {
              setIsActive(data.active);
          }
      } catch (e) {
          console.error("Failed to toggle status");
      } finally {
          setLoading(false);
      }
  };

  return (
    <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
             <span className="flex h-3 w-3 relative">
                  {isActive && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>}
                  <span className={`relative inline-flex rounded-full h-3 w-3 ${isActive ? 'bg-green-500' : 'bg-red-500'}`}></span>
            </span>
            <span className={`text-sm font-medium ${isActive ? 'text-green-500' : 'text-red-500'}`}>
                {isActive ? "System Online" : "System Offline"}
            </span>
        </div>

        <div className="flex items-center gap-2 rounded-lg border border-stroke bg-white p-1 dark:border-dark-3 dark:bg-dark-2">
            <button
                disabled={loading || isActive}
                onClick={() => toggleStatus('start')}
                className={`flex items-center gap-2 rounded px-3 py-1.5 text-xs font-medium transition-colors
                    ${isActive 
                        ? 'bg-gray-100 text-gray-400 dark:bg-dark-3 dark:text-dark-6 opacity-50 cursor-not-allowed' 
                        : 'bg-green-500 text-white hover:bg-green-600'
                    }`}
            >
                <Play className="size-3" fill="currentColor" />
                Start
            </button>
            <button
                disabled={loading || !isActive}
                onClick={() => toggleStatus('stop')}
                className={`flex items-center gap-2 rounded px-3 py-1.5 text-xs font-medium transition-colors
                    ${!isActive 
                        ? 'bg-gray-100 text-gray-400 dark:bg-dark-3 dark:text-dark-6 opacity-50 cursor-not-allowed' 
                        : 'bg-red-500 text-white hover:bg-red-600'
                    }`}
            >
                <Square className="size-3" fill="currentColor" />
                Stop
            </button>
        </div>
    </div>
  );
}
