"use client";

import { useAlerts } from "@/context/AlertContext";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { useState } from "react";

export default function VerifyPage() {
  const { alerts } = useAlerts();
  // Filter alerts that have snapshots (or simulate snapshots for demo if needed)
  // For now, we only show alerts that actually have a snapshot URL provided by backend
  const snapshotAlerts = alerts.filter(a => a.snapshot) || [];
  
  // Local state for handled alerts to hide them without clearing global state if desired
  const [handled, setHandled] = useState<Set<number>>(new Set());

  const handleAction = (timestamp: number, action: 'verify' | 'dismiss') => {
      console.log(`Action: ${action} on alert ${timestamp}`);
      setHandled(prev => new Set(prev).add(timestamp));
      // In a real app, send API request to backend to log verify/dismiss
  };

  return (
    <div className="flex flex-col gap-6">
       <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-dark dark:text-white">
                Signal Verification Center
            </h2>
            <div className="text-sm font-medium text-gray-500">
                {snapshotAlerts.length - handled.size} Pending Review
            </div>
        </div>

        {snapshotAlerts.length === 0 ? (
            <div className="flex h-64 flex-col items-center justify-center rounded-[10px] border border-dashed border-stroke dark:border-dark-3">
                <p className="text-lg font-medium text-dark dark:text-white">No snapshots available for verification</p>
                <p className="text-sm text-gray-400">Snapshots appear here when threats are detected.</p>
            </div>
        ) : (
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3">
                {snapshotAlerts
                    .filter(alert => !handled.has(alert.timestamp))
                    .map((alert, idx) => (
                    <div key={idx} className="rounded-[10px] border border-stroke bg-white p-4 shadow-1 dark:border-dark-3 dark:bg-gray-dark dark:shadow-card">
                        <div className="relative mb-4 aspect-video w-full overflow-hidden rounded-lg bg-black">
                            {/* Use standard img for local dev if Image fails with localhost domains not config */}
                            <img 
                                src={`http://localhost:8000${alert.snapshot}`} 
                                alt="Snapshot" 
                                className="h-full w-full object-contain"
                            />
                            <div className="absolute top-2 right-2 rounded bg-black/60 px-2 py-1 text-xs font-bold text-white backdrop-blur">
                                {((alert.count || 0) * 100).toFixed(0)}% Confidence
                            </div>
                        </div>

                        <div className="mb-4">
                            <div className="flex items-center justify-between">
                                <h4 className="text-lg font-bold capitalize text-dark dark:text-white">{alert.source} Detected</h4>
                                <span className="text-xs text-gray-500">{new Date(alert.timestamp * 1000).toLocaleString()}</span>
                            </div>
                            <p className="text-sm text-gray-500">
                                Source: Camera {alert.source.toUpperCase()}-01
                            </p>
                        </div>

                        <div className="flex gap-3">
                            <button 
                                onClick={() => handleAction(alert.timestamp, 'verify')}
                                className="flex-1 rounded bg-green-500 py-2 text-sm font-medium text-white hover:bg-green-600"
                            >
                                Verify (+1)
                            </button>
                            <button 
                                onClick={() => handleAction(alert.timestamp, 'dismiss')}
                                className="flex-1 rounded border border-stroke py-2 text-sm font-medium text-dark hover:bg-gray-50 dark:border-dark-4 dark:text-white dark:hover:bg-dark-3"
                            >
                                Dismiss
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        )}
    </div>
  );
}
