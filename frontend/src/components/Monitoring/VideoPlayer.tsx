"use client";

import React, { useState } from "react";
import Image from "next/image";

interface VideoPlayerProps {
    source: string;
}

export function VideoPlayer({ source }: VideoPlayerProps) {
    const streamUrl = `http://localhost:8000/video_feed/${source}`;
    
    // Fallback or loading state
    const [error, setError] = useState(false);

    return (
        <div className="relative aspect-video w-full overflow-hidden rounded-[10px] border border-stroke bg-black dark:border-dark-3">
             {/* 
                Since MJPEG is just a stream of images, we can use an <img> tag.
                However, Next.js <Image> requires dimensions or fills. 
                Using a standard <img> tag is often better for MJPEG streams to avoid optimization issues.
             */}
             {!error ? (
                <img 
                    src={streamUrl} 
                    alt={`Live Feed: ${source}`}
                    className="h-full w-full object-contain"
                    onError={() => setError(true)}
                />
             ) : (
                 <div className="flex h-full w-full items-center justify-center text-white">
                     <div className="text-center">
                         <p className="text-xl font-bold text-red-500">Signal Lost</p>
                         <p className="text-sm text-gray-400">Connecting to {source} stream...</p>
                         {/* Retry Button */}
                         <button 
                            onClick={() => setError(false)}
                            className="mt-4 rounded bg-primary px-4 py-2 text-white"
                         >
                             Retry Connection
                         </button>
                     </div>
                 </div>
             )}
             
             {/* HUD Overlay */}
             <div className="absolute left-4 top-4 flex items-center gap-2 rounded bg-black/50 px-2 py-1 text-xs font-bold text-white backdrop-blur">
                 <span className="h-2 w-2 animate-pulse rounded-full bg-red-500"></span>
                 LIVE | {source.toUpperCase()}
             </div>
             
             {/* Scan line effect (CSS only visual) */}
             <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-[1] bg-[length:100%_2px,3px_100%] opacity-20"></div>
        </div>
    );
}
