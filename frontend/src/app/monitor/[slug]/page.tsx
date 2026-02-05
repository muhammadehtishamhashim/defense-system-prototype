import { VideoPlayer } from "@/components/Monitoring/VideoPlayer";
import { MonitorControls } from "@/components/Monitoring/MonitorControls";

export default async function MonitorPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  
  // Valid sources
  const validSources = ['threat', 'theft', 'border'];
  const isValid = validSources.includes(slug);

  return (
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-dark dark:text-white capitalize">
                {slug} Monitoring
            </h2>
            <MonitorControls source={slug} />
        </div>

        {isValid ? (
             <VideoPlayer source={slug} />
        ) : (
            <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-10 text-center text-red-500">
                Invalid Monitoring Source
            </div>
        )}

        <div className="rounded-[10px] border border-stroke bg-white p-6 shadow-1 dark:border-dark-3 dark:bg-gray-dark dark:shadow-card">
            <h3 className="mb-4 text-lg font-bold text-dark dark:text-white">Live Logs</h3>
            <div className="h-40 overflow-hidden relative">
                <div className="absolute inset-0 bg-black text-green-500 font-mono text-sm p-4 overflow-y-auto">
                    <p className="opacity-50">Initializing connection to {slug} stream...</p>
                    <p className="opacity-70">Stream buffer: OK</p>
                    <p>Latency: 23ms</p>
                    <p className="animate-pulse">Monitoring active...</p>
                </div>
            </div>
        </div>
      </div>
  );
}
