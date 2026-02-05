
"use client";

import { useEffect, useState } from "react";
import { StatCard } from "@/components/Dashboard/StatCard";
import dynamic from 'next/dynamic';
const ReactApexChart = dynamic(() => import('react-apexcharts'), { ssr: false });
import { ApexOptions } from "apexcharts";
import { useAlerts } from "@/context/AlertContext";
import { cn } from "@/lib/utils";

export default function Home() {
  const { alerts, latestAlert } = useAlerts();
  const [stats, setStats] = useState({ threats: 0, thefts: 0, border_anomalies: 0 });

  // Poll for stats (Current active detections)
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch("http://localhost:8000/stats");
        const data = await res.json();
        setStats(data);
      } catch (e) {
        console.error("Failed to fetch stats");
      }
    };
    
    // Initial fetch
    fetchStats();
    
    // Poll every second
    const interval = setInterval(fetchStats, 1000);
    return () => clearInterval(interval);
  }, []);

  // Poll for stats and update graph
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch("http://localhost:8000/stats");
        const data = await res.json();
        setStats(data);
      } catch (e) {
        console.error("Failed to fetch stats");
      }
    };
    
    fetchStats();
    const interval = setInterval(fetchStats, 2000);
    return () => clearInterval(interval);
  }, []);

  const options: ApexOptions = {
    chart: {
      type: "bar",
      height: 350,
      fontFamily: "Satoshi, sans-serif",
      toolbar: { show: false },
      background: 'transparent',
    },
    colors: ["#FF003C", "#F97316", "#3C50E0"],
    plotOptions: {
        bar: { 
            borderRadius: 4,
            columnWidth: "45%",
            distributed: true,
        }
    },
    dataLabels: { enabled: false },
    legend: { show: false }, // Legend redundant with distributed bars and x-axis labels
    grid: { show: true, borderColor: "#333", strokeDashArray: 0 },
    xaxis: {
      categories: ["Threats", "Thefts", "Border Anomalies"],
      axisBorder: { show: false },
      axisTicks: { show: false },
      labels: { style: { colors: "#9ca3af", fontSize: "14px" } },
    },
    yaxis: {
        labels: { style: { colors: "#9ca3af" } }
    },
    tooltip: { theme: 'dark' }
  };
  
  const series = [
      { 
          name: "Total Events", 
          data: [stats.threats, stats.thefts, stats.border_anomalies] 
      },
  ];

  return (
    <>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3 md:gap-6 2xl:gap-7.5">
        <StatCard title="Active Threats" count={stats.threats} type="threat" />
        <StatCard title="Theft Attempts" count={stats.thefts} type="theft" />
        <StatCard title="Border Anomalies" count={stats.border_anomalies} type="border" />
      </div>

      <div className="mt-4 grid grid-cols-12 gap-4 md:mt-6 md:gap-6 2xl:mt-7.5 2xl:gap-7.5">
        
        {/* Activity Graph */}
        <div className="col-span-12 rounded-[10px] border border-stroke bg-white px-5 pb-5 pt-7.5 shadow-1 dark:border-dark-3 dark:bg-gray-dark dark:shadow-card xl:col-span-8">
            <h4 className="mb-2 text-xl font-bold text-dark dark:text-white">Event Distribution</h4>
            <div id="chartOne" className="-ml-5 h-[355px] w-[105%]">
              <ReactApexChart options={options} series={series} type="bar" height={350} />
            </div>
        </div>

        {/* Recent Alerts Feed */}
        <div className="col-span-12 rounded-[10px] border border-stroke bg-white py-6 shadow-1 dark:border-dark-3 dark:bg-gray-dark dark:shadow-card xl:col-span-4">
          <h4 className="mb-6 px-7.5 text-xl font-bold text-dark dark:text-white">
            Recent Signals
          </h4>

          <div className="flex flex-col gap-5 px-7.5 max-h-[350px] overflow-y-auto custom-scrollbar">
            {alerts.slice(0, 10).map((alert, idx) => (
                <div key={idx} className="flex items-center gap-5">
                    <div className="relative flex h-14 w-14 items-center justify-center rounded-full bg-gray-2 dark:bg-dark-2">
                         <span className={cn("h-3 w-3 rounded-full", 
                             alert.source === 'threat' ? "bg-cyber-red" : 
                             alert.source === 'theft' ? "bg-orange-500" : "bg-blue-500"
                         )}></span>
                    </div>
                    <div className="flex flex-1 items-center justify-between">
                        <div>
                            <h5 className="font-medium text-dark dark:text-white capitalize">
                                {alert.source} Detected
                            </h5>
                            <span className="text-sm text-dark-5 dark:text-dark-6">
                                {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                            </span>
                        </div>
                        <span className="inline-block rounded-md bg-transparent px-2.5 py-1 text-sm font-medium text-dark dark:text-white">
                           {alert.count}
                        </span>
                    </div>
                </div>
            ))}
            {alerts.length === 0 && <div className="text-center text-gray-500">No recent alerts</div>}
          </div>
        </div>
      </div>
    </>
  );
}
