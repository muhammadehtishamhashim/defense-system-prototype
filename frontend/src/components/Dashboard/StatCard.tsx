"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface StatCardProps {
  title: string;
  count: number;
  type: 'threat' | 'theft' | 'border';
  icon?: React.ReactNode;
}

export function StatCard({ title, count, type, icon }: StatCardProps) {
    const colorClass = type === 'threat' ? "text-cyber-red" : type === 'theft' ? "text-orange-500" : "text-blue-500";
    const bgClass = type === 'threat' ? "bg-cyber-red/10 border-cyber-red/20" : type === 'theft' ? "bg-orange-500/10 border-orange-500/20" : "bg-blue-500/10 border-blue-500/20";
    
    return (
        <div className={cn("rounded-[10px] border p-6 shadow-1 dark:shadow-card", bgClass)}>
            <div className="flex items-center justify-between">
                <div>
                    <h4 className="text-heading-6 font-bold text-dark dark:text-white">
                        {count}
                    </h4>
                    <span className="text-sm font-medium text-dark-5 dark:text-dark-6">{title}</span>
                </div>
                
                <div className={cn("flex h-11.5 w-11.5 items-center justify-center rounded-full text-white", type === 'threat' ? "bg-cyber-red" : type === 'theft' ? "bg-orange-500" : "bg-blue-500")}>
                    {icon || "!"}
                </div>
            </div>
        </div>
    );
}
