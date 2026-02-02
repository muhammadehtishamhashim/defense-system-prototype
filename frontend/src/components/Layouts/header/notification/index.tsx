"use client";

import {
  Dropdown,
  DropdownContent,
  DropdownTrigger,
} from "@/components/ui/dropdown";
import { useIsMobile } from "@/hooks/use-mobile";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { useState } from "react";
import { BellIcon } from "./icons";
import { useAlerts } from "@/context/AlertContext";

export function Notification() {
  const [isOpen, setIsOpen] = useState(false);
  const { alerts, clearAlerts } = useAlerts();
  const isMobile = useIsMobile();
  
  // Show badge if there are alerts
  const isDotVisible = alerts.length > 0;

  return (
    <Dropdown
      isOpen={isOpen}
      setIsOpen={(open) => {
        setIsOpen(open);
      }}
    >
      <DropdownTrigger
        className="grid size-12 place-items-center rounded-full border bg-gray-2 text-dark outline-none hover:text-primary focus-visible:border-primary focus-visible:text-primary dark:border-dark-4 dark:bg-dark-3 dark:text-white dark:focus-visible:border-primary"
        aria-label="View Notifications"
      >
        <span className="relative">
          <BellIcon />

          {isDotVisible && (
            <span
              className={cn(
                "absolute right-0 top-0 z-1 size-2 rounded-full bg-cyber-red ring-2 ring-gray-2 dark:ring-dark-3",
              )}
            >
              <span className="absolute inset-0 -z-1 animate-ping rounded-full bg-cyber-red opacity-75" />
            </span>
          )}
        </span>
      </DropdownTrigger>

      <DropdownContent
        align={isMobile ? "end" : "center"}
        className="border border-stroke bg-white px-3.5 py-3 shadow-md dark:border-dark-3 dark:bg-gray-dark min-[350px]:min-w-[20rem]"
      >
        <div className="mb-1 flex items-center justify-between px-2 py-1.5">
          <span className="text-lg font-medium text-dark dark:text-white">
            Security Alerts
          </span>
          {alerts.length > 0 && (
            <span className="rounded-md bg-cyber-red px-[9px] py-0.5 text-xs font-medium text-white">
              {alerts.length} new
            </span>
          )}
        </div>

        <ul className="mb-3 max-h-92 space-y-1.5 overflow-y-auto custom-scrollbar">
            {alerts.length === 0 ? (
                <li className="px-2 py-4 text-center text-sm text-dark-5 dark:text-dark-6">
                    No active threats detected.
                </li>
            ) : (
                alerts.map((alert, index) => (
                    <li key={index} role="menuitem">
                    <Link
                        href={`/monitor/${alert.source}`}
                        onClick={() => setIsOpen(false)}
                        className="flex items-center gap-4 rounded-lg px-2 py-1.5 outline-none hover:bg-gray-2 focus-visible:bg-gray-2 dark:hover:bg-dark-3 dark:focus-visible:bg-dark-3"
                    >
                        <div className={cn("size-10 rounded-full flex items-center justify-center text-xl font-bold text-white", 
                            alert.source === 'threat' ? "bg-cyber-red" : 
                            alert.source === 'theft' ? "bg-orange-500" : 
                            "bg-blue-500"
                        )}>
                            !
                        </div>

                        <div>
                        <strong className="block text-sm font-medium text-dark dark:text-white capitalize">
                            {alert.source} Detected
                        </strong>

                        <span className="truncate text-sm font-medium text-dark-5 dark:text-dark-6">
                            {(alert.count || 0)} instances detected at {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                        </span>
                        </div>
                    </Link>
                    </li>
                ))
            )}
        </ul>

        {alerts.length > 0 && (
            <button
            onClick={clearAlerts}
            className="block w-full rounded-lg border border-primary p-2 text-center text-sm font-medium tracking-wide text-primary outline-none transition-colors hover:bg-blue-light-5 focus:bg-blue-light-5 focus:text-primary focus-visible:border-primary dark:border-dark-3 dark:text-dark-6 dark:hover:border-dark-5 dark:hover:bg-dark-3 dark:hover:text-dark-7 dark:focus-visible:border-dark-5 dark:focus-visible:bg-dark-3 dark:focus-visible:text-dark-7"
            >
            Clear All Alerts
            </button>
        )}
      </DropdownContent>
    </Dropdown>
  );
}
