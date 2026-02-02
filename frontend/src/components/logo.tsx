import { Shield } from "lucide-react";

export function Logo() {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary dark:bg-white/5 dark:text-white">
        <Shield className="h-6 w-6" />
      </div>
      <span className="text-xl font-bold text-dark dark:text-white text-nowrap">
        Hifazat AI
      </span>
    </div>
  );
}
