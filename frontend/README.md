# Hifazat AI - Frontend

**Hifazat AI** is an intelligent defense and surveillance dashboard built with Next.js 16 and Tailwind CSS. It serves as the Command Center for monitoring real-time threats, thefts, and border anomalies.

## Features

-   **Real-time Dashboard**: Live statistics and activity graphs.
-   **Monitoring**: View live video feeds with bounding box overlays from the backend.
-   **Verification Center**: Persistent alert queue for manual review (`/verify`).
-   **Hifazat AI Branding**: Custom "Cyber" theme with dark mode default.

## Installation

1.  Install dependencies:
    ```bash
    pnpm install
    ```

2.  Run the development server:
    ```bash
    pnpm dev
    ```
    > **Note**: Use `pnpm dev` to ensure the latest changes are active.

3.  Open [http://localhost:3000](http://localhost:3000)

## Tech Stack

-   **Framework**: Next.js 16 (App Router)
-   **Styling**: Tailwind CSS
-   **State Management**: React Context (`AlertContext`)
-   **Icons**: Lucide React
-   **Charts**: ApexCharts
