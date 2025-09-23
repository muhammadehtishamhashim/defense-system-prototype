# HifazatAI Frontend

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── layout/         # Layout components (Header, Sidebar, etc.)
│   └── ui/             # Basic UI components (Button, Card, Modal, etc.)
├── pages/              # Page components for different routes
├── services/           # API service layer
├── types/              # TypeScript type definitions
├── utils/              # Utility functions
├── App.tsx             # Main application component with routing
├── main.tsx            # Application entry point
└── index.css           # Global styles with Tailwind CSS
```

## Key Features

- **Responsive Design**: Built with Tailwind CSS for mobile-first responsive design
- **React Router**: Client-side routing for single-page application navigation
- **TypeScript**: Full TypeScript support for type safety
- **Modular Components**: Reusable UI components following design system principles
- **API Integration**: Axios-based service layer for backend communication

## Available Routes

- `/` - Dashboard (overview and statistics)
- `/alerts` - Alert monitoring interface
- `/video` - Video analysis interface  
- `/system` - System status and monitoring
- `/settings` - System configuration

## UI Components

### Layout Components
- `DashboardLayout` - Main application layout with sidebar and header
- `Header` - Top navigation bar with notifications and user menu
- `Sidebar` - Left navigation sidebar with menu items

### UI Components
- `Button` - Customizable button with variants (default, outline, ghost, etc.)
- `Card` - Container component with header, content, and footer sections
- `Modal` - Overlay dialog component using Headless UI
- `Badge` - Status indicator with color variants
- `LoadingSpinner` - Loading indicator component

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Environment Variables

Create a `.env` file in the frontend directory:

```
VITE_API_URL=http://localhost:8000
VITE_NODE_ENV=development
```