import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import ErrorBoundary from './components/ErrorBoundary'

// Environment and build info logging
console.log('🚀 HifazatAI Frontend Starting...', {
  mode: import.meta.env.MODE,
  dev: import.meta.env.DEV,
  prod: import.meta.env.PROD,
  apiUrl: import.meta.env.VITE_API_URL,
  nodeEnv: import.meta.env.VITE_NODE_ENV,
  timestamp: new Date().toISOString()
});

// Enhanced error handling for production builds
window.addEventListener('error', (event) => {
  const errorInfo = {
    message: event.error?.message || event.message,
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno,
    stack: event.error?.stack,
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    url: window.location.href
  };
  
  console.error('🔥 Global JavaScript Error:', errorInfo);
  
  // In production, you might want to send this to an error tracking service
  if (import.meta.env.PROD) {
    // Example: Send to error tracking service
    // errorTrackingService.captureError(errorInfo);
  }
});

window.addEventListener('unhandledrejection', (event) => {
  const errorInfo = {
    reason: event.reason,
    promise: event.promise,
    timestamp: new Date().toISOString(),
    url: window.location.href
  };
  
  console.error('🔥 Unhandled Promise Rejection:', errorInfo);
  
  // Prevent the default browser behavior (logging to console)
  event.preventDefault();
  
  // In production, you might want to send this to an error tracking service
  if (import.meta.env.PROD) {
    // Example: Send to error tracking service
    // errorTrackingService.captureError(errorInfo);
  }
});

// Check for required environment variables
const requiredEnvVars = ['VITE_API_URL'];
const missingEnvVars = requiredEnvVars.filter(varName => !import.meta.env[varName]);

if (missingEnvVars.length > 0) {
  console.warn('⚠️ Missing environment variables:', missingEnvVars);
}

// DOM ready check
const initializeApp = () => {
  const rootElement = document.getElementById('root');
  
  if (!rootElement) {
    const error = new Error('Root element not found. Make sure there is a div with id="root" in your HTML.');
    console.error('🔥 Critical Error:', error);
    
    // Create a fallback error display
    document.body.innerHTML = `
      <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        font-family: system-ui, -apple-system, sans-serif;
        background: #f3f4f6;
        color: #374151;
        text-align: center;
        padding: 2rem;
      ">
        <div>
          <h1 style="color: #dc2626; margin-bottom: 1rem;">Application Error</h1>
          <p style="margin-bottom: 1rem;">Failed to initialize HifazatAI frontend.</p>
          <p style="font-size: 0.875rem; color: #6b7280;">Check the browser console for more details.</p>
        </div>
      </div>
    `;
    throw error;
  }

  try {
    console.log('✅ Root element found, initializing React app...');
    
    const root = createRoot(rootElement);
    root.render(
      <StrictMode>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </StrictMode>
    );
    
    console.log('✅ React app initialized successfully');
  } catch (error) {
    console.error('🔥 Failed to render React app:', error);
    
    // Fallback error display
    rootElement.innerHTML = `
      <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        font-family: system-ui, -apple-system, sans-serif;
        background: #f3f4f6;
        color: #374151;
        text-align: center;
        padding: 2rem;
      ">
        <div>
          <h1 style="color: #dc2626; margin-bottom: 1rem;">Render Error</h1>
          <p style="margin-bottom: 1rem;">Failed to render HifazatAI application.</p>
          <p style="font-size: 0.875rem; color: #6b7280;">Check the browser console for more details.</p>
          <button onclick="window.location.reload()" style="
            margin-top: 1rem;
            padding: 0.5rem 1rem;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 0.375rem;
            cursor: pointer;
          ">Reload Page</button>
        </div>
      </div>
    `;
    throw error;
  }
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp);
} else {
  initializeApp();
}
