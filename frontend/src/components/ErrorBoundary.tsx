import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
  errorId: string;
}

class ErrorBoundary extends Component<Props, State> {
  private retryCount = 0;
  private maxRetries = 3;

  public state: State = {
    hasError: false,
    errorId: ''
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    // Generate a unique error ID for tracking
    const errorId = `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return { 
      hasError: true, 
      error,
      errorId
    };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Enhanced error logging with context
    const errorDetails = {
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack,
      },
      errorInfo: {
        componentStack: errorInfo.componentStack,
      },
      context: {
        url: window.location.href,
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
        errorId: this.state.errorId,
        retryCount: this.retryCount,
        environment: {
          mode: import.meta.env.MODE,
          dev: import.meta.env.DEV,
          prod: import.meta.env.PROD,
          apiUrl: import.meta.env.VITE_API_URL,
        }
      }
    };

    console.error('🔥 ErrorBoundary caught an error:', errorDetails);

    // Store error info in state for display
    this.setState({ errorInfo });

    // In production, send error to monitoring service
    if (import.meta.env.PROD) {
      try {
        // Example: Send to error tracking service
        // errorTrackingService.captureException(error, errorDetails);
        
        // Store in localStorage for debugging
        const storedErrors = JSON.parse(localStorage.getItem('hifazat_errors') || '[]');
        storedErrors.push(errorDetails);
        // Keep only last 10 errors
        if (storedErrors.length > 10) {
          storedErrors.shift();
        }
        localStorage.setItem('hifazat_errors', JSON.stringify(storedErrors));
      } catch (storageError) {
        console.warn('Failed to store error details:', storageError);
      }
    }
  }

  private handleRetry = () => {
    if (this.retryCount < this.maxRetries) {
      this.retryCount++;
      console.log(`🔄 Retrying... (${this.retryCount}/${this.maxRetries})`);
      this.setState({
        hasError: false,
        error: undefined,
        errorInfo: undefined,
        errorId: ''
      });
    } else {
      console.log('🚫 Max retries reached, reloading page...');
      window.location.reload();
    }
  };

  private handleReload = () => {
    window.location.reload();
  };

  private handleClearErrors = () => {
    localStorage.removeItem('hifazat_errors');
    console.log('🧹 Error history cleared');
  };

  public render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const canRetry = this.retryCount < this.maxRetries;
      const isDev = import.meta.env.DEV;

      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
          <div className="max-w-2xl w-full bg-white shadow-lg rounded-lg p-6">
            <div className="flex items-center justify-center w-12 h-12 mx-auto bg-red-100 rounded-full">
              <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 18.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            
            <div className="mt-4 text-center">
              <h3 className="text-lg font-medium text-gray-900">Application Error</h3>
              <p className="mt-2 text-sm text-gray-500">
                HifazatAI encountered an unexpected error. This has been logged for investigation.
              </p>
              
              {this.state.errorId && (
                <p className="mt-1 text-xs text-gray-400 font-mono">
                  Error ID: {this.state.errorId}
                </p>
              )}
            </div>

            <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-center">
              {canRetry && (
                <button
                  onClick={this.handleRetry}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                >
                  Try Again ({this.maxRetries - this.retryCount} left)
                </button>
              )}
              
              <button
                onClick={this.handleReload}
                className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
              >
                Reload Page
              </button>
            </div>

            {/* Error details for development or debugging */}
            {(isDev || this.state.error) && (
              <details className="mt-6 text-left">
                <summary className="text-sm text-gray-600 cursor-pointer hover:text-gray-800">
                  Technical Details
                </summary>
                <div className="mt-3 space-y-3">
                  {this.state.error && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700">Error Message:</h4>
                      <pre className="mt-1 text-xs bg-red-50 border border-red-200 p-2 rounded overflow-auto text-red-800">
                        {this.state.error.name}: {this.state.error.message}
                      </pre>
                    </div>
                  )}
                  
                  {this.state.error?.stack && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700">Stack Trace:</h4>
                      <pre className="mt-1 text-xs bg-gray-50 border border-gray-200 p-2 rounded overflow-auto max-h-32">
                        {this.state.error.stack}
                      </pre>
                    </div>
                  )}
                  
                  {this.state.errorInfo?.componentStack && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700">Component Stack:</h4>
                      <pre className="mt-1 text-xs bg-gray-50 border border-gray-200 p-2 rounded overflow-auto max-h-32">
                        {this.state.errorInfo.componentStack}
                      </pre>
                    </div>
                  )}
                  
                  <div className="pt-2 border-t border-gray-200">
                    <button
                      onClick={this.handleClearErrors}
                      className="text-xs text-gray-500 hover:text-gray-700 underline"
                    >
                      Clear Error History
                    </button>
                  </div>
                </div>
              </details>
            )}

            {/* Environment info for debugging */}
            {isDev && (
              <details className="mt-4 text-left">
                <summary className="text-sm text-gray-600 cursor-pointer hover:text-gray-800">
                  Environment Info
                </summary>
                <div className="mt-2 text-xs bg-blue-50 border border-blue-200 p-2 rounded">
                  <div><strong>Mode:</strong> {import.meta.env.MODE}</div>
                  <div><strong>API URL:</strong> {import.meta.env.VITE_API_URL}</div>
                  <div><strong>Timestamp:</strong> {new Date().toISOString()}</div>
                  <div><strong>URL:</strong> {window.location.href}</div>
                </div>
              </details>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;