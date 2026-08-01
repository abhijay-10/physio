import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '40px', background: '#222', color: 'white', minHeight: '100vh' }}>
          <h1 style={{ color: '#ef4444' }}>Something went wrong.</h1>
          <p>An unexpected error caused the application to crash.</p>
          <pre style={{ background: '#000', padding: '20px', overflowX: 'auto', color: '#10b981' }}>
            {this.state.error?.toString()}
            <br />
            {this.state.errorInfo?.componentStack}
          </pre>
          <button 
            onClick={() => {
              localStorage.clear();
              window.location.href = '/';
            }}
            style={{ padding: '12px 24px', background: '#ef4444', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', marginTop: '20px' }}
          >
            Clear Data & Restart
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
