import { Component, type ReactNode } from 'react'

export class ErrorBoundary extends Component<{children: ReactNode}, {error: string | null}> {
  state = { error: null as string | null }
  static getDerivedStateFromError(err: unknown) {
    return { error: err instanceof Error ? err.message : String(err) }
  }
  render() {
    if (this.state.error) return <div className="stack"><h2>Stage failed to render</h2><pre>{this.state.error}</pre></div>
    return this.props.children
  }
}
