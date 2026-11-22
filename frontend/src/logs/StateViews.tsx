import type { ReactNode } from "react";

export function Skeleton({ rows = 3, cards = 0 }: { rows?: number; cards?: number }) {
  return (
    <div className="lg-skeleton" aria-busy="true" aria-label="Loading">
      {cards > 0 && (
        <div className="lg-skeleton-cards">
          {Array.from({ length: cards }, (_, i) => <div key={i} className="lg-skel lg-skel--card" />)}
        </div>
      )}
      {Array.from({ length: rows }, (_, i) => <div key={i} className="lg-skel lg-skel--row" />)}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="lg-state" role="alert">
      <p className="lg-state-title">Something went wrong</p>
      <p className="lg-state-text">{message}</p>
      <button className="lg-btn lg-btn--primary" onClick={onRetry}>Retry</button>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="lg-state">
      <p className="lg-state-title">{title}</p>
      {children}
    </div>
  );
}
