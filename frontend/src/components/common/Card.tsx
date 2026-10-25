import React from 'react';

interface CardProps {
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  icon?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  headerAction?: React.ReactNode;
  'data-testid'?: string;
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  badge,
  icon,
  children,
  className = '',
  headerAction,
  'data-testid': testId,
}) => {
  return (
    <div
      className={`rounded-xl bg-studio-card border border-studio-cardBorder p-5 shadow-sm shadow-black/20 ${className}`}
      data-testid={testId}
    >
      <div className="flex items-start justify-between mb-4 border-b border-studio-border/50 pb-3">
        <div className="flex items-center space-x-2.5">
          {icon && <div className="text-studio-accent">{icon}</div>}
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-sm font-semibold text-studio-text tracking-tight">{title}</h2>
              {badge}
            </div>
            {subtitle && <p className="text-[11px] text-studio-muted mt-0.5">{subtitle}</p>}
          </div>
        </div>
        {headerAction && <div>{headerAction}</div>}
      </div>
      <div>{children}</div>
    </div>
  );
};
