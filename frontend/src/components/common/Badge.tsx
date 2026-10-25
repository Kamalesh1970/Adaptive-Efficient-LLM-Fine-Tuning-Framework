import React from 'react';

interface BadgeProps {
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  children: React.ReactNode;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'default',
  children,
  className = '',
}) => {
  const variantStyles = {
    default: 'bg-studio-cardBorder text-studio-muted border-studio-border',
    success: 'bg-emerald-950/70 text-emerald-400 border-emerald-800/60',
    warning: 'bg-amber-950/70 text-amber-400 border-amber-800/60',
    error: 'bg-rose-950/70 text-rose-400 border-rose-800/60',
    info: 'bg-sky-950/70 text-sky-400 border-sky-800/60',
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-medium border ${variantStyles[variant]} ${className}`}
    >
      {children}
    </span>
  );
};
