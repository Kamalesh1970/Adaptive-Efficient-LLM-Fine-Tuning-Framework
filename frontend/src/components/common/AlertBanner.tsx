import React from 'react';
import { AlertCircle, AlertTriangle, CheckCircle, Info, X } from 'lucide-react';

interface AlertBannerProps {
  type?: 'info' | 'warning' | 'error' | 'success';
  title?: string;
  message: string;
  onClose?: () => void;
  className?: string;
}

export const AlertBanner: React.FC<AlertBannerProps> = ({
  type = 'info',
  title,
  message,
  onClose,
  className = '',
}) => {
  const icons = {
    info: <Info className="w-4 h-4 text-sky-400 shrink-0" />,
    warning: <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />,
    error: <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />,
    success: <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />,
  };

  const borders = {
    info: 'border-sky-800/60 bg-sky-950/40 text-sky-200',
    warning: 'border-amber-800/60 bg-amber-950/40 text-amber-200',
    error: 'border-rose-800/60 bg-rose-950/40 text-rose-200',
    success: 'border-emerald-800/60 bg-emerald-950/40 text-emerald-200',
  };

  return (
    <div
      className={`flex items-start gap-3 p-3 rounded-lg border text-xs ${borders[type]} ${className}`}
      data-testid="alert-banner"
    >
      <div className="mt-0.5">{icons[type]}</div>
      <div className="flex-1">
        {title && <h3 className="font-semibold mb-0.5 text-studio-text">{title}</h3>}
        <div className="text-[11px] leading-relaxed opacity-90">{message}</div>
      </div>
      {onClose && (
        <button
          onClick={onClose}
          className="p-1 hover:bg-white/10 rounded transition-colors text-studio-muted"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
};
