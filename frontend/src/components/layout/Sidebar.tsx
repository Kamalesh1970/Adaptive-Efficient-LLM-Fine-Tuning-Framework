import React from 'react';
import { 
  PlusCircle, 
  Box, 
  BarChart2,
  Database, 
  History, 
  Settings, 
  Server,
  Cpu,
  Sparkles
} from 'lucide-react';
import { SystemStatus } from '../../types/training';

interface SidebarProps {
  systemStatus: SystemStatus | null;
  backendOnline: boolean;
  activeNav?: string;
  onSelectNav?: (nav: string) => void;
  onOpenPrivacy?: () => void;
  onOpenTerms?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  systemStatus,
  backendOnline,
  activeNav = 'new-training',
  onSelectNav,
  onOpenPrivacy,
  onOpenTerms,
}) => {
  const navItems = [
    { id: 'new-training', label: 'New Training', icon: PlusCircle },
    { id: 'models', label: 'Models & Artifacts', icon: Box },
    { id: 'evaluation', label: 'Model Evaluation', icon: BarChart2 },
    { id: 'platform', label: 'Platform & Infra', icon: Server },
    { id: 'datasets', label: 'Datasets', icon: Database },
    { id: 'history', label: 'Training History', icon: History },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside 
      className="w-64 bg-studio-sidebar border-r border-studio-border flex flex-col justify-between shrink-0 select-none"
      data-testid="sidebar"
    >
      <div>
        {/* Brand / Logo */}
        <div className="p-4 border-b border-studio-border/60">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-md bg-studio-card border border-studio-border flex items-center justify-center text-studio-accent shadow-xs">
              <Sparkles className="w-4 h-4 stroke-[2.5]" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-studio-text tracking-tight flex items-center gap-1.5">
                <span>Adaptive Studio</span>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-studio-cardBorder text-studio-accent font-medium">
                  v0.1
                </span>
              </h1>
              <p className="text-[11px] text-studio-muted">LLM Fine-Tuning Studio</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="p-3 space-y-1">
          <div className="px-3 py-1.5 text-[10px] font-semibold text-studio-muted/70 uppercase tracking-wider font-mono">
            Workspaces
          </div>
          {navItems.map((item) => {
            const isSelected = activeNav === item.id;
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => onSelectNav?.(item.id)}
                className={`w-full flex items-center space-x-2.5 px-3 py-2 rounded-md text-xs font-medium transition-all ${
                  isSelected
                    ? 'bg-studio-accent/10 text-studio-accent border border-studio-accent/30 shadow-xs font-semibold'
                    : 'text-studio-muted hover:text-studio-text hover:bg-studio-card/60 border border-transparent'
                }`}
                data-testid={`nav-${item.id}`}
              >
                <Icon className={`w-4 h-4 ${isSelected ? 'text-studio-accent' : 'text-studio-muted'}`} />
                <span>{item.label}</span>
                {isSelected && (
                  <span className="ml-auto w-1.5 h-1.5 rounded-full bg-studio-accent" />
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Backend & Environment Status Footer */}
      <div className="p-3 border-t border-studio-border/60 bg-studio-bg/50 space-y-2">
        <div className="rounded-lg p-2.5 bg-studio-card border border-studio-border text-xs space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-studio-muted flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5 text-studio-muted" /> Backend API
            </span>
            <span 
              className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${
                backendOnline 
                  ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-800/60'
                  : 'bg-rose-950/80 text-rose-400 border border-rose-800/60'
              }`}
              data-testid="backend-status-badge"
            >
              <span className={`w-1.5 h-1.5 rounded-full mr-1 ${backendOnline ? 'bg-emerald-400' : 'bg-rose-400'}`} />
              {backendOnline ? 'Connected' : 'Offline'}
            </span>
          </div>

          <div className="text-[11px] text-studio-muted truncate">
            {backendOnline && systemStatus ? (
              <div className="space-y-0.5">
                <div className="text-studio-text font-mono text-[10px] truncate" data-testid="sidebar-device-status">
                  Py {systemStatus.python_version} • {systemStatus.gpu_available || systemStatus.cuda_available ? (systemStatus.gpu_name || systemStatus.primary_gpu_name || 'CUDA GPU') : 'CPU Mode'}
                </div>
              </div>
            ) : (
              <span className="text-amber-400/90 text-[10px]" data-testid="sidebar-device-status">Backend Unavailable</span>
            )}
          </div>
        </div>

        {/* Legal & Policy Links */}
        <div className="flex items-center justify-between px-1 text-[10px] text-studio-muted/80">
          <button
            onClick={onOpenPrivacy}
            className="hover:text-studio-accent transition-colors underline-offset-2 hover:underline"
            data-testid="link-privacy"
          >
            Privacy Policy
          </button>
          <span>-</span>
          <button
            onClick={onOpenTerms}
            className="hover:text-studio-accent transition-colors underline-offset-2 hover:underline"
            data-testid="link-terms"
          >
            Terms & Conditions
          </button>
        </div>
      </div>
    </aside>
  );
};
