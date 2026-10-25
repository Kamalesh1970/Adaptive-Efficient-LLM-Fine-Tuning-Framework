import React from 'react';
import { 
  PlusCircle, 
  Box, 
  Database, 
  History, 
  Settings, 
  Cpu,
  Sparkles
} from 'lucide-react';
import { SystemStatus } from '../../types/training';

interface SidebarProps {
  systemStatus: SystemStatus | null;
  backendOnline: boolean;
  activeNav?: string;
  onSelectNav?: (nav: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  systemStatus,
  backendOnline,
  activeNav = 'new-training',
  onSelectNav,
}) => {
  const navItems = [
    { id: 'new-training', label: 'New Training', icon: PlusCircle, active: true },
    { id: 'models', label: 'Models', icon: Box, active: false },
    { id: 'datasets', label: 'Datasets', icon: Database, active: false },
    { id: 'history', label: 'Training History', icon: History, active: false },
    { id: 'settings', label: 'Settings', icon: Settings, active: false },
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
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-studio-accent to-studio-teal flex items-center justify-center text-studio-bg shadow-sm shadow-studio-accent/20">
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
                    ? 'bg-studio-accent/10 text-studio-accent border border-studio-accent/30 shadow-sm shadow-studio-accent/10 font-semibold'
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
      <div className="p-3 border-t border-studio-border/60 bg-studio-bg/50">
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
                <div className="text-studio-text font-mono text-[10px] truncate">
                  Py {systemStatus.python_version} • {systemStatus.cuda_available ? (systemStatus.primary_gpu_name || 'CUDA GPU') : 'CPU Mode'}
                </div>
              </div>
            ) : (
              <span className="text-amber-400/90 text-[10px]">FastAPI server unreachable</span>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
};
