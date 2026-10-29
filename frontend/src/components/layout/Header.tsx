import React from 'react';
import { Sliders, Activity, Clock, Box, BarChart2, LucideIcon } from 'lucide-react';

export type StudioTab = 'configure' | 'current-run' | 'history' | 'artifacts' | 'evaluation';

interface HeaderProps {
  activeTab: StudioTab;
  onTabChange: (tab: StudioTab) => void;
}

interface TabItem {
  id: StudioTab;
  label: string;
  icon: LucideIcon;
  badge?: string;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, onTabChange }) => {
  const tabs: TabItem[] = [
    { id: 'configure', label: 'Configure', icon: Sliders },
    { id: 'current-run', label: 'Current Run', icon: Activity, badge: 'Monitor' },
    { id: 'history', label: 'History', icon: Clock },
    { id: 'artifacts', label: 'Artifacts', icon: Box },
    { id: 'evaluation', label: 'Evaluation', icon: BarChart2 },
  ];

  return (
    <header className="border-b border-studio-border bg-studio-bg/95 backdrop-blur px-6 py-4 shrink-0">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Title and Subtitle */}
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold text-studio-text tracking-tight">Train</h1>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-studio-card border border-studio-border text-studio-muted">
              Draft Configuration
            </span>
          </div>
          <p className="text-xs text-studio-muted mt-0.5">
            Configure and start fine-tuning
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center space-x-1 p-1 bg-studio-card/80 border border-studio-border rounded-lg self-start md:self-auto">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-studio-accent text-studio-bg font-semibold shadow-sm shadow-studio-accent/20'
                    : 'text-studio-muted hover:text-studio-text hover:bg-studio-cardHover'
                }`}
                data-testid={`tab-${tab.id}`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
                {tab.badge && (
                  <span className={`text-[9px] px-1 py-0.2 rounded font-mono ${isActive ? 'bg-studio-bg/20 text-studio-bg' : 'bg-studio-cardBorder text-studio-muted'}`}>
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
