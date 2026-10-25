import React, { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { TrainingPage } from './features/training/TrainingPage';
import { SystemStatus, HealthStatus } from './types/training';
import { fetchHealth, fetchSystemStatus } from './services/api';
import { Activity, Clock } from 'lucide-react';

export const App: React.FC = () => {
  const [activeNav, setActiveNav] = useState('new-training');
  const [activeTab, setActiveTab] = useState<'configure' | 'current-run' | 'history'>('configure');
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [, setHealth] = useState<HealthStatus | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);

  const checkBackendStatus = useCallback(async () => {
    try {
      const [h, s] = await Promise.all([fetchHealth(), fetchSystemStatus()]);
      setHealth(h);
      setSystemStatus(s);
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
      setSystemStatus(null);
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    checkBackendStatus();
    // Poll backend status periodically
    const interval = setInterval(checkBackendStatus, 15000);
    return () => clearInterval(interval);
  }, [checkBackendStatus]);

  return (
    <div className="flex h-screen bg-studio-bg text-studio-text overflow-hidden font-sans">
      {/* Left Sidebar */}
      <Sidebar
        systemStatus={systemStatus}
        backendOnline={backendOnline}
        activeNav={activeNav}
        onSelectNav={setActiveNav}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header activeTab={activeTab} onTabChange={setActiveTab} />

        {/* Tab Views */}
        {activeTab === 'configure' && (
          <TrainingPage
            systemStatus={systemStatus}
            backendOnline={backendOnline}
            onRefreshSystemStatus={checkBackendStatus}
          />
        )}

        {activeTab === 'current-run' && (
          <div className="flex-1 flex items-center justify-center p-8 text-center" data-testid="current-run-placeholder">
            <div className="max-w-md space-y-3 p-8 rounded-xl bg-studio-card border border-studio-cardBorder">
              <Activity className="w-10 h-10 text-studio-accent mx-auto animate-pulse" />
              <h2 className="text-base font-bold text-studio-text">Real-Time Training Monitor</h2>
              <p className="text-xs text-studio-muted">
                Live loss curves, gradient norms, VRAM consumption telemetry, and speed monitors will be streaming live here in Phase 4.
              </p>
              <div className="inline-block px-2 py-0.5 rounded text-[10px] font-mono bg-studio-cardBorder text-studio-muted">
                Planned for Phase 4: Training Engine Execution
              </div>
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="flex-1 flex items-center justify-center p-8 text-center" data-testid="history-placeholder">
            <div className="max-w-md space-y-3 p-8 rounded-xl bg-studio-card border border-studio-cardBorder">
              <Clock className="w-10 h-10 text-studio-muted mx-auto" />
              <h2 className="text-base font-bold text-studio-text">Training Runs History</h2>
              <p className="text-xs text-studio-muted">
                Archived fine-tuning checkpoints, hyperparameter logs, and evaluation metrics will be indexed here.
              </p>
              <div className="inline-block px-2 py-0.5 rounded text-[10px] font-mono bg-studio-cardBorder text-studio-muted">
                Planned for Phase 5: Evaluation & Experiment Tracking
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
