import React, { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Header, StudioTab } from './components/layout/Header';
import { TrainingPage } from './features/training/TrainingPage';
import { CurrentRunView } from './features/training/CurrentRunView';
import { ArtifactsView } from './features/artifacts/ArtifactsView';
import { EvaluationView } from './features/evaluation/EvaluationView';
import { PlatformOperationsView } from './features/platform/PlatformOperationsView';
import { PrivacyPolicy } from './pages/PrivacyPolicy';
import { TermsConditions } from './pages/TermsConditions';
import { SystemStatus, HealthStatus } from './types/training';
import { fetchHealth, fetchSystemStatus } from './services/api';
import { TrainingHistoryView } from './features/training/TrainingHistoryView';

export const App: React.FC = () => {
  const [activeNav, setActiveNav] = useState('new-training');
  const [activeTab, setActiveTab] = useState<StudioTab>('configure');
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [evaluationArtifactId, setEvaluationArtifactId] = useState<string | null>(null);
  const [modalView, setModalView] = useState<'privacy' | 'terms' | null>(null);
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

  const handleSelectNav = (nav: string) => {
    setModalView(null);
    setActiveNav(nav);
    if (nav === 'new-training') {
      setActiveTab('configure');
    } else if (nav === 'artifacts' || nav === 'models') {
      setActiveTab('artifacts');
    } else if (nav === 'evaluation') {
      setActiveTab('evaluation');
    } else if (nav === 'history') {
      setActiveTab('history');
    } else if (nav === 'platform') {
      setActiveTab('platform');
    }
  };

  const handleTabChange = (tab: StudioTab) => {
    setActiveTab(tab);
    if (tab === 'configure') {
      setActiveNav('new-training');
    } else if (tab === 'artifacts') {
      setActiveNav('artifacts');
    } else if (tab === 'evaluation') {
      setActiveNav('evaluation');
    } else if (tab === 'history') {
      setActiveNav('history');
    } else if (tab === 'platform') {
      setActiveNav('platform');
    }
  };

  return (
    <div className="flex h-screen bg-studio-bg text-studio-text overflow-hidden font-sans">
      {/* Left Sidebar */}
      <Sidebar
        systemStatus={systemStatus}
        backendOnline={backendOnline}
        activeNav={activeNav}
        onSelectNav={handleSelectNav}
        onOpenPrivacy={() => setModalView('privacy')}
        onOpenTerms={() => setModalView('terms')}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {modalView ? (
          modalView === 'privacy' ? (
            <PrivacyPolicy onBack={() => setModalView(null)} />
          ) : (
            <TermsConditions onBack={() => setModalView(null)} />
          )
        ) : (
          <>
            <Header activeTab={activeTab} onTabChange={handleTabChange} />

            {/* Tab Views */}
            {activeTab === 'configure' && (
              <TrainingPage
                systemStatus={systemStatus}
                backendOnline={backendOnline}
                onRefreshSystemStatus={checkBackendStatus}
                onLaunchRun={(runId) => {
                  setCurrentRunId(runId);
                  setActiveTab('current-run');
                }}
              />
            )}

            {activeTab === 'current-run' && (
              <CurrentRunView
                activeRunId={currentRunId}
                onNavigateToConfigure={() => setActiveTab('configure')}
                onNavigateToArtifacts={() => {
                  setActiveTab('artifacts');
                  setActiveNav('artifacts');
                }}
                onNavigateToEvaluation={() => {
                  setActiveTab('evaluation');
                  setActiveNav('evaluation');
                }}
              />
            )}

            {activeTab === 'history' && (
              <TrainingHistoryView
                onSelectRun={(runId) => {
                  setCurrentRunId(runId);
                  setActiveTab('current-run');
                }}
              />
            )}

            {activeTab === 'artifacts' && (
              <ArtifactsView
                initialRunId={currentRunId}
                onNavigateToEvaluation={(artId) => {
                  setEvaluationArtifactId(artId);
                  setActiveTab('evaluation');
                  setActiveNav('evaluation');
                }}
              />
            )}

            {activeTab === 'evaluation' && (
              <EvaluationView
                initialArtifactId={evaluationArtifactId}
                onNavigateToArtifacts={() => {
                  setActiveTab('artifacts');
                  setActiveNav('artifacts');
                }}
              />
            )}

            {activeTab === 'platform' && <PlatformOperationsView />}
          </>
        )}
      </div>
    </div>
  );
};

export default App;
