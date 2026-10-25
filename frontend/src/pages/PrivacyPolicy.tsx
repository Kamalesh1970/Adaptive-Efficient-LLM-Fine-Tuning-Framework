import React from 'react';
import { ArrowLeft, Shield, Lock, FileText, Database } from 'lucide-react';

interface PrivacyPolicyProps {
  onBack: () => void;
}

export const PrivacyPolicy: React.FC<PrivacyPolicyProps> = ({ onBack }) => {
  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-10 max-w-4xl mx-auto space-y-8 text-xs text-studio-text" data-testid="privacy-policy-page">
      <div className="border-b border-studio-border/70 pb-4">
        <button
          onClick={onBack}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-studio-muted hover:text-studio-accent mb-4 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Studio</span>
        </button>
        <div className="flex items-center gap-2.5">
          <Shield className="w-5 h-5 text-studio-accent" />
          <h1 className="text-xl font-bold tracking-tight text-studio-text">Privacy Policy</h1>
        </div>
        <p className="text-studio-muted text-xs mt-1">
          Last updated: October 25, 2026. Effective immediately for Adaptive LLM Fine-Tuning Studio.
        </p>
      </div>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-studio-text flex items-center gap-2">
          <Lock className="w-4 h-4 text-studio-teal" />
          1. Local-First Processing and Data Sovereignty
        </h2>
        <p className="text-studio-muted leading-relaxed">
          Adaptive LLM Fine-Tuning Studio is designed with a local-first architectural model. Your training datasets, model weights, optimizer checkpoints, and hyperparameters remain strictly on your host machine or designated compute cluster. No training tokens, input prompts, or model outputs are transmitted to external cloud services or telemetry endpoints without your explicit configuration.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-studio-text flex items-center gap-2">
          <Database className="w-4 h-4 text-studio-teal" />
          2. API Keys and Credentials
        </h2>
        <p className="text-studio-muted leading-relaxed">
          Authentication credentials (such as Hugging Face Hub tokens or cloud cluster keys) provided via environment variables (e.g., HF_TOKEN) are read directly by the local Python process. These credentials are never logged, never persisted in plain text in browser storage, and never transmitted to any third-party infrastructure.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-studio-text flex items-center gap-2">
          <FileText className="w-4 h-4 text-studio-teal" />
          3. Analytics and Tracking
        </h2>
        <p className="text-studio-muted leading-relaxed">
          This software does not employ third-party behavioral analytics, tracking pixels, session replay software, or fingerprinting scripts. Local system metrics (Python version, CUDA availability, GPU memory) reported by the backend /api/system/status endpoint are computed purely on-demand for local resource allocation and are never stored externally.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-studio-text">4. Security of Storage</h2>
        <p className="text-studio-muted leading-relaxed">
          All configuration files exported via the Studio (YAML/JSON presets) are generated directly in memory inside your browser and downloaded to your local file system. You maintain full ownership and control over all saved configuration artifacts.
        </p>
      </section>

      <div className="pt-6 border-t border-studio-border/50 text-[11px] text-studio-muted">
        For technical inquiries or data security audits, refer to the documentation and source repository.
      </div>
    </div>
  );
};
