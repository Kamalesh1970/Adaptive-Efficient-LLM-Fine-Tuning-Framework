import React from 'react';
import { ArrowLeft, Scale, Terminal, AlertTriangle, ShieldCheck } from 'lucide-react';

interface TermsConditionsProps {
  onBack: () => void;
}

export const TermsConditions: React.FC<TermsConditionsProps> = ({ onBack }) => {
  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-10 max-w-4xl mx-auto space-y-8 text-xs text-studio-text" data-testid="terms-conditions-page">
      <div className="border-b border-studio-border/70 pb-4">
        <button
          onClick={onBack}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-studio-muted hover:text-studio-accent mb-4 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Studio</span>
        </button>
        <div className="flex items-center gap-2.5">
          <Scale className="w-5 h-5 text-studio-accent" />
          <h1 className="text-xl font-bold tracking-tight text-studio-text">Terms and Conditions</h1>
        </div>
        <p className="text-studio-muted text-xs mt-1">
          Last updated: October 25, 2026. Standard terms governing the use of Adaptive LLM Fine-Tuning Studio.
        </p>
      </div>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-studio-text flex items-center gap-2">
          <Terminal className="w-4 h-4 text-studio-teal" />
          1. License and Permitted Use
        </h2>
        <p className="text-studio-muted leading-relaxed">
          Adaptive LLM Fine-Tuning Studio is an engineering platform developed for configuring and running parameter-efficient fine-tuning workflows on Large Language Models. You are permitted to use this software in research, internal development, and commercial model optimization, provided you comply with applicable foundation model licenses and dataset terms.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-studio-text flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-studio-teal" />
          2. Foundation Model and Dataset Compliance
        </h2>
        <p className="text-studio-muted leading-relaxed">
          Users are solely responsible for ensuring that any base model weights (such as Llama, Mistral, or Qwen) and datasets selected for fine-tuning comply with the respective author licenses, usage policies, and intellectual property requirements. Adaptive LLM Fine-Tuning Studio provides orchestration interfaces and does not warrant the licensing rights of third-party models or datasets.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-studio-text flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-studio-teal" />
          3. Disclaimer of Hardware and Output Warranties
        </h2>
        <p className="text-studio-muted leading-relaxed">
          Fine-tuning large neural networks requires substantial compute and power resources. The software is provided "as is", without warranty of any kind, express or implied. Under no circumstances shall the authors be liable for hardware thermal throttling, GPU damage resulting from improper overclocking or cooling, Out-Of-Memory job terminations, or unexpected model generations.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-studio-text">4. Modifications and Source Code</h2>
        <p className="text-studio-muted leading-relaxed">
          You may modify, extend, and integrate this software into your infrastructure in accordance with the project repository license. Contributions and modifications must maintain accurate attribution and preservation of licensing terms.
        </p>
      </section>

      <div className="pt-6 border-t border-studio-border/50 text-[11px] text-studio-muted">
        Questions regarding terms and licensing should be directed to the repository maintainers.
      </div>
    </div>
  );
};
