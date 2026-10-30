import { describe, it, expect, beforeEach, vi } from 'vitest';
import { parse } from '../logic/parseDataset.js';
import { est, compat, alt, ok } from '../logic/compatRules.js';
import { pilot } from '../components/pilot.js';
import { NAV, TITLES } from '../data/nav.js';
import { MODELS, setCatalogModels } from '../data/models.js';
import { S } from '../state.js';
import { P } from '../pages/shared.js';
import '../pages/dashboard.js';
import '../pages/dataset.js';
import '../pages/models.js';
import '../pages/compat.js';
import '../pages/config.js';
import '../pages/runs.js';
import '../pages/eval.js';
import '../pages/artifacts.js';
import '../pages/settings.js';
import { api } from '../api.js';

describe('FineTunePilot Core Logic & UI Tests', () => {
  beforeEach(() => {
    // Reset state before each test
    Object.assign(S, {
      proj: 'Test Project',
      ds: null,
      model: null,
      hw: '16',
      method: 'lora',
      cfg: { epochs: 3, lr: '0.0002', batch: 8, rank: 16, len: 512 },
      saved: false,
      runs: [],
      arts: [],
      log: [],
      prog: 0,
      timer: null,
      hardwareInfo: null,
      backendOnline: true,
      tlog: [],
    });
  });

  describe('1. Navigation & Page Rendering', () => {
    it('contains all required sidebar navigation groups and page keys', () => {
      expect(NAV.length).toBeGreaterThan(3);
      expect(TITLES['dashboard']).toBe('Dashboard');
      expect(TITLES['dataset']).toBe('Dataset Inspector');
      expect(TITLES['models']).toBe('Model Hub');
      expect(TITLES['compat']).toBe('Compatibility Checker');
      expect(TITLES['config']).toBe('Training Configuration');
      expect(TITLES['runs']).toBe('Training Runs');
      expect(TITLES['eval']).toBe('Evaluation and Export');
      expect(TITLES['artifacts']).toBe('Artifact Library');
      expect(TITLES['settings']).toBe('Settings and Project Info');
    });

    it('renders all 9 pages to HTML without throwing exceptions', () => {
      const pageKeys = ['dashboard', 'dataset', 'models', 'compat', 'config', 'runs', 'eval', 'artifacts', 'settings'];
      pageKeys.forEach(key => {
        expect(typeof P[key]).toBe('function');
        const html = P[key]();
        expect(typeof html).toBe('string');
        expect(html.length).toBeGreaterThan(20);
      });
    });
  });

  describe('2. Dataset Parsing & Validation Engine', () => {
    it('parses CSV with quoted strings correctly', () => {
      const csv = 'instruction,response\n"Hello, world!","Greetings"\n"Question","Answer"';
      const parsed = parse('test.csv', csv);
      expect(parsed.name).toBe('test.csv');
      expect(parsed.rows.length).toBe(2);
      expect(parsed.cols).toEqual(['instruction', 'response']);
      expect(parsed.rows[0].instruction).toBe('Hello, world!');
      expect(parsed.rows[0].response).toBe('Greetings');
      expect(parsed.issues).toBe(0);
    });

    it('detects duplicate records and missing values in datasets', () => {
      const csv = 'instruction,response\n"Repeat","Answer"\n"Repeat","Answer"\n"Incomplete",""';
      const parsed = parse('flawed.csv', csv);
      expect(parsed.rows.length).toBe(3);
      expect(parsed.dup).toBe(1);
      expect(parsed.miss.find(m => m[0] === 'response')[1]).toBe(1);
      expect(parsed.issues).toBeGreaterThan(0);
    });

    it('parses JSONL format accurately', () => {
      const jsonl = '{"prompt": "A", "completion": "B"}\n{"prompt": "C", "completion": "D"}';
      const parsed = parse('data.jsonl', jsonl);
      expect(parsed.rows.length).toBe(2);
      expect(parsed.cols).toContain('prompt');
      expect(parsed.cols).toContain('completion');
      expect(parsed.issues).toBe(0);
    });
  });

  describe('3. Compatibility Evaluation Engine', () => {
    it('calculates memory estimates based on model size and method', () => {
      S.model = { id: 'm1', n: 'Demo 2B', p: 2.0, ctx: 4096 };
      S.method = 'lora';
      const estLora = est();
      expect(estLora).toBeCloseTo(2.0 * 2.4 + 1, 1);

      S.method = 'qlora';
      const estQlora = est();
      expect(estQlora).toBeCloseTo(2.0 * 0.9 + 1, 1);
      expect(estQlora).toBeLessThan(estLora);
    });

    it('flags incompatibility when sequence length exceeds model context', () => {
      S.ds = { name: 'test.csv', rows: [{}], issues: 0 };
      S.model = { id: 'm1', n: 'Demo 2B', p: 2.0, ctx: 2048 };
      S.hw = '16';
      S.cfg.len = 4096; // Exceeds 2048

      const report = compat();
      const seqCheck = report.find(r => r[1] === 'Sequence length');
      expect(seqCheck[0]).toBe('bad');
      expect(ok()).toBe(false);
    });

    it('passes compatibility when model, dataset, and hardware align', () => {
      S.ds = { name: 'test.csv', rows: [{}], issues: 0 };
      S.model = { id: 'm1', n: 'Demo 2B', p: 2.0, ctx: 4096 };
      S.hw = '16';
      S.cfg.len = 512;

      expect(ok()).toBe(true);
    });
  });

  describe('4. "Ask the Pilot" Telemetry Assistant', () => {
    it('handles status, hardware, and help commands', () => {
      S.hardwareInfo = {
        cuda_available: false,
        pytorch_version: '2.13.0+cu130',
        primary_gpu_name: null,
      };

      const helpResp = pilot('help');
      expect(helpResp).toContain('Commands:');

      const hwResp = pilot('hardware');
      expect(hwResp).toContain('CPU fallback mode');

      const statusResp = pilot('status');
      expect(statusResp).toContain('Project: Test Project');
    });

    it('explains fine-tuning terminology accurately', () => {
      const resp = pilot('explain lora');
      expect(resp).toContain('LoRA trains small add-on weights');
    });
  });

  describe('5. Centralized API Client & Backend Integration', () => {
    it('defines all required real backend API methods', () => {
      expect(typeof api.getHealth).toBe('function');
      expect(typeof api.getSystemStatus).toBe('function');
      expect(typeof api.getModels).toBe('function');
      expect(typeof api.validateModel).toBe('function');
      expect(typeof api.uploadDataset).toBe('function');
      expect(typeof api.uploadDatasetContent).toBe('function');
      expect(typeof api.getDatasets).toBe('function');
      expect(typeof api.validateConfig).toBe('function');
      expect(typeof api.createTrainingRun).toBe('function');
      expect(typeof api.getTrainingRuns).toBe('function');
      expect(typeof api.getTrainingRun).toBe('function');
      expect(typeof api.getTrainingLogs).toBe('function');
      expect(typeof api.cancelTrainingRun).toBe('function');
      expect(typeof api.getArtifacts).toBe('function');
      expect(typeof api.getEvaluations).toBe('function');
      expect(typeof api.createGGUFExport).toBe('function');
      expect(typeof api.getServingCapabilities).toBe('function');
      expect(typeof api.validateKubernetes).toBe('function');
    });

    it('synchronizes backend models into catalog', () => {
      const mockBackendModels = [
        {
          id: 'meta-llama/Meta-Llama-3-8B',
          name: 'Llama 3 8B',
          parameter_count: '8.0B',
          context_length: 8192,
          is_gated: false,
          supported_methods: ['lora', 'qlora'],
        },
      ];
      setCatalogModels(mockBackendModels);
      expect(MODELS.some(m => m.id === 'meta-llama/Meta-Llama-3-8B')).toBe(true);
    });

    it('manages training run lifecycle and artifacts state cleanly', () => {
      // Simulate run record addition
      const mockRun = {
        id: 'run-test-123',
        m: 'Gemma-2 2B',
        d: 'sample.csv',
        c: 'LORA, 3 epochs',
        done: true,
        status: 'completed',
      };
      S.runs.push(mockRun);
      expect(S.runs.length).toBe(1);
      expect(S.runs[0].status).toBe('completed');

      // Check eval page rendering with completed run
      const evalHtml = P.eval();
      expect(evalHtml).toContain('Active Checkpoint');
      expect(evalHtml).toContain('Export GGUF Binary');

      // Simulate artifact addition
      S.arts.push({
        id: 'art-123',
        n: 'LoRA Adapter',
        t: 'LORA',
        p: 'google/gemma-2-2b',
        s: 'valid',
        size: '120 KB',
        url: '/api/artifacts/art-123/download',
      });
      const artHtml = P.artifacts();
      expect(artHtml).toContain('LoRA Adapter');
      expect(artHtml).toContain('/api/artifacts/art-123/download');
    });
  });
});
