# Non-Negotiable Design, Quality and Release Rules

Build a serious, production-oriented developer tool. Do not use generic AI-product aesthetics.

## Visual Restrictions
- Never use purple gradients or decorative gradient backgrounds.
- Never use pill-shaped buttons. Prefer compact rectangular buttons with restrained corner radii.
- Never use emoji as icons. Use consistent, accessible SVG or Lucide icons.
- Never add fake reviews, testimonials, customer counts, usage statistics, benchmark results or performance claims.
- Never use vague hero copy, filler text, generic AI-generated marketing copy or stock-style AI imagery.
- Never add cursor animations, excessive scroll animations, unnecessary motion or decorative effects that reduce usability.
- Do not add "Made with AI" or AI-builder branding to the finished product, subject to applicable platform and license requirements.
- Prefer real interface screenshots, diagrams, code examples and verified system data over decorative images.

## Product Integrity
- Every visible control must work or be clearly disabled with an explanation.
- Never fabricate GPU information, training status, progress, model availability, memory estimates or benchmark results.
- Distinguish measured values, estimates, placeholders and unavailable information.
- Validate user input and provide actionable errors.
- Never expose Hugging Face tokens, API keys, environment secrets or sensitive file contents in logs or frontend responses.
- Keep the UI accessible, responsive and consistent.

## Testing and Change Management
- Work phase by phase.
- Inspect existing code before changing it.
- Write tests for each new feature.
- Run the relevant tests and then the full regression suite.
- Fix the underlying cause of every failure. Never delete, skip or weaken tests to make the suite pass.
- Verify production builds and security-sensitive flows where applicable.
- Report actual command results. Never claim a test or deployment succeeded without evidence.

## Mandatory Launch Gates
Do not declare the product production-ready or launch it until all applicable requirements are verified:
1. A custom domain is connected and DNS resolves correctly.
2. HTTPS is working.
3. An original favicon is installed and verified.
4. AI-builder attribution has been removed where permitted.
5. A privacy policy page is published and matches actual data practices.
6. A terms and conditions page is published and appropriate to the product.
7. Production build and automated tests pass.
8. Security, file upload, credential handling and error states are reviewed.
9. Critical user flows have been smoke-tested on the deployed domain.

If a requirement depends on external access, credentials, domain ownership or a human decision, identify the blocker. Do not pretend to complete it.

These rules apply to every phase, including UI scaffolding, training functionality, documentation and deployment.
