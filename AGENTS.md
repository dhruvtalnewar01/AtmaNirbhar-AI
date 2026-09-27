# AtmaNirbhar AI Agent Rules

- Never display a hardcoded, mocked, or placeholder value as if it were real detection output. Label mock states explicitly.
- Never use an em dash or en dash in any user-facing text. Use a period, comma, or colon.
- Never use emoji as functional icons. Use Lucide or Heroicons SVGs only.
- Before building any UI screen, read design-system/MASTER.md (and the page override if one exists). Do not invent colors or fonts outside it.
- Motion status must be computed with ego-motion compensation (median displacement subtraction), never raw pixel movement.
- All distance values are estimates. Always label them as such in the UI.
- Prefer Gemini for boilerplate and scaffolding; switch to Claude for CV pipeline logic, risk-scoring, and anything touching the algorithms in the build prompt's Section 6.
- Stay on free-tier models, datasets, and hosting. Ask before introducing a paid dependency.
