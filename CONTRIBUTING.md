# Contributing to AtmaNirbhar AI

Thank you for your interest in contributing to **AtmaNirbhar AI** — India's premier open-source multi-agent autonomous driving intelligence architecture for unstructured roadways.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all contributors. Please treat fellow contributors with respect, intellectual rigor, and constructive communication.

## Development Workflow

1. **Fork & Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Backend Development**:
   - Python 3.10+ recommended.
   - Install dependencies: `pip install -r backend/requirements.txt`
   - Run linter & tests before submitting PRs.
   - Follow PEP 8 and strong type hints (`pydantic` schemas).
3. **Frontend Development**:
   - Node.js 18+ and Next.js 16 (App Router).
   - Install dependencies: `cd frontend && npm install`
   - Ensure `npm run build` passes with zero TypeScript warnings or errors.
4. **Submitting Changes**:
   - Write clear, atomic commit messages following conventional commits (`feat:`, `fix:`, `docs:`, `perf:`).
   - Ensure architecture diagrams or schema updates are reflected in `docs/` and `README.md`.
