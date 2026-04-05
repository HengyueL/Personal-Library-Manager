---
url: https://www.anthropic.com/engineering/harness-design-long-running-apps
---

# Summary of Anthropic_Harness_Design.md

**1. Main Topic / Purpose**  
The post describes how Anthropic’s engineering team improved Claude’s ability to design high‑quality front‑ends and to build complete, functional full‑stack applications autonomously. It focuses on *harness design*—the orchestration of multiple specialized agents (planner, generator, evaluator) and context‑management techniques that enable long‑running, multi‑hour coding sessions without human intervention.

**2. Key Points and Findings**  

| Area | Problem Identified | Solution / Design Choice | Results / Evidence |
|------|-------------------|--------------------------|--------------------|
| **Context handling** | “Context anxiety” and loss of coherence as the model’s token window fills. | Use *context resets* with structured hand‑off artifacts (later dropped for Opus 4.6, which handles longer contexts). | Resets eliminated premature wrap‑up; Opus 4.5 needed resets, Opus 4.6 did not. |
| **Self‑evaluation bias** | Agents over‑rate their own output, especially for subjective tasks (design). | Introduce a separate *evaluator* agent (GAN‑style generator ↔ evaluator). Provide explicit grading criteria. | Evaluator gave more critical feedback; generator iterated toward higher‑quality designs. |
| **Frontend design** | Generator defaulted to safe, bland layouts. | Four‑criterion rubric (Design quality, Originality, Craft, Functionality) weighted toward design/originality; evaluator runs Playwright to interact with the live page. | 5–15 iterative cycles (≈4 h) produced noticeably more distinctive, sometimes radically creative UIs (e.g., 3‑D CSS museum site). |
| **Full‑stack coding** | Solo runs produced incomplete or buggy apps; long‑running harness was expensive and complex. | Three‑agent architecture: <br>• *Planner* expands a short prompt into a detailed spec. <br>• *Generator* works in “sprints”, implements one feature at a time, writes code, self‑evaluates. <br>• *Evaluator* uses Playwright to run end‑to‑end QA, grades against a sprint contract, and supplies concrete bug reports. <br>Communication via files; contracts negotiate “done” criteria before coding. | Compared to a solo run (20 min, $9), the full harness (6 h, $200) delivered a richer spec (16 features), better UI polish, functional AI‑assisted tools, and far fewer critical bugs. |
| **Cost/Performance trade‑offs** | Full harness heavy on token usage; many components may become unnecessary as models improve. | Systematically removed components (e.g., sprint decomposition, evaluator) after Opus 4.6 release; kept only load‑bearing pieces. | With Opus 4.6 the sprint construct could be dropped, evaluator moved to end‑of‑run only when needed, cutting overhead while preserving quality. |
| **Case studies** | *Retro game maker* and *digital audio workstation (DAW)* built from one‑sentence prompts. | Planner generated a 16‑item spec; generator produced React/Vite/FastAPI stack; evaluator caught functional gaps (missing UI actions, stubbed features). | Solo run failed to run; harness run produced a usable editor, playable game, and a functional albeit limited DAW. QA identified missing core interactions (e.g., drag‑to‑move clips). |
| **Iterative tuning** | Evaluator initially too lenient, sometimes “talked itself out of” bugs. | Fine‑tune evaluator prompts using few‑shot examples and log‑based error analysis; iterate until evaluator reliably flags failures. | Evaluator began rejecting sprint contracts that fell below hard thresholds, providing detailed, actionable feedback. |

**3. Conclusions / Recommendations**  

1. **Separate Generation and Evaluation** – A dedicated evaluator (even if also an LLM) is far easier to steer toward a skeptical stance than trying to make the generator self‑critical.  
2. **Explicit Grading Criteria** – Turning subjective notions (e.g., “beautiful design”) into concrete rubric items enables consistent feedback loops and drives the generator toward higher originality and visual quality.  
3. **Decompose Long Tasks** – Breaking a product build into tractable chunks (sprints) and negotiating “done” contracts before coding keeps the model focused and mitigates context‑window limits.  
4. **Context Resets vs. Compaction** – For models that exhibit context anxiety (e.g., Claude Sonnet 4.5), a hard reset with a