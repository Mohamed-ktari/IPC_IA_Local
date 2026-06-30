# proofreading_agent.py
#
# Feature 3 — Relecture et amélioration de la qualité
#
# Flow:
#   1. Receives main document text + optional comparison documents
#   2. Receives a proofreading type (full | redaction | coherence | factual)
#   3. First LLM call: analysis → structured JSON report with all issues found
#   4. Second LLM call: generate corrected version with diff metadata
#   5. Returns ProofreadingResult with both the report and the corrected document
#
# The caller (API route) is responsible for:
#   - Extracting text from uploaded files (via documents/parsers/)
#   - Sending the result back to the frontend
#
# Two-pass design rationale:
#   Pass 1 (json_mode=True, low temp): deterministic issue detection
#   Pass 2 (json_mode=True, low temp): apply only accepted/all issues to produce corrected doc
#   Keeping them separate means the user can review before corrections are applied.

import json
from dataclasses import dataclass, field
from enum import Enum
from app.agents.base_agent import BaseAgent, AgentResponse


# ──────────────────────────────────────────────
# Domain types
# ──────────────────────────────────────────────

class ProofreadingType(str, Enum):
    FULL        = "full"         # all checks
    REDACTION   = "redaction"    # writing quality only
    COHERENCE   = "coherence"    # internal + external consistency only
    FACTUAL     = "factual"      # factual / technical errors only


class IssueType(str, Enum):
    ERREUR_REDACTION     = "ERREUR_REDACTION"
    ERREUR_FACTUELLE     = "ERREUR_FACTUELLE"
    REDONDANCE           = "REDONDANCE"
    INCOHERENCE_INTERNE  = "INCOHERENCE_INTERNE"
    INCOHERENCE_EXTERNE  = "INCOHERENCE_EXTERNE"
    SUGGESTION           = "SUGGESTION"


class Severity(str, Enum):
    CRITIQUE = "CRITIQUE"
    MAJEUR   = "MAJEUR"
    MINEUR   = "MINEUR"


@dataclass
class Issue:
    id: str                        # e.g. "issue_001" — stable ID for accept/reject
    type: IssueType
    severity: Severity
    original_excerpt: str          # exact text from the source document
    explanation: str               # why this is an issue
    suggestion: str                # proposed replacement or fix
    source_document: str           # "main" or the name of the comparison doc
    accepted: bool | None = None   # None = pending, True = accepted, False = rejected


@dataclass
class ProofreadingReport:
    document_name: str
    proofreading_type: ProofreadingType
    issues: list[Issue] = field(default_factory=list)
    summary: str = ""              # one-paragraph human-readable summary

    # Convenience filters the API route can use
    def by_type(self, issue_type: IssueType) -> list[Issue]:
        return [i for i in self.issues if i.type == issue_type]

    def by_severity(self, severity: Severity) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]

    def critical_count(self) -> int:
        return len(self.by_severity(Severity.CRITIQUE))


@dataclass
class CorrectedDocument:
    content: str                   # full corrected text
    applied_issue_ids: list[str]   # which issue IDs were applied
    change_log: list[dict]         # [{id, original, correction}] for traceability


@dataclass
class ProofreadingResult:
    report: ProofreadingReport
    corrected_document: CorrectedDocument | None  # None until user triggers correction pass


# ──────────────────────────────────────────────
# Prompt builders — kept here, not in the .txt,
# because they carry dynamic runtime context
# ──────────────────────────────────────────────

def _build_analysis_prompt(
    document_text: str,
    comparison_texts: dict[str, str],
    proofreading_type: ProofreadingType,
    user_instructions: str | None = None,
) -> str:
    focus_map = {
        ProofreadingType.FULL:      "all issue types: ERREUR_REDACTION, ERREUR_FACTUELLE, REDONDANCE, INCOHERENCE_INTERNE, INCOHERENCE_EXTERNE, SUGGESTION",
        ProofreadingType.REDACTION: "only ERREUR_REDACTION and REDONDANCE",
        ProofreadingType.COHERENCE: "only INCOHERENCE_INTERNE and INCOHERENCE_EXTERNE",
        ProofreadingType.FACTUAL:   "only ERREUR_FACTUELLE",
    }

    # Block 1 — comparison documents (empty string if none provided)
    comparison_block = ""
    if comparison_texts:
        parts = []
        for name, text in comparison_texts.items():
            parts.append(f"--- DOCUMENT DE COMPARAISON : {name} ---\n{text}\n---")
        comparison_block = "\n\nDOCUMENTS DE RÉFÉRENCE POUR COMPARAISON :\n" + "\n\n".join(parts)

    # Block 2 — user instructions (empty string if none provided)
    instructions_block = ""
    if user_instructions:
        instructions_block = f"\n\nSPECIFIC INSTRUCTIONS FROM THE USER:\n{user_instructions.strip()}"

    return f"""Analyse the following document for quality issues.

Focus on: {focus_map[proofreading_type]}

Return a single JSON object with this exact structure:
{{
  "summary": "<one paragraph overview of overall document quality>",
  "issues": [
    {{
      "id": "issue_001",
      "type": "<IssueType>",
      "severity": "<CRITIQUE|MAJEUR|MINEUR>",
      "original_excerpt": "<exact text from document>",
      "explanation": "<why this is an issue>",
      "suggestion": "<concrete fix or improvement>",
      "source_document": "<main or comparison document name>"
    }}
  ]
}}

Issue IDs must be sequential: issue_001, issue_002, ...
If no issues are found for a category, simply omit those from the list.
Return only the JSON object. No text before or after.
{instructions_block}
DOCUMENT TO ANALYSE:
{document_text}
{comparison_block}"""

def _build_correction_prompt(
    document_text: str,
    issues_to_apply: list[Issue],
) -> str:
    # Gives the model the original text and only the accepted issues
    # Asks it to return the corrected text + a change log
    issues_json = json.dumps(
        [
            {
                "id": i.id,
                "original_excerpt": i.original_excerpt,
                "suggestion": i.suggestion,
            }
            for i in issues_to_apply
        ],
        ensure_ascii=False,
        indent=2,
    )

    return f"""Apply the following corrections to the document below.

Rules:
- Apply ONLY the corrections listed. Do not make any other changes.
- Preserve the original structure, formatting markers, and language.
- For each correction applied, record the original text and what replaced it.

Return a single JSON object:
{{
  "corrected_text": "<full corrected document>",
  "change_log": [
    {{
      "id": "<issue_id>",
      "original": "<text that was replaced>",
      "correction": "<text that replaced it>"
    }}
  ]
}}

Return only the JSON object. No text before or after.

CORRECTIONS TO APPLY:
{issues_json}

ORIGINAL DOCUMENT:
{document_text}"""


# ──────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────

class ProofreadingAgent(BaseAgent):

    agent_type  = "proofreading"
    description = "Relit et améliore la qualité d'un document : erreurs, redondances, incohérences, suggestions."

    # ── Pass 1: analyse ────────────────────────

    def analyse(
        self,
        document_text: str,
        document_name: str = "document",
        comparison_texts: dict[str, str] | None = None,
        proofreading_type: ProofreadingType = ProofreadingType.FULL,
        user_instructions: str | None = None,
    ) -> ProofreadingReport:
        """
        Runs the analysis pass. Returns a ProofreadingReport the caller can
        display to the user before any corrections are applied.
        """
        prompt = _build_analysis_prompt(
            document_text,
            comparison_texts or {},
            proofreading_type,
        )

        response: AgentResponse = self.chat(
            user_message=prompt,
            temperature=0.1,     # low: we want deterministic, not creative
            max_tokens=4096,
            json_mode=True,
        )

        return self._parse_analysis_response(
            response.content,
            document_name,
            proofreading_type,
        )

    def _parse_analysis_response(
        self,
        raw: str,
        document_name: str,
        proofreading_type: ProofreadingType,
    ) -> ProofreadingReport:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"ProofreadingAgent: LLM returned invalid JSON — {e}\nRaw: {raw[:300]}")

        issues = []
        for item in data.get("issues", []):
            try:
                issues.append(Issue(
                    id               = item["id"],
                    type             = IssueType(item["type"]),
                    severity         = Severity(item["severity"]),
                    original_excerpt = item["original_excerpt"],
                    explanation      = item["explanation"],
                    suggestion       = item["suggestion"],
                    source_document  = item.get("source_document", "main"),
                ))
            except (KeyError, ValueError) as e:
                # Skip malformed issues rather than crashing — log and continue
                # In production you'd want to surface this as a warning
                continue

        return ProofreadingReport(
            document_name    = document_name,
            proofreading_type= proofreading_type,
            issues           = issues,
            summary          = data.get("summary", ""),
        )

    # ── Pass 2: correction ─────────────────────

    def apply_corrections(
        self,
        document_text: str,
        report: ProofreadingReport,
        accepted_ids: list[str] | None = None,
    ) -> CorrectedDocument:
        """
        Applies corrections to the document.

        accepted_ids:
          - None  → apply ALL issues in the report (user accepted everything)
          - []    → apply nothing (edge case, but handle gracefully)
          - [ids] → apply only the listed issue IDs
        """
        if accepted_ids is not None:
            issues_to_apply = [i for i in report.issues if i.id in accepted_ids]
        else:
            issues_to_apply = report.issues

        if not issues_to_apply:
            # Nothing to apply — return original unchanged
            return CorrectedDocument(
                content            = document_text,
                applied_issue_ids  = [],
                change_log         = [],
            )

        prompt = _build_correction_prompt(document_text, issues_to_apply)

        response: AgentResponse = self.chat(
            user_message=prompt,
            temperature=0.1,
            max_tokens=8192,    # correction pass needs more tokens — full document rewrite
            json_mode=True,
        )

        return self._parse_correction_response(response.content, issues_to_apply)

    def _parse_correction_response(
        self,
        raw: str,
        applied_issues: list[Issue],
    ) -> CorrectedDocument:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"ProofreadingAgent: correction pass returned invalid JSON — {e}\nRaw: {raw[:300]}")

        return CorrectedDocument(
            content           = data.get("corrected_text", ""),
            applied_issue_ids = [i.id for i in applied_issues],
            change_log        = data.get("change_log", []),
        )

    # ── run(): convenience wrapper ─────────────
    # Satisfies BaseAgent's abstract method.
    # One-shot: analyse + apply all corrections immediately.
    # The API route calls analyse() and apply_corrections() separately
    # when the user reviews issues in between — use this only for batch/testing.

    def run(
        self,
        document_text: str,
        document_name: str = "document",
        comparison_texts: dict[str, str] | None = None,
        proofreading_type: ProofreadingType = ProofreadingType.FULL,
    ) -> ProofreadingResult:
        report = self.analyse(
            document_text,
            document_name,
            comparison_texts,
            proofreading_type,
        )
        corrected = self.apply_corrections(document_text, report)
        return ProofreadingResult(report=report, corrected_document=corrected)