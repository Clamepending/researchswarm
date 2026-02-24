from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

from .models import NotebookEntry, TimelineEvent
from .storage import add_notebook_entry, add_timeline_event


@dataclass
class SpecialistOutput:
    agent: str
    summary: str
    confidence: float


class MockSpecialist:
    def __init__(self, name: str, template: str) -> None:
        self.name = name
        self.template = template

    def execute(self, objective: str) -> SpecialistOutput:
        return SpecialistOutput(
            agent=self.name,
            summary=self.template.format(objective=objective),
            confidence=0.68,
        )


class OrchestratorStateMachine:
    def __init__(self) -> None:
        self.specialists = [
            MockSpecialist("literature", "Collected relevant priors for '{objective}'."),
            MockSpecialist("experiment_designer", "Drafted an experiment matrix targeting '{objective}'."),
            MockSpecialist("evaluator", "Prepared evaluation rubric and significance checks for '{objective}'."),
        ]

    def run_discovery(self, run_id: UUID, objective: str) -> Iterable[SpecialistOutput]:
        add_timeline_event(
            TimelineEvent(run_id=run_id, event_type="stage_transition", message="Entering discovery phase", confidence=0.9)
        )

        for specialist in self.specialists:
            output = specialist.execute(objective)
            add_timeline_event(
                TimelineEvent(
                    run_id=run_id,
                    event_type="agent_note",
                    message=f"[{output.agent}] {output.summary}",
                    confidence=output.confidence,
                )
            )
            add_notebook_entry(
                NotebookEntry(
                    run_id=run_id,
                    author_agent=output.agent,
                    summary=output.summary,
                    decision="Continue to next specialist",
                    citations=[],
                )
            )
            yield output
