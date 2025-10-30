from dataclasses import dataclass, field
from datetime import datetime
from project_classes.project_outline import ProjectOutline


@dataclass
class ProjectMetadata:
    project_outline: ProjectOutline
    independent_variables: dict[str, list[str]]
    metadata: dict[str, dict[str, dict[str, str]]]
    created_at: datetime = field(default_factory=datetime.now)

