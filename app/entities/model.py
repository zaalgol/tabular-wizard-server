from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Model:
    id: str = field(default=None)
    user_id: str = field(default=None)
    model_name: str = field(default=None)
    file_name: str = field(default=None)
    description: str = field(default=None)
    model_type: str = field(default=None)
    ensemble: bool = field(default=False)
    training_speed: str = field(default=None)
    target_column: str = field(default=None)
    created_at: datetime = field(default=None)
    columns: list[str] = field(default=None)
    encoding_rules: dict[str, list[str]] = field(default_factory=dict)
    metric: str = field(default=None)
	