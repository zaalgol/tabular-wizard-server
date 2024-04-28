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
    # ensemble: str = field(default='multi')
    # training_speed: str = field(default=None)
    training_strategy: str = field(default=None)
    sampling_strategy: str = field(default=None)
    target_column: str = field(default=None)
    created_at: datetime = field(default=None)
    columns: list[str] = field(default=None)
    encoding_rules: dict[str, list[str]] = field(default_factory=dict)
    metric: str = field(default=None)
    evaluations: str = field(default=None)
	