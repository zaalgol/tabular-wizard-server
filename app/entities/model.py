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
    training_speed: str = field(default=None)
    target_column: str = field(default=None)
    created_at: datetime = field(default=None)
    columns: list[str] = field(default=None)
	