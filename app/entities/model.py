from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Model:
    id: str = field(default=None)
    user_id: str = field(default=None)
    model_name: str = field(default=None)
    file_name: str = field(default=None)
    file_line_num: int = field(default=0)
    description: str = field(default=None)
    model_type: str = field(default=None)
    is_multi_class: bool = field(default=False)
    # ensemble: str = field(default='multi')
    # training_speed: str = field(default=None)
    training_strategy: str = field(default=None)
    sampling_strategy: str = field(default=None)
    target_column: str = field(default=None)
    created_at: datetime = field(default=None)
    columns: list[str] = field(default=None)
    encoding_rules: dict[str, list[str]] = field(default_factory=dict)
    transformations: dict[str, dict] = field(default_factory=dict)
    metric: str = field(default=None)
    formated_evaluations: str = field(default=None)
    is_llm: bool = field(default=False)
    model_description_pdf_file_path: str = field(default=None)
    is_time_series: bool = field(default=False)
    time_series_code: str = field(default=None)
	