"""rhgym: a small RL environment for studying reward hacking in code-writing language models."""

from .detect import GAMING_LABELS, HACK_LABELS, LABELS, HackDetector, static_flags, uses_visible_literals
from .graders import RobustGrader, VulnerableGrader
from .prompts import chat_prompt, extract_code, user_prompt
from .tasks import TASKS, TASKS_BY_ID, Task, split_tasks

__all__ = [
    "GAMING_LABELS", "HACK_LABELS", "LABELS", "HackDetector", "static_flags", "uses_visible_literals",
    "RobustGrader", "VulnerableGrader",
    "chat_prompt", "extract_code", "user_prompt",
    "TASKS", "TASKS_BY_ID", "Task", "split_tasks",
]
__version__ = "0.2.0"
