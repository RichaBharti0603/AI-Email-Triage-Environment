from typing import Dict, List, Any
from enum import Enum
class TaskType(Enum):
    CLASSIFICATION = "classification"
    PRIORITIZATION = "prioritization"
    ROUTING = "routing"
class EmailTask:
    """Define tasks for the email triage system"""
    def __init__(self, task_id: str, task_type: TaskType, description: str):
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
    def evaluate(self, agent_output: Any, ground_truth: Any) -> Dict[str, Any]:
        """Evaluate agent performance on this task"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'correct': agent_output == ground_truth,
            'agent_output': agent_output,
            'ground_truth': ground_truth
        }
class TaskManager:
    """Manage and generate tasks for the environment"""
    def __init__(self):
        self.tasks = self._create_tasks()
        self.task_results = []
    def _create_tasks(self) -> List[EmailTask]:
        """Create a set of tasks"""
        return [
            EmailTask(
                "task_1",
                TaskType.CLASSIFICATION,
                "Classify email as urgent, normal, or spam"
            ),
            EmailTask(
                "task_2",
                TaskType.PRIORITIZATION,
                "Prioritize email for handling"
            ),
            EmailTask(
                "task_3",
                TaskType.ROUTING,
                "Route email to appropriate department"
            )
        ]
    def get_task(self, task_id: str) -> EmailTask:
        """Get a specific task"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    def evaluate_task(self, task_id: str, agent_output: Any, ground_truth: Any) -> Dict:
        """Evaluate a specific task"""
        task = self.get_task(task_id)
        if task:
            result = task.evaluate(agent_output, ground_truth)
            self.task_results.append(result)
            return result
        return {'error': 'Task not found'}
    def get_performance_summary(self) -> Dict:
        """Get summary of task performance"""
        if not self.task_results:
            return {'total_tasks': 0, 'success_rate': 0}
        correct = sum(1 for r in self.task_results if r['correct'])
        total = len(self.task_results)
        return {
            'total_tasks': total,
            'correct': correct,
            'failed': total - correct,
            'success_rate': correct / total if total > 0 else 0
        }