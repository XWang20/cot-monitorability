from inspect_ai import Task


def stamp_version(task: Task) -> Task:
    task.version = "1-A"
    return task
