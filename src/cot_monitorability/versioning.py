from inspect_ai import Task


def stamp_version(task: Task) -> Task:
    task.version = "3-A"
    return task
