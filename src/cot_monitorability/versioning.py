from inspect_ai import Task


def stamp_version(task: Task) -> Task:
    task.version = "4-A"
    return task
