from inspect_ai import Task


def stamp_version(task: Task) -> Task:
    task.version = "2-A"
    return task
