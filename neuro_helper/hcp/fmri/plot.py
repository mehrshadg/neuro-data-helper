from neuro_helper.abstract.map import Space

__all__ = ["task_colors"]


def task_colors(space: Space, with_rest=False):
    if space in [Space.K59_CORTEX, Space.K59]:
        colors = ["#f032e6", "#aaffc3"]
        if with_rest:
            return ["#B89B49"] + colors
        else:
            return colors

    raise ValueError(f"no task color is defined for {space}")
