from neuro_helper.entity import Space


def task_order(space: Space, with_rest=True):
    if space in [Space.K59, Space.K59_CORTEX]:
        if with_rest:
            out = ["REST", ]
        else:
            out = []
        return out + ["MOVIE", "RET"]
    elif space in [Space.K32, Space.K32_CORTEX]:
        return ["REST", ]

    raise ValueError(f"task order for {space} is not defined")
