def task_colors(with_rest=False):
    colors = ["#f032e6", "#aaffc3", "#637687"]
    if with_rest:
        return ["#B89B49"] + task_colors
    else:
        return colors
