def task_order(with_rest=True):
    if with_rest:
        out = ["Rest"]
    else:
        out = []
    return out + ["StoryM", "Motort", "Wrkmem"]