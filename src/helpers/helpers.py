


def _safe_ratio(a, b):
    if a is None or b is None:
        return None
    total = a + b
    return 0.5 if total == 0 else round(a / total, 4)