from typing import List


def calculate_class_overlapping(classes1: List[str], classes2: List[str]) -> List[bool]:
    words1 = [word for item in classes1 for word in item.split(',')]
    results = []
    for item in classes2:
        flag: bool = False
        for word in item.split(','):
            if word in words1:
                flag = True
                break
        results.append(flag)
    return results
