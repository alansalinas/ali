import pathlib
from bs4 import BeautifulSoup


def read_documents(path):
    path = pathlib.Path(path)
    files = path.rglob('*.xml')
    documents = []
    entities = []
    for filename in sorted(files):
        with open(filename, 'r', encoding='UTF-8') as file:
            data = file.read()
            soup = BeautifulSoup(data, 'lxml')
            documents += soup.html.body.find_all('document')
            entities += soup.html.body.find_all('entity', recursive=False)
    return documents, entities


def is_within(a, b, reverse=False):
    if a.start_token >= b.start_token and a.end_token <= b.end_token:
        return True
    if reverse and b.start_token >= a.start_token and b.end_token <= a.end_token:
        return True
    return False


def overlap(a, b):
    if a.end_token >= b.start_token:
        return 0
    tokens_a = set(range(a.start_token, a.end_token))
    tokens_b = set(range(b.start_token, b.end_token))
    return len(tokens_a.intersection(tokens_b))


def iou(a, b):
    if a.end_token >= b.start_token:
        return 0
    tokens_a = set(range(a.start_token, a.end_token))
    tokens_b = set(range(b.start_token, b.end_token))
    return len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b))


def deduplicate_by_overlap(grp):
    remaining = []
    a = grp[0]
    for b in grp[1:]:
        if overlap(a, b) > 0:
            a = max(a, b, key=lambda x: x.confidence)
        else:
            remaining.append(b)

    if len(remaining) < 2:
        return [a] + remaining

    return [a] + deduplicate_by_overlap(remaining)
