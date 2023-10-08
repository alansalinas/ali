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


def overlap(a, b):
    tokens_a = set(range(a.start_token, a.end_token))
    tokens_b = set(range(b.start_token, b.end_token))
    return len(tokens_a.intersection(tokens_b))


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