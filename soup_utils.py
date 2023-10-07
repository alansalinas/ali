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
