import pickle
from collections import defaultdict
from itertools import product

import numpy as np
from sentence_transformers.util import cos_sim

from embedder import Embedder
from typedefs import Entity
from soup_utils import deduplicate_by_overlap


class EntityIndex:
    def __init__(
        self,
        embedder,
        entities=None,
        parents=None,
        children=None,
        names=None,
        vectors=None,
        name_entities=None,
        entity_names=None,
        metadata=None,
    ):
        self.embedder = embedder
        self.parents = parents
        self.children = children
        self.names = names
        self.vectors = vectors
        self.name_entities = name_entities
        self.entity_names = entity_names
        self.metadata = metadata
        # If building from scratch
        if entities:
            self.build_entity_tree(entities)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        embedder = Embedder(data['embedder_path'])
        ei = cls(
            embedder,
            None,
            data['parents'],
            data['children'],
            data['names'],
            data['vectors'],
            data['name_entities'],
            data['entity_names'],
            data['metadata'],
        )
        return ei

    def build_entity_tree(self, entities):
        """
        Given all the entities build the hierarchy between them
        """

        def crawl_tree(entities, cid):
            pid = cid
            self.children.append([])
            for e in entities:
                cid += 1
                self.children[pid].append(cid)
                self.parents.append(pid)

                names = []
                if e.get('name'):
                    names += [name.strip() for name in e['name'].split(',')]
                names += [
                    name.text.strip() for name in e.find_all('name', recursive=False)
                ]
                self.names.update(names)

                for name in names:
                    self.name_entities[name].append(cid)
                    self.entity_names[cid].append(name)

                self.metadata.append(
                    {'children_only': bool(e.get('children-only'))}
                )

                sub_entities = e.find_all('entity', recursive=False)
                if len(sub_entities) > 0:
                    cid = crawl_tree(sub_entities, cid)
            return cid

        # Initializations need to consider a master parent node 0 with no other parents for
        # the recursion algorithm, this way our first level entities are children of 0 and
        # entities can be addressed by indexes starting from 0 for the master parent
        self.parents = [0]
        self.children = []
        self.names = set()
        self.entity_names = defaultdict(list)
        self.name_entities = defaultdict(list)
        self.metadata = [{}]
        crawl_tree(entities, 0)
        # Create vector index
        self.names = list(self.names)
        self.vectors = np.array([self.embedder.encode(e) for e in self.names])
        return True

    def expand_topics_with_entitites(self, topics):
        """
        Get a list of topics and expand the entities within them with all
        entity names and children names
        """
        expanded_topics = []
        for topic in topics:
            entities = topic.find_all('entity', recursive=False)
            # per entity get a list of its children names
            names = [
                self.get_expanded_names(
                    e := self.search(node.text)[0],
                    (node.get('children-only') is not None) or e.children_only,
                )
                for node in entities
            ]
            # replace every name combination for all entity nodes
            entity_tuples = product(*names)
            for tup in entity_tuples:
                for i, node in enumerate(entities):
                    node.string = ' ' + tup[i] + ' '
                expanded_topics.append(' '.join(topic.text.split()))
        return expanded_topics

    def get_expanded_names(self, entity, children_only=False):
        """
        Get names of all children entities and if True also the current entity names
        """
        names = []
        if entity.id < len(self.children):
            names += [
                name
                for child_id in self.children[entity.id]
                for name in self.entity_names[child_id]
            ]
        if not children_only:
            names += self.entity_names[entity.id]
        return names

    def tag(self, text):
        tags = self.scan(text)
        tokens = text.split()
        output = []
        current = 0
        for t in tags:
            output += tokens[current:t.start_token]
            output.append(
                f'<entity id={t.id}, confidence={t.confidence}>{t.text}</entity>'
            )
            current = t.end_token
        output += tokens[current:]
        return ' '.join(output)

    def scan(self, text, n_tokens=20):
        tokens = text.split()
        chunks = [
            [
                Entity(
                    id=e.id,
                    name=e.name,
                    text=e.text,
                    confidence=e.confidence,
                    start_token=i,
                    end_token=i+j,
                )
                for e in self.search(' '.join(tokens[i:i+j]))
            ][0]
            for i in range(len(tokens))
            for j in range(1, min(n_tokens, len(tokens[i:])) + 1)
        ]

        chunks = [c for c in chunks if c.confidence >= 0.7]
        return deduplicate_by_overlap(chunks)

    def search(self, query, k=1):
        q_vec = np.array([self.embedder.encode(query.strip())])
        cs = cos_sim(q_vec, self.vectors)[0].numpy()
        top_k_idx = np.argpartition(cs, -k)[-k:]
        # TODO: Disambiguate entities, returning only the first associated to name
        names = [self.names[i] for i in top_k_idx]
        entity_idxes = [self.name_entities[name][0] for name in names]
        results = [
            Entity(
                id=entity_idx,
                name=self.names[name_idx],
                text=query,
                confidence=cs[name_idx],
                children_only=self.metadata[entity_idx]['children_only'],
            )
            for name_idx, entity_idx in zip(top_k_idx, entity_idxes)
        ]
        return sorted(results, key=lambda x: x.confidence, reverse=True)

    def save(self, filename):
        data = {
            'embedder_path': self.embedder.model_path,
            'parents': self.parents,
            'children': self.children,
            'names': self.names,
            'vectors': self.vectors,
            'name_entities': self.name_entities,
            'entity_names': self.entity_names,
            'metadata': self.metadata,
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
