import pickle

import numpy as np
from sentence_transformers.util import cos_sim

from embedder import Embedder
from entity_index import EntityIndex
from soup_utils import read_documents
from typedefs import Topic


class SemanticIndex:
    def __init__(
        self,
        embedder_model_path,
        entity_index=None,
        topics_values=None,
        topics_vecs=None,
        topics_document=None,
    ):
        self.embedder = Embedder(embedder_model_path)
        self.entity_index = entity_index
        self.topics_values = topics_values
        self.topics_vecs = topics_vecs
        self.topics_document = topics_document

    @classmethod
    def load(cls, directory):
        with open(f"{directory}/semantic.index", 'rb') as file:
            data = pickle.load(file)
        embedder = Embedder(data['embedder_path'])
        ei = EntityIndex.load(f"{directory}/entity.index")
        si = cls(
            embedder.model_path,
            ei,
            data['topics_values'],
            data['topics_vecs'],
            data['topics_document'],
        )
        return si

    def build_index(
        self,
        documents_path,
    ):
        documents, entities = read_documents(documents_path)
        self.entity_index = EntityIndex(self.embedder, entities)
        self.build_topic_index(documents)

    def build_topic_index(self, documents):
        cid = 0
        self.topics_values = []
        self.topics_vecs = []
        self.topics_document = []

        for doc in documents:
            topics = doc.find_all('topic', recursive=False)
            topics = self.entity_index.expand_topics_with_entitites(topics)
            topics_values = [topic for topic in topics]
            self.topics_values += topics_values
            self.topics_vecs += [self.embedder.encode(topic) for topic in topics_values]
            self.topics_document += [cid for i in range(len(topics))]
            cid += 1

        self.topics_vecs = np.array(self.topics_vecs)

    def tag(self, text):
        tags = self.scan(text)
        tokens = text.split()
        output = []
        current = 0
        for t in tags:
            output += tokens[current:t.start_token]
            body = self.entity_index.tag(t.text)
            output.append(
                f'<topic id={t.id}, confidence={t.confidence}>{body}</topic>'
            )
            current = t.end_token
        output += tokens[current:]
        return ' '.join(output)

    def scan(self, text, n_tokens=20):
        tokens = text.split()
        chunks = [
            [
                Topic(
                    id=t.id,
                    topic=t.topic,
                    text=t.text,
                    document=t.document,
                    confidence=t.confidence,
                    vector=t.vector,
                    start_token=i,
                    end_token=i+j,
                )
                for t in self.search(' '.join(tokens[i:i+j]))
            ][0]
            for i in range(len(tokens))
            for j in range(1, min(n_tokens, len(tokens[i:])) + 1)
        ]

        chunks = [c for c in chunks if c.confidence >= 0.7]
        topics = self.deduplicate_by_overlap(chunks)

        return topics

    def search(self, query, k=1):
        q_vec = self.embedder.encode(query.strip())
        topics_cs = cos_sim(np.array([q_vec]), self.topics_vecs)[0].numpy()
        top_k_idx = np.argpartition(topics_cs, -k)[-k:]
        results = []
        for topic_idx in top_k_idx:
            confidence = topics_cs[topic_idx]
            document_idx = self.topics_document[topic_idx]
            results.append(
                Topic(
                    id=topic_idx,
                    topic=self.topics_values[topic_idx],
                    text=query,
                    document=document_idx,
                    confidence=confidence,
                    vector=q_vec,
                )
            )
        return sorted(results, key=lambda x: x.confidence, reverse=True)

    def deduplicate_by_within(self, grp):
        remaining = []
        a = grp[0]
        for b in grp[1:]:
            if self.is_within(a, b, reverse=True):
                a = max(a, b, key=lambda x: x.confidence)
            else:
                remaining.append(b)

        if len(remaining) < 2:
            return [a] + remaining

        return [a] + self.deduplicate_by_within(remaining)

    def deduplicate_by_overlap(self, grp):
        remaining = []
        a = grp[0]
        for b in grp[1:]:
            if self.overlap(a, b) > 0:
                a = max(a, b, key=lambda x: x.confidence)
            else:
                remaining.append(b)

        if len(remaining) < 2:
            return [a] + remaining

        return [a] + self.deduplicate_by_overlap(remaining)

    def deduplicate_by_iou(self, grp):
        remaining = []
        a = grp[0]
        for b in grp[1:]:
            # TODO efficient iou calculation when we know its already below t
            if self.iou(a, b) > 0.7:
                a = max(a, b, key=lambda x: x.confidence)
            else:
                remaining.append(b)

        if len(remaining) < 2:
            return [a] + remaining

        return [a] + self.deduplicate_by_iou(remaining)

    def is_within(self, a, b, reverse=False):
        if a.start_token >= b.start_token and a.end_token <= b.end_token:
            return True
        if reverse and b.start_token >= a.start_token and b.end_token <= a.end_token:
            return True
        return False

    def overlap(self, a, b):
        tokens_a = set(range(a.start_token, a.end_token))
        tokens_b = set(range(b.start_token, b.end_token))
        return len(tokens_a.intersection(tokens_b))

    def iou(self, a, b):
        tokens_a = set(range(a.start_token, a.end_token))
        tokens_b = set(range(b.start_token, b.end_token))
        return len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b))

    def get_all_topics_for_doc(self, doc_idx):
        topics_idxes = np.argwhere(
            np.array(self.topics_document) == np.array(doc_idx)
        ).flatten()
        return [self.topics_values[i] for i in topics_idxes]

    def save(self, directory):
        self.entity_index.save(f"{directory}/entity.index")
        data = {
            'embedder_path': self.embedder.model_path,
            'topics_values': self.topics_values,
            'topics_vecs': self.topics_vecs,
            'topics_document': self.topics_document,
        }
        with open(f"{directory}/semantic.index", 'wb') as file:
            pickle.dump(data, file)
        return True
