import re
from itertools import combinations

import numpy as np
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
from tqdm import tqdm


class Embedder:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = SentenceTransformer(model_path)

    def adapt_embedder(self, documents, entity_index, epochs, output_model):
        self.train_sbert(self.build_sbert_training_set(documents, entity_index, epochs))
        self.model.save(output_model)

    def build_sbert_training_set(self, documents, entity_index, epochs, seed=0):
        """
        For all documents' topics generate triplets
        """

        doc_topics = [
            entity_index.expand_topics_with_entitites(
                doc.find_all('topic', recursive=False)
            )
            for doc in tqdm(documents)
        ]

        # A document of entity names marks them as positive samples
        for entity_id in entity_index.entity_names:
            names = entity_index.entity_names[entity_id]
            doc_topics.append(names)

        positive_pairs = [
            list(combinations(topics, 2)) for topics in doc_topics if len(topics) > 1
        ]
        print(
            "Total positive pairs:", len([p for pairs in positive_pairs for p in pairs])
        )
        triplets = []
        np.random.seed(seed)

        for i, positives in enumerate(positive_pairs):
            positives *= epochs
            all_other_idx = [j for j in range(len(doc_topics)) if j != i]
            all_other_docs_topics = [doc_topics[j] for j in all_other_idx]
            all_other_docs_topics = [
                topic for doc in all_other_docs_topics for topic in doc
            ]
            negatives = np.random.choice(all_other_docs_topics, len(positives))
            triplets += [
                InputExample(texts=[pos[0], pos[1], neg])
                for pos, neg in zip(positives, negatives)
            ]
        print("Total triplets:", len(triplets))
        return triplets

    def train_sbert(self, training_set):
        train_dataloader = DataLoader(training_set, shuffle=True, batch_size=8)
        train_loss = losses.TripletLoss(self.model)
        # Epochs already built into training set for unbalanced negative sampling
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
        )

    def encode(self, inputs):
        return self.model.encode(inputs)

    def embed_paragraph(self, text):
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        one_lines = [line.strip() for line in text.split('\n')]
        one_lines = [line for line in one_lines if line != '']
        two_lines = ['. '.join(one_lines[i:i + 2]) for i in range(len(one_lines))]
        chunks = one_lines + two_lines
        vecs = self.model.encode(chunks)
        return vecs.mean(axis=0)
