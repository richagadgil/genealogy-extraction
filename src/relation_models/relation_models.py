import os
from abc import ABC, abstractmethod
from src.wiki_referencer.wiki_reference import WikiReferencer
import numpy as np
import pandas as pd
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = '/'.join(ROOT_DIR.split('/')[:-2])

wiki_referencer = WikiReferencer()
train_labels = pd.read_pickle(PROJECT_ROOT + '/data/train_labels.pkl')
test_labels = pd.read_pickle(PROJECT_ROOT + '/data/test_labels.pkl')


class RelationModel:
    def __init__(self, train_labels=train_labels,
                 test_labels=test_labels, wiki_referencer=wiki_referencer):
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.wiki_referencer = wiki_referencer

    @abstractmethod
    def fit_train(self):
        return self

    @abstractmethod
    def predict_relation_from_ids(self, entity_a_id, entity_b_id, article_id):
        name_a = self.wiki_referencer.get_entity_name(entity_a_id)
        name_b = self.wiki_referencer.get_entity_name(entity_b_id)
        article = self.wiki_referencer.get_article_text(article_id)
        return self.predict(name_a, name_b, article)

    def predict_relations(self, labels):
        return labels.apply(lambda row: self.predict_relation_from_ids(row.entity_a, row.entity_b, row.article_id), axis=1)

    @abstractmethod
    def predict(self, name_a, name_b, article):
        pass

    def evaluate_labels(self, labels):
        true_relation_values = labels['relation']
        predicted_relations = self.predict_relations(labels)
        accuracy = np.mean(true_relation_values == predicted_relations)
        print('Test accuracy: ', accuracy)

    def evaluate(self):
        self.evaluate_labels(self.test_labels)


class BaselineRelationModel(RelationModel):
    def __init__(self):
        super().__init__()

    def fit_train(self):
        self.train_count_relations = self.train_labels.groupby('relation')['relation'].count()
        self.proportion_relations = self.train_count_relations / self.train_count_relations.sum()
        return self

    def predict(self, name_a, name_b, article):
        return np.random.choice(list(self.proportion_relations.index), 1, p=list(self.proportion_relations.values))


if __name__ == '__main__':
    baseline_model = BaselineRelationModel()
    baseline_model.fit_train()
    baseline_model.evaluate()

