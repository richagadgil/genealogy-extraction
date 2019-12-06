import os
from abc import ABC, abstractmethod
from src.wiki_referencer.wiki_reference import WikiReferencer
import numpy as np
import pandas as pd
from src.article_processor.article_processor import ArticleProcessor
import nltk

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
        """

        Returns
        -------

        """
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
        # TODO: Add more extensive evaluation metrics.
        true_relation_values = labels['relation']
        self.predicted_relations = self.predict_relations(labels)
        accuracy = np.mean(true_relation_values == self.predicted_relations)
        print('Test accuracy: ', accuracy)

    def evaluate_test(self):
        self.evaluate_labels(self.test_labels)



class BaselineRelationModel(RelationModel):
    """Randomly samples the relationship.
    TODO: Change predict to highest probable class.

    """
    def __init__(self):
        super().__init__()

    def fit_train(self):
        self.train_count_relations = self.train_labels.groupby('relation')['relation'].count()
        self.proportion_relations = self.train_count_relations / self.train_count_relations.sum()
        return self

    def predict(self, name_a, name_b, article):
        return np.random.choice(list(self.proportion_relations.index), 1, p=list(self.proportion_relations.values))[0]


class EntityFeatureRelationModel(RelationModel):
    def __init__(self, num_train=10, num_test=5):
        super().__init__()
        self.train_labels = self.train_labels.iloc[:num_train]
        self.test_labels = self.train_labels.iloc[num_train:num_train + num_test]
        self.i = 0

    def fit_article(self, article_id, entity_1, entity_2):
        self.i += 1
        print(self.i)
        return ArticleProcessor(article_id, entity_1, entity_2).features

    def fit_train(self):
        self.train_fts = self.train_labels.apply(lambda x: self.fit_article(x.article_id, x.entity_a, x.entity_b), axis=1)
        self.train_fts_list = self.train_fts.tolist()
        self.train_fts_dict = [(ft, relation) for relation, ft in zip(self.train_labels['relation'].tolist(), self.train_fts_list)]
        self.classifier = nltk.NaiveBayesClassifier.train(self.train_fts_dict)


    def predict_relation_from_ids(self, entity_a_id, entity_b_id, article_id):
        predict_fts = self.fit_article(article_id, entity_a_id, entity_b_id)
        probs = self.classifier.prob_classify(predict_fts)
        classif = self.classifier.classify(predict_fts)
        for label in probs.samples():
            print("%s: %f" % (label, probs.prob(label)))
        return probs, classif



if __name__ == '__main__':
    # baseline_model = BaselineRelationModel()
    # baseline_model.fit_train()
    # baseline_model.evaluate_test()
    # baseline_model.predict('Berengar I of Italy')

    entity_fts = EntityFeatureRelationModel()
    entity_fts.fit_train()
    print(entity_fts.classifier.show_most_informative_features(5))
    print(entity_fts.predict_relation_from_ids('Q274606', 'Q3769073', '1467835'))

