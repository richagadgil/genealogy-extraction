import os
from abc import ABC, abstractmethod
from src.wiki_referencer.wiki_reference import WikiReferencer
import numpy as np
import pandas as pd
from src.article_processor.article_processor import ArticleProcessor
import nltk
from sklearn.model_selection import train_test_split

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

    @abstractmethod
    def predict_test(self):
        pass

    def evaluate_labels(self, labels):
        # TODO: Add more extensive evaluation metrics.
        true_relation_values = labels['relation']
        self.predicted_relations = self.predict_relations(labels)
        accuracy = np.mean(true_relation_values == self.predicted_relations)
        print('Test accuracy: ', accuracy)





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
        # randomly sample number of relations to train and test on.
        self.labels = self.train_labels.sample(num_train).dropna()
        self.train_labels, self.test_labels = self.train_test_split(self.labels)
        self.i = 0

    def train_test_split(self, labels):
        X = labels[['entity_a', 'entity_b', 'article_id']].values
        y = labels['relation'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, shuffle=True)
        train_df = pd.DataFrame(X_train, columns=labels[['entity_a', 'entity_b', 'article_id']].columns)
        train_df['relation'] = y_train

        test_df = pd.DataFrame(X_test, columns=labels[['entity_a', 'entity_b', 'article_id']].columns)
        test_df['relation'] = y_test

        return train_df, test_df




    def fit_article(self, article_id, entity_1, entity_2):
        return ArticleProcessor(article_id, entity_1, entity_2).features

    def fit_train(self):
        self.train_fts = self.train_labels.apply(lambda x: self.fit_article(x.article_id, x.entity_a, x.entity_b), axis=1)
        self.train_fts_list = self.train_fts.tolist()
        self.train_fts_dict = [(ft, relation) for relation, ft in zip(self.train_labels['relation'].tolist(), self.train_fts_list)]
        self.classifier = nltk.NaiveBayesClassifier.train(self.train_fts_dict)


    def predict_relation_from_ids(self, entity_a_id, entity_b_id, article_id):
        predict_fts = self.fit_article(article_id, entity_a_id, entity_b_id)
        #probs = self.classifier.prob_classify(predict_fts)
        classif = self.classifier.classify(predict_fts)
        # for label in probs.samples():
        #     print("%s: %f" % (label, probs.prob(label)))
        return classif

    def predict_test(self):
        test_ft = self.test_labels.apply(lambda x: self.fit_article(x.article_id, x.entity_a, x.entity_b), axis=1)
        test_fts_list = test_ft.tolist()
        test_fts_dict = [(ft, relation) for relation, ft in
                               zip(self.test_labels['relation'].tolist(), test_fts_list)]
        #test_preds = nltk.classify.accuracy(self.classifier, test_fts_dict)
        self.test_preds = self.classifier.classify_many(test_fts_list)

    def evaluate_test(self):
        self.predict_test()
        true_labels = self.test_labels['relation']
        accuracy = np.mean(self.test_preds == true_labels)
        print('test accuracy: ', accuracy)
        return accuracy


if __name__ == '__main__':
    # baseline_model = BaselineRelationModel()
    # baseline_model.fit_train()
    # baseline_model.evaluate_test()
    # baseline_model.predict('Berengar I of Italy')

    entity_fts = EntityFeatureRelationModel()
    entity_fts.fit_train()
    entity_fts.predict_test()


