import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from wikidata.client import Client
import numpy as np



def load_json(file_path):
    with open(file_path, 'r') as a:
        return json.load(a)


class WikiReferencer:
    def __init__(self, wiki_references_path=ROOT_DIR + '/wiki_references'):
        self.article_entities = load_json(f'{wiki_references_path}/article_entities.json')
        self.article_id_text = load_json(f'{wiki_references_path}/article_id_text.json')
        self.entity_id_aliases = load_json(f'{wiki_references_path}/entity_id_aliases.json')
        self.entity_id_name = load_json(f'{wiki_references_path}/entity_id_name.json')
        self.entity_article = load_json(f'{wiki_references_path}/entity_article.json')
        self.relations = load_json(f'{wiki_references_path}/relations.json')
        self.article_ids = list(self.article_id_text.keys())
        self.client = Client()
        self.genders = pd.read_pickle(f'{wiki_references_path}/genders_df.pkl').set_index('entity_id')
        self.genders['gender'] = self.genders['gender_id'].map({'Q6581072': 'female', 'Q6581097': 'male'})
        self.i = 0

    def get_entity_name(self, entity_id):
        return self.entity_id_name[str(entity_id)]

    def get_entity_aliases(self, entity_id):
        return self.entity_id_aliases[str(entity_id)]

    def get_entity_relations(self, entity_id):
        return self.relations[str(entity_id)]

    def get_entity_article(self, entity_id):
        return self.entity_article[str(entity_id)]

    def get_article_text(self, article_id):
        return self.article_id_text[str(article_id)]

    def get_article_entities(self, article_id):
        return self.article_entities[str(article_id)]

    def related(self, a_relations, entity_b):
        return entity_b in set(a_relations.keys())

    def get_relation(self, entity_a, entity_b):
        a_relations = self.get_entity_relations(entity_a)
        if self.related(a_relations, entity_b):
            return a_relations[entity_b]
        else:
            return None

    def get_article_relations(self, article_id):
        article_relations_df = pd.DataFrame(columns=['entity_a', 'entity_b', 'relation', 'article_id'])
        article_entities_list = self.get_article_entities(article_id)
        entity_a = article_entities_list[0]
        article_relations = [self.get_relation(entity_a, entity_b) for entity_b in article_entities_list[1:]]
        article_relations_df['relation'] = article_relations
        article_relations_df['entity_b'] = article_entities_list[1:]
        article_relations_df['entity_a'] = entity_a
        article_relations_df['article_id'] = article_id
        return article_relations_df

    def get_labels_articles(self, article_id_list):
        article_relations_df_list = [self.get_article_relations(article_id) for
                                     article_id in article_id_list]
        labels_df = pd.concat(article_relations_df_list)
        return labels_df

    def _articles_train_test(self, article_id_list, test_size=0.20):
        train_articles, test_articles = train_test_split(article_id_list,
                                                         test_size=test_size,
                                                         random_state=42)
        return train_articles, test_articles

    def _get_labels(self):
        self.train_articles, self.test_articles = self._articles_train_test(self.article_ids)
        train_labels = self.get_labels_articles(self.train_articles)
        test_labels = self.get_labels_articles(self.test_articles)
        return train_labels, test_labels

    def _get_gender(self, entity_id):
        try:
            relation = self.client.get(entity_id).attributes['claims']['P21'][0]['mainsnak']['datavalue']['value']['id']
            return relation
        except:
            print('could not find gender for: ', entity_id)
            return None

    def _get_gender_entities(self):
        genders_df = pd.DataFrame()
        entities = pd.Series(np.array(list(self.entity_id_name.keys())))
        genders = entities.apply(self._get_gender)
        genders_df['entity_id'] = entities
        genders_df['gender_id'] = genders
        return genders_df

    def get_entity_gender(self, entity_id):
        return self.genders.loc[entity_id]['gender']













