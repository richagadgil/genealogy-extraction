import json
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


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
