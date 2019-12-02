import wikipedia
from wikidata.client import Client
import json
from wiki_loader.load_relations import RelationExtractor
from wiki_loader.load_people import load_people
import time


class WikiLoader:
    def __init__(self, num_articles, verbose=True):
        self.num_articles = num_articles
        self.verbose = verbose
        self.client = Client()
        self.relation_extractor = RelationExtractor(self.client)
        self.top_entities_dict = load_people(self.num_articles)
        self.wiki_collection = WikiCollection(self.client)
        self.progress = 0

    def append_top_entity(self, top_entity_id, top_entity):
        start = time.time()

        top_entity_processor = TopEntityProcessor(relation_extractor=self.relation_extractor,
                                                     client=self.client,
                                                     wikidata_id=top_entity_id,
                                                     entity=top_entity)
        self.wiki_collection.add_entity_valid(top_entity_processor)
        end = time.time()
        diff_time = end-start
        self.get_progress(diff_time)

    def load_wiki(self):
        for top_entity_id, top_entity in self.top_entities_dict.items():
            self.append_top_entity(top_entity_id, top_entity)
        return self.wiki_collection

    def get_progress(self, diff_time):
        self.progress += 1
        print(self.progress, diff_time, ' num_articles ', len(self.wiki_collection.articles.keys()))


        if self.verbose:
            percent_done = round((self.progress/self.num_articles)*100, 3)
            if percent_done % 2 == 0:
                print_statement = f'{percent_done}% percent done.'
                print(print_statement)

    def save_wiki(self, wiki_collection, save_path):
        saving_dictionaries = [('entity_id_name', wiki_collection.entity_id_name),
                               ('entity_id_aliases', wiki_collection.entity_id_aliases),
                               ('article_id_text', wiki_collection.articles),
                               ('article_entities', wiki_collection.article_entities),
                               ('entity_article', wiki_collection.entity_articles),
                               ('relations', wiki_collection.relations)]

        for dictionary in saving_dictionaries:
            with open(f'{save_path}{dictionary[0]}.json', 'w') as handle:
                print(dictionary)
                json.dump(dictionary[1], handle, ensure_ascii=False)
        print(f'Reference dictionaries stored in {save_path}')

    def load_save_wiki(self, save_path='wiki_references/'):
        wiki_collection = self.load_wiki()
        self.save_wiki(wiki_collection, save_path)


class WikiCollection:
    def __init__(self, client):
        self.client = client
        self.entity_id_name = {}
        self.entity_id_aliases = {}
        self.articles = {}
        self.entity_articles = {}
        self.article_entities = {}
        self.relations = {}

    def append_article(self, article):
        article_id = article.pageid
        self.articles[article_id] = article.content

    def append_article_entities(self, article_id, entity_id):
        self.entity_articles[entity_id] = article_id
        if article_id not in set(self.article_entities.keys()):
            self.article_entities[article_id] = [entity_id]
        else:
            entities = self.article_entities[article_id] + [entity_id]
            self.article_entities[article_id] = entities

    def add_entity(self, entity_processor):
        entity_id = entity_processor.wikidata_id
        article_id = entity_processor.article_id
        self.entity_id_name[entity_id] = entity_processor.name
        self.entity_id_aliases[entity_id] = entity_processor.aliases
        self.append_article_entities(article_id, entity_id)


        if type(entity_processor) == (TopEntityProcessor):
            article = entity_processor.article
            self.append_article(article)
            self.append_relations(entity_processor)

    def add_entity_valid(self, entity_processor):
        if entity_processor.check_valid():
            self.add_entity(entity_processor)

    def append_relations(self, top_entity):
        sub_entity_processors, relations = top_entity.sub_entity_processors, top_entity.relations
        self.relations[top_entity.wikidata_id] = relations
        for sub_entity_processor in sub_entity_processors:
            self.add_entity_valid(sub_entity_processor)


class EntityProcessor:
    def __init__(self, client, wikidata_id, entity=None, article_id=None):
        self.client = client
        self.wikidata_id = wikidata_id
        self.entity = self._get_entity(entity)
        self.name = self.get_name(self.entity)
        self.aliases = self.get_aliases(self.entity)
        self.article_id = article_id

    def _get_entity(self, entity):
        if entity is None:
            return self.client.get(self.wikidata_id)
        return entity

    def get_aliases(self, entity):
        try:
            return [name['value'] for name in entity.attributes['aliases']['en']]
        except Exception as e:
            return []

    def get_name(self, entity):
        try:
            return str(entity.label)
        except Exception as e:
            return None

    def check_valid(self):
        return self.name is not None and self.article_id is not None


class TopEntityProcessor(EntityProcessor):
    def __init__(self, relation_extractor, client, wikidata_id, entity=None):
        super().__init__(client, wikidata_id, entity)
        self.relation_extractor = relation_extractor
        self.article_name = self.get_article_name()
        self.article = self.find_article(self.article_name)
        self.article_id = self.find_article_id(self.article)
        self.sub_entity_processors = None
        self.relations = None
        self.find_relations()

    def get_article_name(self):
        try:
            return self.entity.attributes['labels']['en']['value']
        except:
            return None

    def find_article(self, article_name, i=0):
        try:
            article = wikipedia.WikipediaPage(article_name)
            return article
        except Exception as e:
            print('no article', article_name)
            if len(self.aliases) > i:
                new_loc = i + 1
                return self.find_article(self.aliases[i], i=new_loc)
            print(self.aliases)
            print('Heree....')
            return None

    def find_article_id(self, article):
        try:
            return article.pageid
        except Exception as e:
            print('no article id')
            return None

    def check_valid(self):
        if super(TopEntityProcessor, self).check_valid() and len(self.sub_entity_processors) > 0:
            return True
        else:
            print('invalid!')
            print('id: ', self.wikidata_id)
            return False

    def find_relations(self):
        entity_relations = self.relation_extractor.extract_relations(self.entity)
        sub_entities = list(entity_relations.keys())
        sub_entities_processors = [EntityProcessor(client=self.client,
                                                   wikidata_id=sub_entity.id,
                                                   entity=sub_entity,
                                                   article_id=self.article_id) for sub_entity in sub_entities]
        relations = {sub_entity.id: relation for sub_entity, relation in entity_relations.items()}
        self.sub_entity_processors = sub_entities_processors
        self.relations = relations


if __name__ == '__main__':
    WikiLoader(2).load_save_wiki()


