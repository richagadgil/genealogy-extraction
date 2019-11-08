from wikidata.client import Client


class RelationExtractor:
    """
    Ex. Find Obama's relationships by:

    relation_extractor = RelationExtractor()
    relation_extractor.extract_relations('Q76')

    # To find Kim Kardashian's relations, you don't have to re-initialize the class.
    Simply type:

    relation_extractor.extract_relations('Q186304')

    """
    def __init__(self):
        self.client = Client()
        self.relationship_ids = ['P26', 'P3373', 'P22', 'P25', 'P40']

    def _extract_relation(self, person, relation_id):
        """

        :param entity: From client.get(wikidata_id). The person's entity.
        :param relatilon_id: The Wikidata id for that entity. Ex: P3373 for siblings.
        :return: All the entities in that person's Wikidata that satsifoies the
        """
        related_entities = person.entity.getlist(self.client.get(relation_id))
        return {(person, Person(related_entity)): relation_id for related_entity in related_entities}

    def extract_relations(self, wikidata_id):
        """

        :param wikidata_id: (String) The Wikidata id for the person we're interested in.
        :return: A dictionary in which the keys are the relationship ids and the values are the entities.
        """
        entity = self.client.get(wikidata_id, load=True)
        person = Person(entity)
        entity_relations = [self._extract_relation(person, relation_id) for relation_id in self.relationship_ids]
        flattened_relations = self._flatten_dict(entity_relations)
        return flattened_relations

    def _flatten_dict(self, dictionary):
        return {k: v for d in dictionary for k, v in d.items()}


class Person:
    def __init__(self, entity):
        self.entity = entity
        self.father = None
        self.mother = None
        self.siblings = None
        self.spouses = None
        self.children = None

    def __repr__(self):
        return str(self.entity.label)


class Relation:
    def __init__(self, person_a, person_b, relation_id):
        self.person_a = person_a
        self.person_b = person_b
        self.relation_id = relation_id




