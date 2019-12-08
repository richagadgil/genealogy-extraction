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

    def __init__(self, client):
        self.client = client
        self.relationship_ids = ['P26', 'P3373', 'P22', 'P25', 'P40']
        self.relationship_ids_set = set(self.relationship_ids)
        self.relationship_ids_dict = {relation_id: self.client.get(relation_id) for relation_id in self.relationship_ids}

    def _extract_relation(self, entity, relation_id):
        """

        :param entity: From client.get(wikidata_id). The person's entity.
        :param relatilon_id: The Wikidata id for that entity. Ex: P3373 for siblings.
        :return: All the entities in that person's Wikidata that satsifoies the
        """
        try:
            relation = self.relationship_ids_dict[relation_id]
            related_entities = entity.getlist(relation)
            return {related_entity: relation_id for related_entity in related_entities}

        except Exception as e:
            print(e)
            return None

    def extract_relations(self, entity):
        """

        :param wikidata_id: (String) The Wikidata id for the person we're interested in.
        :return: A dictionary in which the keys are the relationship ids and the values are the entities.
        """

        valid_relations = list(self.relationship_ids_set -
                               (self.relationship_ids_set - set(entity.attributes['claims'].keys())))

        entity_relations = list(filter(None, [self._extract_relation(entity, relation_id)
                                              for relation_id in valid_relations]))
        flattened_relations = self._flatten_dict(entity_relations)
        return flattened_relations

    def _flatten_dict(self, dictionary):
        return {k: v for d in dictionary for k, v in d.items()}




