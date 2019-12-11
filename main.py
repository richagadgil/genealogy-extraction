import warnings
warnings.filterwarnings('ignore')
from src.relation_models.relation_models import *
from src.wiki_referencer.wiki_reference import WikiReferencer
from src.article_processor.article_processor import ArticleProcessor
from src.utils import plot_trees
from itertools import combinations 
import random
import re
from spacy.tokens import Span
import dateparser
import spacy
nlp = spacy.load('en')


model = LogisticRelationModel()
model.persist('log_01')
wiki_referencer = WikiReferencer()


def expand_person_entities(doc):
    new_ents = []
    for ent in doc.ents:
        # Only check for title if it's a person and not the first token
        if ent.label_ == "PERSON":
            if ent.start != 0:
                # if person preceded by title, include title in entity
                prev_token = doc[ent.start - 1]
                if prev_token.text in ("Dr", "Dr.", "Mr", "Mr.", "Ms", "Ms."):
                    new_ent = Span(doc, ent.start - 1, ent.end, label=ent.label)
                    new_ents.append(new_ent)
                else:
                    # if entity can be parsed as a date, it's not a person
                    if dateparser.parse(ent.text) is None:
                        new_ents.append(ent)
        else:
            new_ents.append(ent)
    doc.ents = new_ents
    return doc

# Add the component after the named entity recognizer
# nlp.remove_pipe('expand_person_entities')
nlp.add_pipe(expand_person_entities, after='ner')


def replace_text(text):
    doc = nlp(text)
    person_names = [ent.text for ent in doc.ents if ent.label_=='PERSON']
    new_doc = text
    clean_names = []
    for name in list(set(person_names)):
        name = re.sub(r'[^a-zA-Z ]+', '', name)
        clean_names.append(name)
        new_doc = re.sub(str(name), f'@{name}@', new_doc)

    return new_doc, clean_names


def predict_relations_text(text):
    tagged_article, person_names = replace_text(text)
    entities_combinations = list(combinations(person_names, 2))
    probs = [model.predict_relation_from_ids(article_id=tagged_article, entity_a_id=ft[0], entity_b_id=ft[1], wiki_fit=False) for ft in entities_combinations]
    entities_probs = [(entities_combinations[i], probs[i]) for i in range(len(entities_combinations))]
    return entities_probs


def predict_relations_article(article_id, wiki_referencer, model):
    article_entities = wiki_referencer.get_article_entities(article_id)
    entities_combinations = list(combinations(article_entities, 2))
    entities_probs = [model.predict_relation_from_ids(article_id=article_id, entity_a_id=e_a, entity_b_id=e_b) for e_a, e_b in entities_combinations]
    entities_probs = [(entities_combinations[i], entities_probs[i]) for i in range(len(entities_combinations))]
    return entities_probs


class ArticleTree:
    def __init__(self, article_id, wiki_referencer, entities_probs, article_entities=None):
        self.article_id = article_id
        self.wiki_referencer = wiki_referencer
        self.model = model
        if article_entities is None:
            self.article_entities = self.wiki_referencer.get_article_entities(self.article_id)
        else:
            self.article_entities = article_entities

        self.family_tree = self.initialize_tree(self.article_id)
        self.entities_probs = entities_probs
        self.entities_probs_dict = {comb[0]: comb[1] for comb in self.entities_probs}
        self.relations = ['P26', 'P25', 'P40', 'P22', 'P3373']
        self.relation_maps = {'P22': self.add_father, 
                        'P26': self.add_spouse, 
                        'P25': self.add_mother, 'P3373': self.add_sibling,
                        'P40': self.add_child}

        self.most_prob = self.initialize_most_prob()
        
    def initialize_most_prob(self):
        max_class = [(p[0], p[1], p[1].max()) for p in self.entities_probs]
        max_prob = [(c[0], c[2], c[1].prob(c[2])) for c in max_class]
        return sorted(max_prob, key=lambda x: x[2], reverse=True)
    
    def get_most_prob(self, comb):
        only_one_possible = {'P26', 'P22', '25'}
        possible_relations = self.get_possible_relations(comb[0])
        comb_prob = self.entities_probs_dict[comb]
        possible_relations_prob = [(relation, comb_prob.prob(relation)) for relation in possible_relations]
        return max(possible_relations_prob, key=lambda x: x[1])
        
    def get_possible_relations(self, entity):
        entity_relations = set(self.family_tree[entity].values())
        once_per_entity = {'P25', 'P22', 'P26'}
        many_per_entity = {'P40', 'P3373'}
        possible_relations = once_per_entity - entity_relations
        possible_relations.update(many_per_entity)
        return list(possible_relations)
    
    def updated_probs(self, combs):
        try:
            return max([(comb, self.get_most_prob(comb)) for comb in combs], key=lambda x: x[1][1])
        except:
            return None
        
        
    def initialize_tree(self, article_id):
        family_tree = {}
        for entity in self.article_entities:
            article_entities_c = self.article_entities.copy()
            article_entities_c.remove(entity)
            entity_tree = {entity: {ar_ent: None for ar_ent in article_entities_c}}
            family_tree.update(entity_tree)
        return family_tree
    
    def add_father(self, comb):
        # comb[0] and comb[1] can't have relation.
        # comb[0] can't have father. 

        if 'P22' not in set(self.family_tree[comb[0]].values()):
            if self.family_tree[comb[0]][comb[1]] is None and self.family_tree[comb[1]][comb[0]] is None:
                self.family_tree[comb[0]][comb[1]] = 'P22'
                self.family_tree[comb[1]][comb[0]] = 'P40'
                return True
        return False
    
    def add_mother(self, comb):
        # comb[0] and comb[1] can't have relation.
        # comb[0] can't have father. 

        if 'P25' not in set(self.family_tree[comb[0]].values()):
            if self.family_tree[comb[0]][comb[1]] is None and self.family_tree[comb[1]][comb[0]] is None:
                self.family_tree[comb[0]][comb[1]] = 'P25'
                self.family_tree[comb[1]][comb[0]] = 'P40'
                return True
        return False
                
    def add_spouse(self, comb):
        # comb[0] and comb[1] can't have relation.
        # comb[0] can't have father. 

        if 'P26' not in set(self.family_tree[comb[0]].values()):
            if self.family_tree[comb[0]][comb[1]] is None and self.family_tree[comb[1]][comb[0]] is None:
                self.family_tree[comb[0]][comb[1]] = 'P26'
                self.family_tree[comb[1]][comb[0]] = 'P26'
                return True
        return False
                
                
    def add_sibling(self, comb):
        if self.family_tree[comb[0]][comb[1]] is None and self.family_tree[comb[1]][comb[0]] is None:
                self.family_tree[comb[0]][comb[1]] = 'P3373'
                self.family_tree[comb[1]][comb[0]] = 'P3373'
                return True
        return False
                
    def add_child(self, comb):
        if 'has_parent' not in set(self.family_tree[comb[1]].keys()):
            if self.family_tree[comb[0]][comb[1]] is None and self.family_tree[comb[1]][comb[0]] is None:
                    self.family_tree[comb[0]][comb[1]] = 'P40'
                    self.family_tree[comb[1]]['has_parent'] = True
                    return True
            return False
        # TODO: Fix this part!!!! 
                
    def add_relation(self, comb, relation):
        return self.relation_maps[relation](comb)
        
    def get_relations(self, threshold_probability):
        combos = list(self.entities_probs_dict.keys())
        added_relations = []
        most_prob_comb = self.most_prob[0]

        while len(combos) > 0:
            relation, relation_prob = self.get_most_prob(most_prob_comb[0])
            if relation_prob > threshold_probability:
                if self.add_relation(most_prob_comb[0], relation):
                    print('added_relation')
                    added_relations.append((most_prob_comb[0], relation))
                else:
                    print('Relationship did not meet threshold. ')
            # get rid of relation possibility, rather than comb
            combos.remove(most_prob_comb[0])
            most_prob_comb = self.updated_probs(combos)
        #print(added_relations)
            
        return added_relations
    
    def get_relations_name(self, threshold_probability=0.3):
        r_map = {'P22': 'father', 'P26': 'spouse', 'P25': 'mother', 'P3373': 'sibling', 'P40': 'child'}
        relations = self.get_relations(threshold_probability)
        try:
            relations_names = [(self.wiki_referencer.get_entity_name(ents[0]), 
                                self.wiki_referencer.get_entity_name(ents[1]), r_map[r]) for ents, r in relations]
            return relations_names
        except:
            return [(r[0][0].replace('@', ''), r[0][1].replace('@', ''), r_map[r[1]]) for r in relations]


def get_family_trees(article_id, wiki_referencer, model):
    article_entity_probs = predict_relations_article(article_id, wiki_referencer, model)
    article_tree = ArticleTree(article_id, wiki_referencer, article_entity_probs)
    return article_tree


def random_article_id():
    test_articles = model.test_labels['article_id'].tolist()
    random_article_id = random.sample(test_articles, 1)[0]
    return random_article_id


def random_article():
    test_articles = model.test_labels['article_id'].tolist()
    random_article_id = random.sample(test_articles, 1)[0]
    print(wiki_referencer.get_article_text(random_article_id))
    article_tree = get_family_trees(random_article_id, wiki_referencer, model)
    return article_tree

def convert_to_gedcom(relationships):
    """ Converts a list of relationship tuples to GEDCOM format
    """
    ret_string = "0 HEAD\n1 GEDC\n1 CHAR ASCII\n"
    i = 0
    individuals = []
    gedcom_relationships = []
    for rel in relationships:
        if rel[0] not in individuals: # adding all of the individuals with their names
            individuals.append(rel[0])
            gedcom_relationships.append(("@I"+str(i)+"@",rel[0],rel[2]))
            i += 1
        if rel[1] not in individuals:
            if rel[2] == "father" or rel[2] == "mother":
                individuals.append(rel[1])
                gedcom_relationships.append(("@I"+str(i)+"@",rel[1],"child"))
                i += 1
                
    for gedcom_rel in gedcom_relationships: # add all of the individual relationships to return, assume same family
        ret_string += ("0 "+gedcom_rel[0]+" INDI\n")
        ret_string += ("1 NAME "+gedcom_rel[1]+"\n")
        if gedcom_rel[2] == "child": # add the family code for child
            ret_string += ("1 FAMC @F0@\n")
        else: # add the family code for spouse
            ret_string += ("1 FAMS @F0@\n")
    
    ret_string += "0 @F0@ FAM\n" # adding the family code
    for gedcom_rel in gedcom_relationships:
        if gedcom_rel[2] == "child":
            ret_string += "1 CHIL"+gedcom_rel[0]+"\n"
        elif gedcom_rel[2] == "father":
            ret_string += "1 HUSB"+gedcom_rel[0]+"\n"
        else: # mother
            ret_string += "1 WIFE"+gedcom_rel[0]+"\n"
    
    ret_string += "0 TRLR\n"
    
    return ret_string


def main():
    print("Write '--exit' to escape program. Press enter and type '--genealogy' after pasting your article or type '--generate' to use a random article. ")
    n = input("Copy-paste your essay here:    ")

    while True:
        current_input = input()
        n += current_input

        if current_input.lower() == '--exit':
            break

        elif "--genealogy" in current_input:
            print('\nGenerate random example. ')
            n = re.sub(r'--genealogy', '', n)

            article_entity_probs = predict_relations_text(n)
            article_entities = list(set([x[0][0] for x in article_entity_probs] + [x[0][1] for x in article_entity_probs]))
            print(article_entities)

            #print('article_ents: ', article_entities)
            article_tree =  ArticleTree(article_id=None,
                      wiki_referencer=wiki_referencer,
                      entities_probs=article_entity_probs,
                      article_entities=article_entities)

            relations = article_tree.get_relations_name(threshold_probability=0.4)
            print('relations: ', relations)
            plot_trees(relations)
            gedcom = convert_to_gedcom(relations)
            print()
            print(gedcom)
            with open(PROJECT_ROOT + '/gedcom.ged', 'w') as a:
                a.write(gedcom)
            print('Saved gedcom file to: ', PROJECT_ROOT + '/gedcom.ged')
            n = ''

        elif "--generate" in current_input:
            random_article_tree = random_article()
            relations = random_article_tree.get_relations_name(threshold_probability=0.2)
            plot_trees(relations)

            gedcom = convert_to_gedcom(relations)
            print()
            print(gedcom)
            with open(PROJECT_ROOT + '/gedcom.ged', 'w') as a:
                a.write(gedcom)
            print('Saved gedcom file to: ', PROJECT_ROOT + '/gedcom.ged')

            n = ''




if __name__ == '__main__':
    #TODO: print found this number of entities. do you want relations?
    main()