import pandas as pd
import spacy
from src.wiki_referencer.wiki_reference import WikiReferencer
nlp = spacy.load('en_core_web_sm')
wiki_referencer = WikiReferencer()
import timeit
import sys
import math
from itertools import product
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy import displacy

class ArticleProcessor:

    def __init__(self, article, entity1, entity2):
        self.article = wiki_referencer.get_article_text(article)
        self.article = wiki_referencer.get_article_tags(article)
        # get all entities and variations for NER as part of feature extraction
        self.entity1 = "@" + entity1 + "@"
        self.entity2 = "@" + entity2 + "@"

        #get all entities and variations for NER as part of feature extraction
        self.entity1 = [wiki_referencer.get_entity_name(entity1)] + wiki_referencer.get_entity_aliases(entity1)
        self.entity1_gender = wiki_referencer.get_entity_gender(entity1)

        if(self.entity1_gender == "female"):
            self.entity1 += ["her", "Her", "She", "she"]
        elif(self.entity1_gender == "male"):
            self.entity1 += ["his", "His", "Him", "him"]
        print(self.entity1)


        self.entity2 = [wiki_referencer.get_entity_name(entity2)] + wiki_referencer.get_entity_aliases(entity2)
        self.all_entities = []

        # NATE -- WORK ON NER
        for i in wiki_referencer.get_article_entities(article):
            self.all_entities = self.all_entities + [wiki_referencer.get_entity_name(i)] + wiki_referencer.get_entity_aliases(i)

        self.entity1 = self.first_entity_names(self.entity1)
        self.entity2 = self.first_entity_names(self.entity2)
        self.all_entities = self.first_entity_names(self.all_entities)

        self.features = {}

        self.text = wiki_referencer.get_article_text(article)
        self.occurences()


    def first_entity_names(self, entities):
        new_entities = entities.copy()
        for i in entities:
            if(len(i.split(' ')) > 0 and i.split(' ')[0] not in new_entities):
                new_entities.append(i.split(' ')[0])
        return new_entities

    def occurences(self):

        paragraphs = self.text.split("\n") #TO-DO: improve paragraph tokenizer?
        paragraphs = [p for p in paragraphs if len(p) > 0]

        #defaults for classifier features
        self.features["first_occurrence_entities_in_same_sentence"] = False
        self.features["first_occurrence_e2_possessive"] = False
        self.features["first_occurrence_e1_possessive"] = False
        self.features["first_occurrence_words_in_between"] = 0
        self.features["first_occurrence_entities_in_between"] = 0

        self.features["shortest_occurrence_entities_in_same_sentence"] = False
        self.features["shortest_occurrence_e2_possessive"] = False
        self.features["shortest_occurrence_e1_possessive"] = False
        self.features["shortest_occurrence_words_in_between"] = 0
        self.features["shortest_occurrence_entities_in_between"] = 0

        #triple-relationship extraction
        self.features["spouse_relationship"] = True
        self.features["child_relationship"] = True
        self.features["sibling_relationship"] = True
        self.features["mother_relationship"] = True
        self.features["father_relationship"] = True


        first_paragraph_with_both_entities = None
        shortest_distance_between_entities = sys.maxsize

        #dictionary for triple extraction features
        # for married, look for instance of married between two entities


        for p in range(0, len(paragraphs)):
            text = nlp(paragraphs[p])
            sentences = list(text.sents)
            tokenized_text = [w.text for w in text]
            place = 0

            first_e1_index, first_e2_index = None, None
            e1_occurences, e2_occurences = [], []

            for s in sentences:
                tokenized = [w.text for w in s]
                #print(s)
                sentence_entities = list(set([(i, tokenized.index(i)) for i in self.all_entities if i in tokenized]))
                found_entities = [(i[0], place + i[1]) for i in sentence_entities]


                e1_found = [i[1] for i in found_entities if i[0] in self.entity1]
                e2_found = [i[1] for i in found_entities if i[0] in self.entity2]

                #------------------
                #TRIPLE EXTRACTION

                #nsubj = [w for w in s if w.dep_ == 'nsubj']
                #root = [w for w in s if w.dep_ == 'ROOT']
                #entity_nsubj = [w.text for w in nsubj if w.text in self.all_entities]

                # MARRIAGE-DETECTION
                #mdict = ["married", "remarried", "marry"]
                #married_root = [w.text for w in root if w.text in mdict]

                #if(len(entity_nsubj) > 0 and len(married_root) > 0):
                #    index = tokenized.index(married_root[0])
                #    if(len([i for i in sentence_entities if i[1] > index and i[0] in self.entity2]) > 0):
                #        self.features["spouse"] = True
                #        print(s, "\n")



                # POSESSIVE-DETECTION
                matcher = Matcher(nlp.vocab)
                pattern1 = [{'DEP': 'poss'},  # adjectival modifier
                           {'DEP': 'amod', 'OP': "?"},
                           {'DEP': 'pobj'},
                           {'DEP': 'punct', 'OP': "?"},
                           {'POS': 'PROPN'}]
                pattern2 = [{'DEP': 'poss'},  # adjectival modifier
                           {'DEP': 'amod', 'OP': "?"},
                           {'DEP': 'dobj'},
                           {'DEP': 'punct', 'OP': "?"},
                           {'POS': 'PROPN'}]
                pattern3 = [{'DEP': 'dobj'},  # adjectival modifier
                            {'DEP': 'punct', 'OP': "?"},
                            {'DEP': 'poss'},
                            {'DEP': 'case', 'OP': "?"},
                            {'DEP': 'appos'}]

                #patterns = [pattern1, pattern2, pattern3]
                #matcher.add("matching", None, pattern1, pattern2, pattern3)
                #matches = matcher(s.as_doc())
                #hi= []
                #for match_id, start, end in matches:
                #    print(s, "\n", s[start:end], "\n\n")



                #------------------
                #CLASSIFIER_FEATURES

                e1_occurences = e1_occurences + e1_found
                e2_occurences = e2_occurences + e2_found

                if (len(e1_occurences) > 0 and len(e2_occurences) > 0):
                    #if (first_paragraph_with_both_entities == None):
                    #    self.features["first_occurrence_entities_in_same_sentence"] = True
                    first_e1_index = min(e1_occurences)
                    first_e2_index = min(e2_occurences)
                    if(first_e1_index in e1_found and first_e2_index in e2_found):
                        self.features["first_occurrence_entities_in_same_sentence"] = True

                    shortest_e1_index = sorted(product(e1_occurences, e2_occurences), key=lambda t: abs(t[0]-t[1]))[0][0]
                    shortest_e2_index = sorted(product(e1_occurences, e2_occurences), key=lambda t: abs(t[0] - t[1]))[0][1]
                    if (shortest_e1_index in e1_found and shortest_e2_index in e2_found):
                        self.features["shortest_occurrence_entities_in_same_sentence"] = True

                   # print(e1_occurences, e2_occurences, shortest_e2_index, shortest_e1_index)

                    if(first_paragraph_with_both_entities == None):
                        if(first_e1_index != None and first_e2_index != None): #MATCH FOUND
                            self.features["first_occurrence_words_in_between"] = abs(first_e1_index - first_e2_index)
                            entities_in_between = [i for i in found_entities if (first_e1_index < i[1] < first_e2_index or first_e1_index > i[1] > first_e2_index) and i[0] not in self.entity1 and i[0] not in self.entity2]  # TO-DO: focus on chrnological order here or not?
                            self.features["first_occurrence_entities_in_between"] = len(entities_in_between)
                            if (tokenized_text[first_e1_index + 1] == "'s"):
                                self.features["first_occurrence_e1_possessive"] = True
                            if (tokenized_text[first_e2_index + 1] == "'s"):
                                self.features["first_occurrence_e2_possessive"] = True
                            first_paragraph_with_both_entities = paragraphs[p]


                    words_in_between = abs(shortest_e1_index - shortest_e2_index)
                    if(words_in_between < shortest_distance_between_entities):
                        shortest_distance_between_entities = words_in_between
                        self.features["shortest_occurrence_words_in_between"] = abs(shortest_e1_index - shortest_e2_index)
                        entities_in_between = [i for i in found_entities if (
                                    shortest_e1_index < i[1] < shortest_e2_index or shortest_e1_index > i[1] > shortest_e2_index) and i[
                                                   0] not in self.entity1 and i[
                                                   0] not in self.entity2]  # TO-DO: focus on chrnological order here or not?
                        self.features["shortest_occurrence_entities_in_between"] = len(entities_in_between)
                        if (tokenized_text[shortest_e1_index + 1] == "'s"):
                            self.features["shortest_occurrence_e1_possessive"] = True
                        if (tokenized_text[shortest_e2_index + 1] == "'s"):
                            self.features["shortest_occurrence_e2_possessive"] = True

                      #  print(paragraphs[p])
                place += len(tokenized)

        #print( self.features)



if __name__ == '__main__':
    p = ArticleProcessor('148301', 'Q77335', 'Q75392161')
    p = ArticleProcessor('1467835', 'Q274606', 'Q3769073')


#Q76343	Q3434236	P40	148301
#Q274606 Q3769073 1467835



