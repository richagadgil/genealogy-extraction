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
        self.article = wiki_referencer.get_article_tags(article)
        #get all entities and variations for NER as part of feature extraction
        self.entity1 = "@"+entity1+"@"
        self.entity2 = "@"+entity2+"@"

        self.all_entities = wiki_referencer.get_article_entities(article)
        self.all_entities = ["@"+i+"@" for i in self.all_entities]
        self.entity1_gender = wiki_referencer.get_entity_gender(entity1)

        if (self.entity1_gender == "female"):
            self.gender = ["her", "Her", "She", "she"]
        elif (self.entity1_gender == "male"):
            self.gender = ["his", "His", "Him", "him"]

        #TO-DO: append Gender Pronouns

        self.features = {}


        self.text = wiki_referencer.get_article_text(article)

        #self.doc = nlp(self.text)
        #start = timeit.timeit()
        self.occurences()



    def occurences(self):

        paragraphs = self.article.split("\n") #TO-DO: improve paragraph tokenizer?
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

        self.relationships = {}
        self.relationships["child"] = ["son", "daughter"]
        self.relationships["mother"] = ["mother"]
        self.relationships["father"] = ["father"]
        self.relationships["sibling"] = ["brother", "sister", "sibling"]


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
                sentence_entities = list(set([(i, tokenized.index(i)) for i in self.all_entities if i in tokenized or i[0] in self.gender]))
                found_entities = [(i[0], place + i[1]) for i in sentence_entities]


                e1_found = [i[1] for i in found_entities if i[0] in self.entity1 or i[0] in self.gender]
                e2_found = [i[1] for i in found_entities if i[0] in self.entity2 or i[0] in self.gender]

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

                pattern1 = [{'DEP': 'poss'},  # adjectival modifier
                           {'DEP': 'amod', 'OP': "?"},
                           {'DEP': 'pobj'},
                           {'DEP': 'punct', 'OP': "?"},
                           {'POS': 'PROPN'},
                            {'DEP': 'appos', 'OP': "?"}]
                pattern2 = [{'DEP': 'poss'},  # adjectival modifier
                           {'DEP': 'amod', 'OP': "?"},
                           {'DEP': 'dobj'},
                           {'DEP': 'punct', 'OP': "?"},
                           {'POS': 'PROPN'},
                            {'DEP': 'appos', 'OP': "?"}]
                pattern3 = [{'DEP': 'dobj'},  # adjectival modifier
                            {'DEP': 'punct', 'OP': "?"},
                            {'DEP': 'poss'},
                            {'DEP': 'case', 'OP': "?"},
                            {'DEP': 'appos'}]

                patterns = [pattern1, pattern2, pattern3]
                matcher = Matcher(nlp.vocab)
                matcher.add("matching", None, pattern1, pattern2, pattern3)
                matches = matcher(s.as_doc())
                for match_id, start, end in matches:
                    # print([(a.text, a.dep_, a.pos_) for a in s])
                    found = []
                    entity2_found = None
                    for i in s[start:end]:
                        # print(s[start:end])
                        found = found + [k for k, v in self.relationships.items() if i.text in v]
                        if (i.text in self.entity2):
                            entity2_found = True
                    if (len(found) > 0 and entity2_found == True):
                        self.features[found[0]] = True
                        print(found[0], s[start:end])



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

    def triple_occurence(self, text):
        pass


if __name__ == '__main__':
    p = ArticleProcessor('148301', 'Q77335', 'Q75392161')
    p = ArticleProcessor('1467835', 'Q274606', 'Q3769073')


#Q76343	Q3434236	P40	148301
#Q274606 Q3769073 1467835



