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
import re



def preprocess_article_id(article_id, entity1_id, entity2_id):
    article = wiki_referencer.get_article_tags(article_id)
    entity1 = "@" + entity1_id+ "@"
    entity2 = "@"+entity2_id+"@"
    return ArticleProcessor(article, entity1, entity2)


class ArticleProcessor:
    def __init__(self, article, entity1, entity2, load_wiki=True):
        print("processing_relationship...")
        if load_wiki:
            self.article = wiki_referencer.get_article_tags(article)
            #get all entities and variations for NER as part of feature extraction

        else:
            self.article = article

        self.entity1 = "@"+str(entity1)+"@"
        self.entity2 = "@" + str(entity2) + "@"

        self.features = {}


        self.relationships = {}
        self.relationships["child"] = ["son", "daughter"]
        self.relationships["mother"] = ["mother"]
        self.relationships["father"] = ["father"]
        self.relationships["sibling"] = ["brother", "sister", "sibling"]
        self.relationships["spouse"] = ["husband", "wife"]
        self.relationship_words = ["son", "daughter", "mother", "father", "brother", "sister", "sibling", "husband",
                                   "wife", "married"]


        self.occurences()



    def occurences(self):

        paragraphs = self.article.split("\n") #TO-DO: improve paragraph tokenizer?
        paragraphs = [p for p in paragraphs if len(p) > 0]

        first_paragraph_with_both_entities = None
        shortest_distance_between_entities = sys.maxsize

        #dictionary for triple extraction features

        e1_eventual = False
        e2_eventual = False

        for p in range(0, len(paragraphs)):
            place = 0
            p_ents = {}
            text = nlp(paragraphs[p])
            sentences = list(text.sents)

            regex = re.compile('@(\w+)@')
            matches = regex.finditer(text.text)
            for i in matches:
                #print(i)
                if i.group() in p_ents:
                    p_ents[i.group()].append(i.span()[0])
                else:
                    p_ents[i.group()] = [i.span()[0]]



            p_ents["gender"] = []
            #if (self.entity1_gender == "male"):
            #    regex = re.compile(' (his|him|himself|he)[ |\,|\.]', re.IGNORECASE)
            #elif (self.entity1_gender == "female"):
            #    regex = re.compile(' (hers|her|herself|she)[ |\,|\.]', re.IGNORECASE)
            #matches = regex.finditer(text.text)
            #for i in matches:
            #    p_ents["gender"].append(i.span()[0])


            if(self.entity1 in p_ents.keys()):
                e1_eventual = True
            if (self.entity2 in p_ents.keys()):
                e2_eventual = True

            if(self.entity1 in p_ents.keys() and self.entity2 in p_ents.keys()): #if paragraph contains both entities


                if(first_paragraph_with_both_entities == None):
                    self.features["first_occurence_entity_ratio"] = len(p_ents[self.entity1]) / len(p_ents[self.entity2])
                    first_paragraph_with_both_entities = True
                    e1_first = min(p_ents[self.entity1]) #+ p_ents["gender"])
                    e2_first = min(p_ents[self.entity2])

                    if (e1_first < e2_first):
                        text_in_between = nlp(text.text[e1_first:e2_first])
                    else:
                        text_in_between = nlp(text.text[e2_first:e1_first])


                    self.features['first_occurence_words_in_between'] = len(text_in_between)
                    if(len(list(text_in_between.sents)) == 1):
                        self.features['first_occurence_same_sentence'] = True


                    poss1_index = e1_first+len(self.entity1)
                    poss2_index = e2_first + len(self.entity2)
                    if(poss1_index+2 < len(text.text) and text.text[poss1_index:poss1_index+2] == "'s"):
                        self.features['first_occurence_e1_posessive'] = True
                    if(poss2_index+2 < len(text.text) and text.text[poss2_index:poss2_index+2] == "'s"):
                        self.features['first_occurence_e2_posessive'] = True

                    entities_in_between = []
                    for key in p_ents:
                        vals = p_ents[key]
                        if(len([i for i in vals if e1_first < i < e2_first or e2_first < i < e1_first]) > 0):
                            entities_in_between.append(key)

                    if(len(entities_in_between) > 0):
                        self.features['first_occurence_entities_in_between'] = True
                        self.features['first_occurence_entities_in_between_count'] = len(entities_in_between)

                    for word in text_in_between:
                        for key in self.relationships:
                            if (word.text in self.relationships[key]):
                                feature = 'first_occurence_' + key
                                self.features[feature] = True


                min_difference = sys.maxsize

                for x in sorted(p_ents[self.entity1]): #+ p_ents["gender"]):
                    for y in sorted(p_ents[self.entity2]):
                        if(x < y and y - x < min_difference):
                            text_in_between = nlp(text.text[x:y])
                            min_difference = len(text_in_between)
                        elif(x - y < sys.maxsize):
                            text_in_between = nlp(text.text[y:x])
                            min_difference = len(text_in_between)


                if(len(text_in_between) < shortest_distance_between_entities):
                    shortest_distance_between_entities = len(text_in_between)
                    self.features["shortest_occurence_entity_ratio"] = len(p_ents[self.entity1]) / len(p_ents[self.entity2])
                    self.features['shortest_occurence_words_in_between'] = len(text_in_between)
                    if (len(list(text_in_between.sents)) == 1):
                        self.features['shortest_occurence_same_sentence'] = True

                    poss1_index = e1_first + len(self.entity1)
                    poss2_index = e2_first + len(self.entity2)
                    if (poss1_index + 2 < len(text.text) and text.text[poss1_index:poss1_index + 2] == "'s"):
                        self.features['shortest_occurence_e1_posessive'] = True
                    if (poss2_index + 2 < len(text.text) and text.text[poss2_index:poss2_index + 2] == "'s"):
                        self.features['shortest_occurence_e2_posessive'] = True

                    entities_in_between = []
                    for key in p_ents:
                        vals = p_ents[key]
                        if (len([i for i in vals if e1_first < i < e2_first or e2_first < i < e1_first]) > 0):
                            entities_in_between.append(key)

                    if (len(entities_in_between) > 0):
                        self.features['shortest_occurence_entities_in_between'] = True
                        self.features['shortest_occurence_entities_in_between_count'] = len(entities_in_between)

                    #for word in text_in_between:
                    #    if word.text in self.relationship_words:
                    #        self.features['shortest_occurence_relationships'] = True
                    for word in text_in_between:
                        for key in self.relationships:
                            if(word.text in self.relationships[key]):
                                feature = 'shortest_occurence_'+key
                                self.features[feature] = True

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
                matches = matcher(text)
                for match_id, start, end in matches:
                    # print([(a.text, a.dep_, a.pos_) for a in s])
                    found = []
                    entity2_found = None
                    for i in text[start:end]:
                        # print(s[start:end])
                        found = found + [k for k, v in self.relationships.items() if i.text in v]
                        if (i.text in self.entity2):
                            entity2_found = True
                    if (len(found) > 0 and entity2_found == True):
                        features = "paragraph_"+found[0]
                        self.features[features] = True











                #for i in p_ents[self.entity1]:
                #    print(text.text[i])






                #print([a.i for a in s], "\n")
                #tokenized = [w.text for w in s]
                #print(s)
                #sentence_entities = list(set([(i, tokenized.index(i)) for i in self.all_entities if i in tokenized or i[0] in self.gender]))
                #found_entities = [(i[0], place + i[1]) for i in sentence_entities]


                #e1_found = [i[1] for i in found_entities if i[0] in self.entity1 or i[0] in self.gender]
                #e2_found = [i[1] for i in found_entities if i[0] in self.entity2 or i[0] in self.gender]

                #---------------------
                #CLASSIFIER_FEATURES

                #if (len(e1_occurences) > 0 and len(e2_occurences) > 0):
                    #if (first_paragraph_with_both_entities == None):
                    #    self.features["first_occurrence_entities_in_same_sentence"] = True
                #    first_e1_index = min(e1_occurences)
                #    first_e2_index = min(e2_occurences)
                #    if(first_e1_index in e1_found and first_e2_index in e2_found):
                #        self.features["first_occurrence_entities_in_same_sentence"] = True

                #    shortest_e1_index = sorted(product(e1_occurences, e2_occurences), key=lambda t: abs(t[0]-t[1]))[0][0]
                #    shortest_e2_index = sorted(product(e1_occurences, e2_occurences), key=lambda t: abs(t[0] - t[1]))[0][1]
                #    if (shortest_e1_index in e1_found and shortest_e2_index in e2_found):
                #        self.features["shortest_occurrence_entities_in_same_sentence"] = True

                   # print(e1_occurences, e2_occurences, shortest_e2_index, shortest_e1_index)

                #    if(first_paragraph_with_both_entities == None):
                #        if(first_e1_index != None and first_e2_index != None): #MATCH FOUND
                #            self.features["first_occurrence_words_in_between"] = abs(first_e1_index - first_e2_index)
                #            entities_in_between = [i for i in found_entities if (first_e1_index < i[1] < first_e2_index or first_e1_index > i[1] > first_e2_index) and i[0] not in self.entity1 and i[0] not in self.entity2]  # TO-DO: focus on chrnological order here or not?
                #            self.features["first_occurrence_entities_in_between"] = len(entities_in_between)
                #            if (tokenized_text[first_e1_index + 1] == "'s"):
                #                self.features["first_occurrence_e1_possessive"] = True
                #            if (tokenized_text[first_e2_index + 1] == "'s"):
                #                self.features["first_occurrence_e2_possessive"] = True
                #            first_paragraph_with_both_entities = paragraphs[p]


                ##    words_in_between = abs(shortest_e1_index - shortest_e2_index)
                ##    if(words_in_between < shortest_distance_between_entities):
                #        shortest_distance_between_entities = words_in_between
                #        self.features["shortest_occurrence_words_in_between"] = abs(shortest_e1_index - shortest_e2_index)
                #        entities_in_between = [i for i in found_entities if (
                #                    shortest_e1_index < i[1] < shortest_e2_index or shortest_e1_index > i[1] > shortest_e2_index) and i[
                #                                   0] not in self.entity1 and i[
                #                                   0] not in self.entity2]  # TO-DO: focus on chrnological order here or not?
                #        self.features["shortest_occurrence_entities_in_between"] = len(entities_in_between)
                #        if (tokenized_text[shortest_e1_index + 1] == "'s"):
                #            self.features["shortest_occurrence_e1_possessive"] = True
                #        if (tokenized_text[shortest_e2_index + 1] == "'s"):
                #            self.features["shortest_occurrence_e2_possessive"] = True

                      #  print(paragraphs[p])
                #place += len(tokenized)

if __name__ == '__main__':
    p = ArticleProcessor('148301', 'Q77335', 'Q75392161')
    print("\n")
    p = ArticleProcessor('1467835', 'Q274606', 'Q3769073')


#Q76343	Q3434236	P40	148301
#Q274606 Q3769073 1467835



