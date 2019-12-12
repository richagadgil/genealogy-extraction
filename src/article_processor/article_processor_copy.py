import pandas as pd
import spacy
from src.wiki_referencer.wiki_reference import WikiReferencer

nlp = spacy.load('en_core_web_sm')
wiki_referencer = WikiReferencer()
import timeit
import sys
import math
import itertools


class ArticleProcessor:

    def __init__(self, article, entity1, entity2):
        self.article = wiki_referencer.get_article_text(article)

        # get all entities and variations for NER as part of feature extraction
        self.entity1 = [wiki_referencer.get_entity_name(entity1)] + wiki_referencer.get_entity_aliases(entity1)
        self.entity2 = [wiki_referencer.get_entity_name(entity2)] + wiki_referencer.get_entity_aliases(entity2)
        self.all_entities = []
        for i in wiki_referencer.get_article_entities(article):
            self.all_entities = self.all_entities + [
                wiki_referencer.get_entity_name(i)] + wiki_referencer.get_entity_aliases(i)

        self.entity1 = self.first_entity_names(self.entity1)
        self.entity2 = self.first_entity_names(self.entity2)
        self.all_entities = self.first_entity_names(self.all_entities)

        # TO-DO: append Gender Pronouns

        self.features = {}

        self.text = wiki_referencer.get_article_text(article)

        # self.doc = nlp(self.text)
        # start = timeit.timeit()
        self.occurences()

    def first_entity_names(self, entities):
        new_entities = entities.copy()
        for i in entities:
            print(i)
            if (len(i.split(' ')) > 0 and i.split(' ')[0] not in new_entities):
                new_entities.append(i.split(' ')[0])
        return new_entities

    def occurences(self):

        closest_occurence_words_in_between = sys.maxsize
        paragraphs = self.text.split("\n")  # TO-DO: improve paragraph tokenizer?
        paragraphs = [p for p in paragraphs if len(p) > 0]

        for p in range(0, len(paragraphs)):
            tokenized_text = nlp(paragraphs[p])
            sentences = list(tokenized_text.sents)

            # start with first place in paragraph
            place = 0

            first_e1, first_e2 = sys.maxsize, sys.maxsize

            for s in sentences:
                e1_found, e2_found, closest_changed = False, False, False
                tokenized = [w.text for w in s]
                found_entities = [(i, place + tokenized.index(i)) for i in self.all_entities if i in tokenized]

                if (first_e1 == sys.maxsize):
                    e1 = [i[1] for i in found_entities if i[0] in self.entity1]
                    if (len(e1) > 0):
                        first_e1 = e1[0]
                        # e1_found = True
                if (first_e2 == sys.maxsize):
                    e2 = [i[1] for i in found_entities if i[0] in self.entity2]
                    if (len(e2) > 0):
                        first_e2 = e2[0]
                        # e2_found = True

                # WORDS IN BETWEEN ENTITIES
                if (first_e1 != sys.maxsize and first_e2 != sys.maxsize):
                    words_between = abs(first_e2 - first_e1)
                    if ("first_occurence_words_in_between" not in self.features):
                        self.features["first_occurence_words_in_between"] = words_between
                    elif (words_between < closest_occurence_words_in_between):
                        self.features["closest_occurence_words_in_between"] = words_between
                        closest_occurence_words_in_between = words_between
                        closest_changed = True

                # FIND ENTITIES IN BETWEEN EACH ENTITIES
                entities_in_between = [i for i in found_entities if
                                       (first_e1 < i[1] < first_e2 or first_e1 > i[1] > first_e2) and i[
                                           0] not in self.entity1 and i[0] not in self.entity2]
                if ("first_occurence_entities_in_between" not in self.features):
                    self.features["first_occurence_entities_in_between"] = entities_in_between
                    self.features["first_occurence_entities_in_between_count"] = len(entities_in_between)
                elif (
                        closest_changed == True):  # TO-DO: Maybe see if closest_changed can be equivalent to the first paragraph?
                    self.features["closest_occurence_words_in_between_entities"] = entities_in_between
                    self.features["closest_occurence_words_in_between_entities_count"] = len(entities_in_between)

                # ENTITIES IN SAME SENTENCE
                same_sentence = e1_found and e2_found
                if (p == 1):
                    self.features["first_occurence_words_same_sentence"] = same_sentence
                elif (closest_changed == True):
                    self.features["closes_occurence_words_same_sentence"] = same_sentence
                if (same_sentence):
                    if ("total_number_of_same_sentences" not in self.features):
                        self.features["total_number_of_same_sentences"] = 1
                    else:
                        self.features["total_number_of_same_sentences"] += 1

                # POSSESSIVE
                # if( [i[1] for i in found_entities if i[0] in self.entity1]

                place += len(s)


if __name__ == '__main__':
    p = ArticleProcessor('1467835', 'Q274606', 'Q3769073')






