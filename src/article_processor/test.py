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

hi = nlp("dog was the master. of dog")
print(len(list(hi.sents)))