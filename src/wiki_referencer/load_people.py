from wikidata.client import Client
import requests
import re

client = Client()


def get_links(limit):
    url = 'https://query.wikidata.org/sparql'
    query = """
    SELECT DISTINCT ?humanLabel
    WHERE
    {
      ?human wdt:P31 wd:Q5.
      ?human wdt:P26 ?spouse .
      ?human wdt:P3373 ?sibling .
      ?human wdt:P22 ?father .
      ?human wdt:P25 ?mother .
      ?human wdt:P40 ?child .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
    }
    LIMIT %d
    """ % limit
    r = requests.get(url, params={'format': 'json', 'query': query})
    data = r.json()
    return [d['humanLabel']['value'] for d in data['results']['bindings']]


def load_people(limit=1000):
    wiki_ids = get_links(limit)
    return {wiki_id: client.get(wiki_id) for wiki_id in wiki_ids}


if __name__ == '__main__':
    print(load_people(100))