# wiki_referencer
Mateo Ibarguen
 
```python
from src.wiki_referencer.wiki_reference import WikiReferencer
wiki_referencer = WikiReferencer()

#### Get all entities that appear in an article.
wiki_referencer.article_entities

### Get articles
wiki_referencer.article_id_text

# Get a list of aliases per entity id.
# Not all entities have aliases but
# they should all have names. 
wiki_referencer.entity_id_aliases

# Get a name per entity id.
wiki_referencer.entity_id_name

# Get the latest article where an entity id appears.
wiki_referencer.entity_article

# Get relationships between entities. 
wiki_referencer.relations

```
