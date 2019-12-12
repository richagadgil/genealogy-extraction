## genealogy-extraction
NLP 482 Final Project

Authors: 

        Richa Gadgil
        Nate Andre
        Mateo Ibarguen
        

#### Dependencies
```
pip install -r requirements.txt
```

If you get the following error:
```
ERROR: Could not find a version that satisfies the requirement en-core-web-sm==2.2.5 (from -r requirements.txt (line 10))
```
Make sure to run: 
```
python -m spacy download en
```

```
python -m spacy download en_core_web_sm
```

#### Data (Optional)
This directory contains a subdirectory called: `data` which only contains the labels for the data,
the trained models, and the features. It does not contain the actual data.
The actual Wikipedia articles and data about our parsed entities are stored under the `src/wiki_referencer` directory.
If you wish to re-append this data, download the `genealogy-extraction_data.zip` file and follow these steps:

1. Download and unzip the file `genealogy-extraction_data.zip `.
2. Rename the directory to: `wiki_referencer`
3. Add that directory under `src`

> We have submitted our project so that it already includes the data 
in its correct location.


### Running the genealogy-extraction

```
python3 main.py
```

There are two options when running our `main.py` script:

- If you wish to copy-paste an article, do so in the command line. After you are done, type:
```
--genealogy
```

- If you instead wish to test our model on the pre-loaded wikipedia articles, type the following:
```
--generate
``` 

After the features are calculated, a family tree will pop up. Once you close it, 
the gedcom file will be saved as `gedcom.ged` in this directory.

In order to exit the program, type:
```
--exit
```