class LogisticRelationModel(RelationModel):
    def __init__(self, num_train=100, num_test=20):
        super().__init__()
        # randomly sample number of relations to train and test on.
        self.labels = self.train_labels.sample(num_train, random_state=13).dropna()
        self.train_labels, self.test_labels = self.train_test_split(self.labels)
        self.i = 0
        self.vec1 = DictVectorizer(sparse=False)

    def train_test_split(self, labels):
        X = labels[['entity_a', 'entity_b', 'article_id']].values
        y = labels['relation'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y,
                                                            shuffle=True, random_state=42)
        train_df = pd.DataFrame(X_train, columns=labels[['entity_a', 'entity_b', 'article_id']].columns)
        train_df['relation'] = y_train

        test_df = pd.DataFrame(X_test, columns=labels[['entity_a', 'entity_b', 'article_id']].columns)
        test_df['relation'] = y_test
        return train_df, test_df


    def fit_article(self, article_id, entity_1, entity_2, wiki_fit=True):
        return ArticleProcessor(article_id, entity_1, entity_2, load_wiki=wiki_fit).features

    def fit_train(self):
        self.train_fts = self.train_labels.apply(lambda x: self.fit_article(x.article_id, x.entity_a, x.entity_b), axis=1)
        X_train = self.vec1.fit_transform(self.train_fts)
        self.classifier = LogisticRegression()
        self.classifier.fit(X_train, self.train_labels['relation'])
        self.test_fts = self.test_labels.apply(lambda x: self.fit_article(x.article_id, x.entity_a, x.entity_b), axis=1)
        X_test = self.vec1.transform(self.test_fts)
        test_preds = self.classifier.predict(X_test)
        accuracy = np.mean(test_preds == self.test_labels['relation'])
        print('test accuracy: ', accuracy)
        save_model(self.vec1, 'log_01_transformer')
        save_model(self.classifier, 'log_01')

        return accuracy

    def persist(self, model_name):
        self.vec1 = load_model(model_name+'_transformer')
        self.classifier = load_model(model_name)

    def predict_relation_from_ids(self, entity_a_id, entity_b_id, article_id, wiki_fit=True):
        if wiki_fit:
            predict_fts = self.fit_article(article_id, entity_a_id, entity_b_id)
            #probs = self.classifier.prob_classify(predict_fts)
        else:
            predict_fts = self.fit_article(article_id, entity_a_id, entity_b_id, wiki_fit=False)
        predict_vec = self.vec1.transform(predict_fts)

        results = self.classifier.predict_proba(predict_vec)[0]
        prob_per_class_dictionary = dict(zip(self.classifier.classes_, results))
        return D(prob_per_class_dictionary)