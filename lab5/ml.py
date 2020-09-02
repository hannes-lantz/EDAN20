from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class ML:
    def __init__(self):
        self.classifier = None
        self.vec = None

    def load_model(self):
        pass


    def train(self, X_dict, y):

        self.vec = DictVectorizer(sparse=True)
        X = self.vec.fit_transform(X_dict)

        print("Training the model...")
        self.classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
        #classifier = LinearDiscriminantAnalysis()
        #classifier = KNeighborsClassifier(n_neighbors=5)
        #classifier = DecisionTreeClassifier()
        model = self.classifier.fit(X, y)
        print(model)


    def predict(self, X_dict):
        vec = DictVectorizer(sparse=True)
        X = vec.fit_transform(X_dict)

        return self.classifier.predict(X)


    def test(self, X_dict, y):
        # Vectorize the test set and one-hot encoding
        X = self.vec.transform(X_dict)  # Possible to add: .toarray()
        y_predicted = self.classifier.predict(X)
        print("Classification report for classifier %s:\n%s\n"
              % (self.classifier, metrics.classification_report(y, y_predicted)))