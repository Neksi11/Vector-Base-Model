from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class DocumentClassifier:
    """Document classifier using Random Forest algorithm."""
    
    def __init__(self):
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
    def train(self, documents, labels):
        """Train the classifier on documents with their labels."""
        self.pipeline.fit(documents, labels)
        
    def predict(self, documents):
        """Predict labels for new documents."""
        return self.pipeline.predict(documents)
    
    def predict_proba(self, documents):
        """Get probability estimates for each class."""
        return self.pipeline.predict_proba(documents)