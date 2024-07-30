import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class SVMModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.svm_model = svm.SVC(kernel='linear', C=1.0, random_state=42)
        self.label_encoder = LabelEncoder()
        self.stop_words = ['is', 'are', 'am', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']

    def fit(self, X, y):
        """
        Train the SVM model on a dataset of symptoms and diseases.
        """
        X_preprocessed = self.vectorizer.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        self.svm_model.fit(X_preprocessed, y_encoded)

    def predict_disease(self, input_symptoms):
        """
        Make a prediction on new input data.
        """
        input_vector = self.preprocess_symptoms(input_symptoms)
        prediction = self.svm_model.predict(input_vector)
        return self.get_disease_from_prediction(prediction)

    def get_disease_from_prediction(self, prediction):
        """
        Convert the numerical prediction to a human-readable disease name.
        """
        return self.label_encoder.inverse_transform(prediction)[0]

    def preprocess_symptoms(self, symptoms):
        """
        Preprocess the symptoms, handling missing values, creating tokens, removing stopwords, and removing special characters.
        """
        # Tokenize the symptoms
        tokens = [word for word in symptoms.split() if word.isalpha()]

        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]

        # Create a bag-of-words representation
        vector = self.vectorizer.transform([' '.join(tokens)])

        return vector

    def run(self):
        # Load the CSV dataset
        df = pd.read_csv('dataset.csv')

        # Assume the CSV has two columns: 'text' and 'label'
        X = df['text']
        y = df['label']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.fit(X_train, y_train)

        # For testing purpose: Get user input
        input_symptoms = input("Enter your symptoms: ")

        # Make a prediction on the user input
        predicted_disease = self.predict_disease(input_symptoms)

        print("Predicted disease:", predicted_disease)

if __name__ == "__main__":
    svm_model = SVMModel()
    svm_model.run()
