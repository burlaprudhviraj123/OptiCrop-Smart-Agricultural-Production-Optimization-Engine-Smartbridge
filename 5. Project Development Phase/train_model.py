import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def main():
    print("Loading dataset...")
    df = pd.read_csv("Crop_recommendation.csv")
    
    # Extract crop statistics
    print("Calculating crop statistics...")
    stats = {}
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for crop in df['label'].unique():
        crop_df = df[df['label'] == crop]
        stats[crop] = {}
        for feat in features:
            stats[crop][feat] = {
                'min': float(crop_df[feat].min()),
                'max': float(crop_df[feat].max()),
                'mean': float(crop_df[feat].mean())
            }
            
    with open("crop_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    print("Crop statistics saved to crop_stats.json")

    # Features and target
    X = df[features]
    y = df["label"]

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=5000)
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    # Train and evaluate models
    print("\nEvaluating machine learning models:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f" - {name} Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    # Save best model
    joblib.dump(best_model, "model.pkl")
    print(f"\nBest Model: {best_model_name} with Accuracy {best_accuracy:.4f}")
    print("Model saved as model.pkl")

if __name__ == "__main__":
    main()
