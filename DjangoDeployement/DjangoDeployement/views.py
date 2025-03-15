import joblib
from django.shortcuts import render
from django.http import HttpResponse
import shap
import numpy as np

def home(request):
    return render(request, "home.html")

def prediction(request):
    return render(request, "prediction.html")

def about(request):
    context = {
       "team_members": {
            "OGODJA Juste": "Documentation Manager",
            "KOUYELE Ashley": "Data Exploration Manager",
            "MOROU salahou":"Data processing Manager",
            "LOFO Carine": "Model Training Manager",
            "OBENAS Orens": "Model Integration Manager"
        }
    }
    return render(request, "about.html", context)

# OGODJA Juste
def result(request):
    try:
        # Charger le modèle
        cls = joblib.load('catboost_model.sav')  # Assure-toi que c'est bien un modèle CatBoost

        # Dictionnaire de correspondance (à placer avant toute utilisation)
        NObeyesdad = {
            'Underweight': 0,
            'Normal Weight': 1,
            'Obesity I': 2,
            'Obesity II': 3,
            'Obesity III': 4,
            'Overweight I': 5,
            'Overweight II': 6
        }

        # Charger le scaler utilisé lors de l'entraînement
        scaler = joblib.load('scaler.pkl')  # Assure-toi que c'est bien le bon fichier

        # Récupérer les données du formulaire et les convertir
        lis = np.array([
            int(request.GET['gender']),
            float(request.GET['age']),
            float(request.GET['height']),
            float(request.GET['weight']),
            int(request.GET['family_history']),
            int(request.GET['caec']),
            int(request.GET['faf']),
            int(request.GET['calc']),
            int(request.GET['mtrans']),
        ]).reshape(1, -1)  # Convertir en array NumPy et reshaper

        # Convertir la liste en un tableau 2D pour le scaler
        X_new_scaled = scaler.transform(lis) 
        
        # Créer un explainer SHAP pour CatBoost
        explainer = shap.Explainer(cls)
        
        # Obtenir l'explication SHAP pour la nouvelle donnée
        shap_values = explainer(X_new_scaled)
        feature_names = ['gender', 'age', 'height', 'weight', 'family_history', 'caec', 'faf', 'calc', 'mtrans']
        shap_dict = {feature_names[i]: shap_values.values[0][i] for i in range(len(feature_names))}
        
        # Obtenir les probabilités de chaque classe
        proba = cls.predict_proba(X_new_scaled)[0]  # [0] pour récupérer les valeurs sous forme de liste
        
        # Associer chaque classe avec sa probabilité
        class_probabilities = {key: round(proba[value] * 100, 2) for key, value in NObeyesdad.items()}

        print("Données normalisées :", X_new_scaled)
        print("Impact des features sur la prédiction :", shap_dict)
        print("Probabilités des classes :", class_probabilities)
        
        # Faire la prédiction
        ans = cls.predict(X_new_scaled)
        predicted_value = int(ans[0]) if isinstance(ans[0], (int, float)) else ans[0]

        print("Valeur prédite :", predicted_value)

        # Trouver le type d'obésité correspondant
        obesity_type = next((clef for clef, value in NObeyesdad.items() if value == predicted_value), "Inconnu")

        return render(request, "result.html", {
            "obesity_type": obesity_type, 
            "shap_values": shap_dict,
            "class_probabilities": class_probabilities
        })

    except Exception as e:
        print(f"Erreur détectée : {e}")
        return render(request, "waiting.html")