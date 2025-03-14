import joblib
from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    return render(request, "home.html")

def prediction(request):
    return render(request, "prediction.html")

def result(request):
    try:
        # Charger le modèle
        cls = joblib.load('catboost_model.sav')  # Assure-toi que c'est bien un modèle CatBoost

        # Charger le scaler utilisé lors de l'entraînement
        scaler = joblib.load('scaler.pkl')  # Remplace par le vrai nom du fichier scaler

        # Récupérer les données du formulaire et les convertir
        lis = [
            int(request.GET['gender']),
            float(request.GET['age']),
            float(request.GET['height']),
            float(request.GET['weight']),
            int(request.GET['family_history']),
            int(request.GET['caec']),
            int(request.GET['faf']),
            int(request.GET['calc']),
            int(request.GET['mtrans']),
        ]

        # Convertir la liste en un tableau 2D pour le scaler
        X_new_scaled = scaler.transform([lis])  # 🔥 Corrigé ici

        print("Données normalisées :", X_new_scaled)

        # Faire la prédiction
        ans = cls.predict(X_new_scaled)
        predicted_value = int(ans[0]) if isinstance(ans[0], (int, float)) else ans[0]

        print("Valeur prédite :", predicted_value)

        # Dictionnaire de correspondance
        NObeyesdad = {
            'Insufficient_Weight': 0,
            'Normal_Weight': 1,
            'Obesity_Type_I': 2,
            'Obesity_Type_II': 3,
            'Obesity_Type_III': 4,
            'Overweight_Level_I': 5,
            'Overweight_Level_II': 6
        }

        # Trouver le type d'obésité correspondant
        obesity_type = next((clef for clef, value in NObeyesdad.items() if value == predicted_value), "Inconnu")

        return render(request, "result.html", {"obesity_type": obesity_type})

    except Exception as e:
        print(f"Erreur détectée : {e}")
        return render(request, "waiting.html")
