import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Assurez-vous que l'importation de 'set_background' est correcte
from util import set_background

set_background('./image.jpeg')
st.markdown(
    """
    <style>
    body {
        background-color: #E6F7FF;  /* Vous pouvez changer cette couleur à celle que vous préférez */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Modèle capable de prédire la présence du cancer du sein")
st.markdown("Cette application utilise un modèle de deep learning pour prédire la présence du cancer du sein sur une photo")

# Création d'un bouton de téléchargement
uploaded_file = st.file_uploader("Télécharger une image", type=['jpg', 'png', 'jpeg'])

# Vérifier si un fichier a été téléchargé
if uploaded_file is not None:
    # Afficher l'image téléchargée
    st.image(uploaded_file, use_column_width=True)
    
    # Chargement du modèle sauvegardé
    try:
        model = load_model('./best_model.h5')  # Assurez-vous d'adapter le chemin au modèle sauvegardé au format Keras
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

    # Définir une fonction pour effectuer des prédictions sur une nouvelle image
    def predict_image(img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            predictions = model.predict(img_array)
            
            # Interpréter les résultats
            confidence = predictions[0][0] * 100  # En pourcentage
            if predictions[0][0] < 0.5:
                result = f"Pas de cancer à {confidence:.1f}%."
            else:
                result = f"Présence de cancer {confidence:.1f}%."
            
            st.markdown(f"<p style='font-size: 24px; color: green;'>{result}</p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

    # Bouton de prédiction
    if st.button("Prédire"):
        # Appelez la fonction predict_image avec le chemin de l'image à prédire
        predict_image(uploaded_file)
