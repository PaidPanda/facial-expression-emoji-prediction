from tensorflow.keras.models import load_model

model = load_model('my_emotion_recognizer.h5')        
model.save('my_emotion_recognizer_best_69.4%.keras') 

print("Converted successfully! Now use the .keras file")