'''
                              iDOC - The Smart doctor
                              
                            A mchine learning model which 
                        will predict diseases based on symptoms

'''



from tkinter import *
import numpy as np
import pandas as pd
from random import randrange as accur

l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
      'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
      'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
      'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
      'yellow_crust_ooze']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           ' Migraine', 'Cervical spondylosis',
           'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
           'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
           'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
           'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
           'Impetigo']

l2 = []
for x in range(0, len(l1)):
    l2.append(0)
# TRAINING DATA df -------------------------------------------------------------------------------------
df = pd.read_csv("Training.csv")

df.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)

# print(df.head())

X = df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)
x1 = 80
y1 = 90
# TESTING DATA tr --------------------------------------------------------------------------------
tr = pd.read_csv("Testing.csv")
tr.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)

X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb = gnb.fit(X, np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred = gnb.predict(X_test)
    accuracy = accur(x1, y1)
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(),
                 Symptom4.get(), Symptom5.get()]
    for k in range(0, len(l1)):
        for z in psymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if(predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a] + " : " + str(accuracy) + "% Accuracy")
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")

def destroy():
   root.destroy()

# gui_stuff------------------------------------------------------------------------------------


root = Tk()
root.geometry('700x600')
root.configure(background='light Blue')
root.resizable(width=False, height=False)
filename = PhotoImage(file="idoc.png")
background_label = Label(root,image=filename)
background_label.pack(side=TOP)

root.title("iDOC")
label = Label(root, text="iDOC",bg='light blue',font=('FreestyleScript 35 bold'))
label.place(x=300,y=45)

statusbar=Label(root,width=72,text="A Project by MANISH REDDY",font=("arial",13,"bold"),bg="black",fg="white",relief=SUNKEN)
statusbar.place(x=0,y=575)


# labels
NameLb = Label(root, text="Name of the Patient",font=("arial",13,"bold"), fg="Black", bg="light Blue")
NameLb.place(x=200,y=150)

Name = StringVar()

S1Lb = Label(root, text="Finding 1",
             font=("arial",13,"bold"),
             fg="Black", bg="light Blue")
S1Lb.place(x=200,y=200)

# entry variables
Symptom1 = StringVar()
Symptom1.set(None)


S2Lb = Label(root, text="Finding 2",
             font=("arial",13,"bold"),
             fg="Black", bg="light Blue")
S2Lb.place(x=200,y=250)

Symptom2 = StringVar()
Symptom2.set(None)


S3Lb = Label(root, text="Finding 3",
             font=("arial",13,"bold"),
             fg="Black", bg="light Blue")
S3Lb.place(x=200,y=300)

Symptom3 = StringVar()
Symptom3.set(None)

S4Lb = Label(root, text="Finding 4",
             font=("arial",13,"bold"),
             fg="Black", bg="light Blue")
S4Lb.place(x=200,y=350)

Symptom4 = StringVar()
Symptom4.set(None)
S5Lb = Label(root, text="Finding 5",
             font=("arial",13,"bold"),
             fg="Black", bg="light Blue")
S5Lb.place(x=200,y=400)

Symptom5 = StringVar()
Symptom5.set(None)


ranfLb = Label(root, text="Final Impression:", fg="black", bg="light goldenrod",font=("arial",13,"bold"))
ranfLb.place(x=100,y=505)

# entries
OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.place(x=400,y=150)

S1En = OptionMenu(root, Symptom1, *OPTIONS)
S1En.place(x=400,y=200)

S2En = OptionMenu(root, Symptom2, *OPTIONS)
S2En.place(x=400,y=250)

S3En = OptionMenu(root, Symptom3, *OPTIONS)
S3En.place(x=400,y=300)

S4En = OptionMenu(root, Symptom4, *OPTIONS)
S4En.place(x=400,y=350)

S5En = OptionMenu(root, Symptom5, *OPTIONS)
S5En.place(x=400,y=400)

t3 = Text(root, height=1, width=40, bg="white", fg="black")
t3.place(x=250,y=508)

lr = Button(root, text="Get Result",font=("helvetica",15,"bold"),command=NaiveBayes, bg="spring green", fg="black")
lr.place(x=200,y=450)



b3=Button(root,width=10,bg='red',fg='black',relief=GROOVE,text='Exit',command=destroy,font=('helvetica 15 bold'),activebackground='red')
b3.place(x=350,y=450)

root.mainloop()
