# NLP-Project
Check out our application: https://bila9630-nlp-project-applicationstreamlit-app-w1i1j6.streamlit.app/

## project description

Idee: Inhaltliche Klassifikation von Zeitungsartikeln - Zeitungsartikel sollen automatisch je nach Titeln und kurzer Beschreibung in Kategorien eingeteilt werden.
<br>Dabei werden verschiedene ML Modelle getestet, evaluiert und auf die Kompatibilität mit dem Anwendungsfall überprüft.
<br>Für die Modelle nutzen wir einen Datensatz von Kaggle. Dieser enthält etwa 210.000 Daten. Der Datensatz besteht aus Zeitungsartikel Überschriften und kurze Beschreibungen in englischer Sprache. Die Daten sind aus den Jahren 2012 – 2022.
<br>Wenn das für unseren Anwendungsfall optimale Modell gefunden wurde, werden wir eine Anwenderoberfläche mit dem Framework Streamlit entwickeln. Der Benutzer soll einen Titel und (wenn vorhanden) eine kurze Beschreibung oder das Abstract des Artikels eingeben. Daraufhin gibt das Modell gibt eine Kategorie als Antwort zurück. Darüber hinaus werden wir die Anwendung deployen, damit jeder unsere Anwendung nutzen kann.
<br>Gruppe: Hannah Schult, Viet Duc Kieu, Marvin Spurk, Caroline Schmidt, Sofie Pischl


## application
### how to start locally
```
cd application
pip install -r requirements.txt
streamlit run streamlit_app.py
```
application is now running on http://localhost:8501

to freeze the requirements:
```
pip freeze > requirements.txt
```

to create a virtual environment:
```
# create a virtual environment
virtualenv env
# activate the virtual environment
env\Scripts\activate
```


## data
dataset: https://www.kaggle.com/datasets/rmisra/news-category-dataset