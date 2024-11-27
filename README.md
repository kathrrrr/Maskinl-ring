---

## **Superviseret maskinlæring med Iris-datasættet**

### **Mål**
1. Forstå superviseret læring og klassifikation.
2. Træne og evaluere simple maskinlærings-modeller.
3. Visualisere resultaterne for bedre forståelse.

---

### **Trin 1: Introduktion til Iris-datasættet**
**Formål**: Lær, hvad datasættet indeholder.

Kode:
```python
from sklearn.datasets import load_iris
import pandas as pd

# Indlæs datasættet
iris = load_iris()

# Konverter til en pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['target_name'] = iris_df['target'].apply(lambda x: iris.target_names[x])

# Vis de første rækker i datasættet
print(iris_df.head())
```

**Læg mærke til**:
- Datasættet indeholder data om tre typer iris-blomster: Setosa, Versicolor og Virginica.
- Egenskaber: `sepal length`, `sepal width`, `petal length`, `petal width`.
- Mål: Klassificere blomsten ud fra disse egenskaber.

---

### **Trin 2: Visualisering af data**
**Formål**: Udforsk data ved hjælp af plots.

Kode:
```python
import matplotlib.pyplot as plt

# Scatterplot for to funktioner
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Scatterplot of Iris Data')
plt.show()
```

**Opgave**:
- Brug forskellige par af egenskaber for at undersøge datafordelingen.
- Hvordan ser de tre klasser ud i plottet?

---

### **Trin 3: Opdeling af data i trænings- og testdata**
**Formål**: Forstå konceptet med at dele data i træning og test.

Kode:
```python
from sklearn.model_selection import train_test_split

# Opdel data
# Stratificeret opdeling
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
)



print(f"Antal træningsdata: {len(X_train)}")
print(f"Antal testdata: {len(X_test)}")

```

**Opgave**:
- Hvorfor deler vi data?
- Hvad betyder `test_size`?

---

### **Trin 4: Træning af en simpel klassifikationsmodel**
**Formål**: Lær at træne en superviseret klassifikationsmodel (Logistisk Regression).

Kode:
```python
from sklearn.linear_model import LogisticRegression

# Træn modellen
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Forudsig med testdata
y_pred = model.predict(X_test)

print("Forudsigelser:", y_pred)
```

**Opgave**:
- Hvad betyder "træning" af en model?
  

---

### **Trin 5: Evaluering af modellen**
**Formål**: Lær at måle modellens nøjagtighed og præstation.

Kode:
```python
from sklearn.metrics import accuracy_score, classification_report

# Evaluer nøjagtighed
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelens nøjagtighed: {accuracy:.2f}")

# Klassifikationsrapport
print("Klassifikationsrapport:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Opgave**:
- Hvad fortæller nøjagtighed?
- Hvad kan vi lære fra en klassifikationsrapport?

---

### **Trin 6: Visualisering af resultater**
**Formål**: Lav en simpel visualisering af modellens præstation.

Kode:
```python
import numpy as np

# Lav en confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names, cmap='viridis')
plt.title("Confusion Matrix")
plt.show()
```

**Opgave**:
- Hvad viser en confusion matrix?
- Hvor ser modellen ud til at fejle?

---

### **Trin 7: Prøv en anden model**
**Formål**: Eksperimentér med en ny algoritme, K-Nearest Neighbors (KNN).

Kode:
```python
from sklearn.neighbors import KNeighborsClassifier

# Træn KNN-modellen
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Forudsig og evaluer
y_pred_knn = knn_model.predict(X_test)
print(f"KNN-modelens nøjagtighed: {accuracy_score(y_test, y_pred_knn):.2f}")
```

**Opgave**:
- Hvordan fungerer KNN?
- Hvordan kan vi vælge `n_neighbors`?
- Lav en konfusionsmatrix for knn-modellen.

---

### **Opsummering**
1. Introduktion til Iris-datasættet.
2. Visualisering af data.
3. Træning og evaluering af supervised modeller.
4. Eksperimenter med en anden model.

