# **Wyznaczanie liczby obiektów, klas, atrybutów, etc**


## NAJLEPSZA (i najprostsza) WERSJA:
```python
import numpy as np
import pandas as pd
import seaborn as sns

"""
# liczba obiektów (obserwacji)
d0.shape[0]

# liczba klas
d0['klasa'].nunique()

# liczba obiektów w każdej klasie
d0['klasa'].value_counts()

# liczba atrybutów/cech
d0.drop(columns=['klasa']).shape[1]

# liczba brakujących danych
d0.isna().sum().sum()
"""

print("Liczba obiektow:", d0.shape[0])
print("Liczba klas:", d0['klasa'].nunique())
print("Obiekty w klasach:")
print(d0['klasa'].value_counts())
print("Liczba atrybutow:", d0.shape[1] - 1)  # POD WARUNKIEM, że kolumna "klasa" faktycznie istnieje i jest jedną kolumną (dlatego - 1)
print("Liczba brakow danych:", d0.isna().sum().sum())


```
-----


## *Upośledzona* wersja
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('dane.csv')
d0 = df.drop(columns=['Unnamed: 0'])
d0.head()
# liczba obiektów
count_objs = d0.shape[0]
print("liczba obiektow:", count_objs)

# liczba klas
classes_count = d0.groupby('klasa').ngroups
print("liczba klas:", classes_count)

# liczba obiektów w każdej klasie
arr = [] 
classes = d0['klasa'].unique()
for i in range(0,len(classes)):
    arr.append(0)

for i in range(0,count_objs):
    for j in classes:
        if d0.loc[i,"klasa"] == j:
            arr[j] +=classes_count-1
print("liczba obiektow w kazdej klasie:", arr)

# liczba atrybutów
print("liczba atrybutow:", d0.shape[1] - 1) # "-1" jeżeli kolumna "klasa" ISTNIEJE (bo wtedy bierze ją niepotrzebnie)

# liczba danych brakujących
print("liczba danych brakujacych:", d0.isna().sum().sum())

# usunięcie danych brakujących
dna = d0.dropna()

#dane bez grup
data_ng = dna.iloc[:,:-1]
```
----





# **KORELACJA i "WZAJEMNE" WYKRESY**

Korelacja:
```python
corr = dna.corr()    # "corr" czyli correlation - korelacja 

plt.figure(figsize=(10, 10))    # przygotowuje miejsce na nasz wykresik, o podanym rozmiarze 10x10
sns.heatmap(corr, annot=True, cmap="coolwarm", square=True)
plt.show()

# annot=True --> wpisuje wartości wewnątrz kwadracików - teraz jest łatwiej i czytelnie
# square=True --> ładne kwadraty, zamiast "rozjechanych" prostokątów - estetyka!
```
----

* ile, wizualnie, możesz wyróżnic grup
* które atrybuty są bardziej, a które mniej istotne w kontekście uczenia maszynowego nadzorowanego i nienadzorowanego

*Wzajemne* wykresy:
```python
sns.pairplot(dna.iloc[:,:-1], kind="scatter")
plt.show()
```

# **UCZENIE NADZOROWANE**
## Różne klasyfikatory (Bayes, drzewo, NN, etc)

```python

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree



def podziel(df,proporcja):
    # dzieli macierz (ramkę) danych na zbiór uczacy i testowy
    # df - ramka danych; proporcja - proporcja podzialu (0-1)
    # zwraca słownik z kluczami:
    # opis_ucz/opis_test - macierz atrybutów opisujących zbioru uczącego/testowego
    # dec_ucz/dec_test - wektor wartosci atrybutu decyzyjnego zbioru uczącego/testowego
    # uwaga: atrybut opisujący jest zawsze na końcu (ostatnia kolumna ramki)
    
    df = df.dropna()
    
    opis_ucz, opis_test, dec_ucz, dec_test = train_test_split(df.iloc[:,0:-1], df.iloc[:,-1].astype('category').cat.codes, test_size=proporcja)  #, random_state=0)
    return {"opis_ucz":opis_ucz, "opis_test":opis_test, "dec_ucz":dec_ucz, "dec_test":dec_test}




def granice(model,dane,atr_x, atr_y,tytul,kontur = 1):
    # wyświetla granice decyzyjne
    # model - model klasyfikatora
    # dane - dane (słownik zwracany przez funkcje podziel)
    # atr_x/atr_y - artybut wyswietlany na osi x/y
    # tytul - wyświetlany tytuł wykresu
    # kontur - par. opcjonalny (=0 -> brak konturu)
    if (kontur == 1):    
        model.fit(np.array(dane["opis_ucz"].iloc[:,[atr_x,atr_y]]), np.array(dane["dec_ucz"]))
        x_min = min(dane["opis_ucz"].iloc[:, atr_x].min(),dane["opis_test"].iloc[:, atr_x].min())
        x_max = max(dane["opis_ucz"].iloc[:, atr_x].max(),dane["opis_test"].iloc[:, atr_x].max())
        y_min = min(dane["opis_ucz"].iloc[:, atr_y].min(),dane["opis_test"].iloc[:, atr_y].min())
        y_max = max(dane["opis_ucz"].iloc[:, atr_y].max(),dane["opis_test"].iloc[:, atr_y].max())
        rozst_x = x_max - x_min
        rozst_y = y_max - y_min
        x_min = x_min - 0.1*rozst_x
        x_max = x_max + 0.1*rozst_x
        y_min = y_min - 0.1*rozst_y
        y_max = y_max + 0.1*rozst_y       
        xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/150),
                     np.arange(y_min, y_max, (y_max-y_min)/150))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    plt.figure(dpi = 100)
    plt.title(tytul)
    if (kontur == 1):
        plt.contourf(xx, yy, Z, levels = 4, alpha=0.2)
    plt.scatter(dane["opis_ucz"].iloc[:, atr_x], dane["opis_ucz"].iloc[:, atr_y], c=dane["dec_ucz"], marker = '.')
    plt.scatter(dane["opis_test"].iloc[:, atr_x], dane["opis_test"].iloc[:, atr_y], c=dane["dec_test"], marker = 'x')


def weryfikuj(model, dane, atryb):
    model.fit(dane["opis_ucz"].iloc[:, atryb], dane["dec_ucz"])

    wynik_ucz  = model.predict(dane["opis_ucz"].iloc[:, atryb])
    wynik_test = model.predict(dane["opis_test"].iloc[:, atryb])

    mp_ucz = confusion_matrix(dane["dec_ucz"], wynik_ucz)
    print("macierz pomyłek - zbiór uczący, dokładność:", np.sum(np.diag(mp_ucz)) / np.sum(mp_ucz))
    print(model.score(dane["opis_ucz"].iloc[:, atryb], dane["dec_ucz"]))
    print(mp_ucz)

    mp_test = confusion_matrix(dane["dec_test"], wynik_test)
    print("macierz pomyłek - zbiór testowy, dokładność:", np.sum(np.diag(mp_test)) / np.sum(mp_test))
    print(model.score(dane["opis_test"].iloc[:, atryb], dane["dec_test"]))
    print(mp_test)


nazwa_pliku = 'dane.csv'

# wczytanie badanego zbioru danych
df = pd.read_csv(nazwa_pliku)

# podział zbioru danych
dane = podziel(df,0.3)
print('Liczba obiektów zbioru uczącego: ', len(dane["opis_ucz"]))
print('Liczba obiektów zbioru testowego: ', len(dane["opis_test"]))

# ---------- K-NAJBLIŻSZYCH SĄSIADÓW ----------

model = KNeighborsClassifier(n_neighbors=5)

# wybór atrybutów
ax, ay = 0,1

# granice dycyzyjne
granice(model,dane,ax,ay,"klasyfikator NN dla zbioru " + nazwa_pliku)

# weryfikacja
weryfikuj(model,dane,[ax,ay])



# ---------- BAYES ----------
model_bayes = GaussianNB()
granice(model_bayes, dp, 1, 3,f"Klasyfikator Bayesa dla x=1, y=3")
granice(model_bayes, dp, 3, 5,f"Klasyfikator Bayesa dla x=3, y=5")




# ---------- DRZEWO DECYZYJNE ----------
for g in [2,3,4,5,6]:
    drzewo = tree.DecisionTreeClassifier(max_depth=g)
    tekst = "drzewo o głębokości " + str(g) + " dla x=1, y=3"
    granice(drzewo ,dp,1,3,tekst)
    
for g in [2,3,4,5,6]:
    drzewo = tree.DecisionTreeClassifier(max_depth=g)
    tekst = "drzewo o głębokości " + str(g) + " dla x=3, y=5"
    granice(drzewo ,dp,3,5,tekst)


```



------
# **Uczenie NIEnadzorowane**

Grupowanie danych, w szczególności:

* oceń wizualnie wyniki grupowania,
* określ optymalną liczbę grup,
* porównaj wyniki grupowania z wartościami atrybutu decyzyjnego,
* wykonaj eksperymenty dla wszystkich oraz dla wybranych atrybtów - czy wybór atrykutów ma wpływ na wynik grupowania ?



```python
plt.title("Zbiór danych")
plt.scatter(data_ng['1'], data_ng['3'])
```
----


## Klasyfikator k-średnich (k-means) + analiza **SYLWETEK**:
```python
# -------- klasyfikator k-średnich + analiza sylwetek -------- 
def test_kMean(x, y):
    liczby_grup = [2, 3, 4, 5, 6, 7, 8]
    miara_jakosci = pd.Series()

    for ile_grup in liczby_grup :

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(15, 5)

        ax1.set_xlim([-0.2, 1]) # zakres zmian współczynnika sylwetki
        ax1.set_ylim([0, len(dna) + (ile_grup + 1) * 10]) # organizacja wydruku

        kmeans = KMeans(n_clusters = ile_grup, random_state = 0).fit(data_ng)
        miara_jakosci = pd.concat([miara_jakosci, pd.Series(kmeans.inertia_)])
        etykiety_grup = kmeans.fit_predict(dna[[str(x),str(y)]])

        # oblicza miary sylwetek dla każdek grupy
        sylwetka = silhouette_samples(dna[[str(x),str(y)]], etykiety_grup)   
        # średnia wartość sylwetki
        srednia_sylwetka = silhouette_score(dna[[str(x),str(y)]], etykiety_grup)

        pozycja_kreski = 10
        for i in range(ile_grup):
            # Zebranie wyników sylwetek do próbek należących do klastra i ich sortowanie
            sylwetka_w_grupie = sylwetka[etykiety_grup == i]
            sylwetka_w_grupie.sort()

            liczebnosc_grupy = sylwetka_w_grupie.shape[0]
            kolor = cm.tab10(float(i) / ile_grup)
            ax1.fill_betweenx(np.arange(pozycja_kreski, pozycja_kreski + liczebnosc_grupy), 0, sylwetka_w_grupie ,color = kolor)

            # Wyliczenie przesunięcia w pionie dla wykresu kolejnej grupy
            pozycja_kreski += liczebnosc_grupy + 10  # 10 dla kolejnej próbki

        ax1.set_title("Wykres sylwetek dla poszczegolnych liczb grup")
        ax1.set_xlabel("Wartosc sylwetek")
        ax1.set_ylabel("Etykiety grup")

        # Wyrysowanie wartości średniej sylwetki 
        ax1.axvline(x = srednia_sylwetka, color = "black", linestyle = "--")
        ax1.set_yticks([])  # Wyczyszczenie etykiety osi Y
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Drugi wykres będzie przedstawiał klastry
        kolory = cm.tab10(etykiety_grup.astype(float) / ile_grup)
        ax2.scatter(dna.iloc[:,x], dna.iloc[:,y], marker = '.', s = 30, lw = 0, alpha = 0.7,c = kolory)

        ax2.set_title("Wizualizacja grupowania danych")
        ax2.set_xlabel("Atrybut %d" % x)
        ax2.set_ylabel("Atrybut %d" % y)

        plt.suptitle(("Analiza sylwetki k = %d" % ile_grup),  fontweight = 'bold')
        plt.figtext(0.14, 0, ("Dla n = %d, średnia wartosc sylwetki wynosi: %.3f, Suma odleglosci od centroidow: %.2f"
                     % (ile_grup, srednia_sylwetka, kmeans.inertia_ ) ))
        plt.show()



    # ---------- Wykres Łokciowy ----------
    plt.plot(liczby_grup, miara_jakosci,'bo-')
    plt.title("Wykres lokciowy", fontsize = 14, fontweight = 'bold')
    plt.xlabel("Liczba grup")
    plt.ylabel("Suma odleglosci od centroidow")
    
test_kMean(1, 3)
```




----




## Grupowanie aglomeracyjne - TODO (W.I.P., nie wiem czy dobrze)

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage

def test_agglomerative(x, y):
    liczby_grup = [2, 3, 4, 5, 6, 7, 8]

    for ile_grup in liczby_grup:

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(15, 5)

        ax1.set_xlim([-0.2, 1])
        ax1.set_ylim([0, len(dna) + (ile_grup + 1) * 10])

        # -------- AGLOMERACYJNE GRUPOWANIE --------
        model = AgglomerativeClustering(
            n_clusters=ile_grup,
            linkage='ward'  # klasyczne, euklidesowe
        )

        etykiety_grup = model.fit_predict(dna[[str(x), str(y)]])

        # -------- ANALIZA SYLWETKI --------
        sylwetka = silhouette_samples(dna[[str(x), str(y)]], etykiety_grup)
        srednia_sylwetka = silhouette_score(dna[[str(x), str(y)]], etykiety_grup)

        pozycja_kreski = 10
        for i in range(ile_grup):
            sylwetka_w_grupie = sylwetka[etykiety_grup == i]
            sylwetka_w_grupie.sort()

            liczebnosc_grupy = sylwetka_w_grupie.shape[0]
            kolor = cm.tab10(float(i) / ile_grup)

            ax1.fill_betweenx(
                np.arange(pozycja_kreski, pozycja_kreski + liczebnosc_grupy),
                0,
                sylwetka_w_grupie,
                color=kolor
            )

            pozycja_kreski += liczebnosc_grupy + 10

        ax1.axvline(x=srednia_sylwetka, color="black", linestyle="--")
        ax1.set_title("Wykres sylwetek dla poszczegolnych liczb grup")
        ax1.set_xlabel("Wartosc sylwetek")
        ax1.set_ylabel("Etykiety grup")
        ax1.set_yticks([])

        # -------- WIZUALIZACJA --------
        kolory = cm.tab10(etykiety_grup.astype(float) / ile_grup)
        ax2.scatter(
            dna.iloc[:, x],
            dna.iloc[:, y],
            marker='.',
            s=30,
            alpha=0.7,
            c=kolory
        )

        ax2.set_title("Wizualizacja grupowania danych")
        ax2.set_xlabel("Atrybut %d" % x)
        ax2.set_ylabel("Atrybut %d" % y)

        plt.suptitle(
            "Analiza sylwetki (aglomeracyjne), k = %d" % ile_grup,
            fontweight='bold'
        )

        plt.figtext(
            0.14, 0,
            "Dla k = %d, srednia wartosc sylwetki: %.3f"
            % (ile_grup, srednia_sylwetka)
        )

        plt.show()


x = 0   # numer atrybutu
y = 1   # numer atrybutu

test_agglomerative(x, y)


```
----
### Dodatkowo: dendrogram:

```python
def rysuj_dendrogram(x, y):
    X = dna[[str(x), str(y)]].values

    # macierz połączeń (to jest "drzewo")
    Z = linkage(X, method='ward')

    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title("Dendrogram (aglomeracyjne, ward)")
    plt.xlabel("Obiekty")
    plt.ylabel("Odleglosc")
    plt.show()


x = 0   # numer atrybutu
y = 1   # numer atrybutu

rysuj_dendrogram(x, y)
test_agglomerative(x, y)


```

# **DODATKOWE WNIOSKI/PROTIPY** (pisemnie)

* przetrenowanie (overfitting, nadmierne dopasowanie, przeuczenie) w uczeniu NADZOROWANYM jest wtedy, kiedy funkcja czy klasyfikator **za bardzo** się dopasowuje, wtedy tworzą się jakieś dziwne "wysepki" na wykresie, etc... Dopasowanie jest za bardzo specyficzne do konkretnych danych, gdyby dać nowe dane to by źle działało.
* underfitting (niedouczenie, niedotrenowanie) w uczeniu NADZOROWANYM jest wtedy, kiedy klasyfikator jest za słaby i za bardzo generalizuje i np. grupuje dwie oddzielne grupy w jedną grupę (nie rozróżnia dwóch grup). Ogólnie zawsze wtedy, kiedy źle dopasowuje się do danych, np. bierze wszystko *"do jednego wora"*.
* ...






