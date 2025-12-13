## Jak wyznaczać?
- łączna liczbę obiektów (obserwacji)
- liczbę klas
- liczbę obiektów w każdej klasie
- liczbę atrybutów/cech
- liczbę brakujących danych


```python
# liczba obiektów
d0.shape[0]

# liczba klas
d0['klasa'].nunique()

# liczba obiektów w każdej klasie
d0['klasa'].value_counts()

# liczba atrybutów
d0.drop(columns=['klasa']).shape[1]

# liczba brakujących danych
d0.isna().sum().sum()
