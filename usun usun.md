# INSTRUKCJA do programu kodowanie huffmana (CW_5)

Program realizuje kompresję i dekompresję plików tekstowych przy użyciu
algorytmu kodowania Huffmana.

---

## Wymagania:

- Java JDK 17 lub nowsze
- Apache Maven 3.3.0 +

## Kompilacja:

* Otwórz projekt w NetBeans (File -> Open Project)
* Upewnij się, że używany jest minimum JDK 17.
* Kliknij Clean and Build Project.

Po poprawnej kompilacji w katalogu:

`CW_5\AiSD2025ZEx5\target`

pojawi się plik:

`AiSD2025ZEx5-1.0.jar`

Ten plik JAR jest już gotowy do uruchamiania z linii poleceń.

## Uruchamianie programu (wyłącznie **linia poleceń**):

Program uruchamia się wyłącznie przez `java -jar`, najlepiej w powershell'u, CMD (lub w terminalu bash na linuxie).

Przykładowe uruchomienie programu z lini poleceń:
`java -jar AiSD2025ZEx5.jar -m comp -s input.txt -d output.comp -l 2`
w którym argument:
* `-m` może przyjmować jedną z dwóch wartości: `comp`, `decomp` – w zależności od trybu działania programu, odpowiednio dla kompresji i dekompresji;
* `-s` ścieżka do pliku wejściowego (poddawanego kompresji lub dekompresji);
* `-d` ścieżka do pliku wyjściowego
* `-l` określa maksymalną długość łańcuchów, które powinny stanowić pojedynczą sekwencję odpowiadającą słowu kodowemu (np. `-l 2` oznacza, że zamiast wyznaczać słowa dla pojedynczych znaków, program będzie kodował pary znaków). Domyślna wartość tego argumentu to l=1.
