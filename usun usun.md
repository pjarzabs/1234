# INSTRUKCJA (CW_5)

Program realizuje kompresję i dekompresję plików tekstowych przy użyciu
algorytmu kodowania Huffmana.

------------------------------------------------------------------------------

## Wymagania:

- Java JDK 17 lub nowsze
- Apache Maven 3.3.0 +

## Kompilacja:

* Najpierw należy otworzyć projekt w NetBeans (File -> Open Project)
* Upewnić się, że używany jest minimum JDK 17.
* Następnie kliknąć `Clean and Build Project`.

Po poprawnej kompilacji w katalogu:
`CW_5\AiSD2025ZEx5\target`

pojawi się plik:
`AiSD2025ZEx5-1.0.jar`

Opcjonalna kompilacja z linii poleceń:
W katalogu głównym projektu (tam gdzie jest `pom.xml`) należy uruchomić komendę w lini poleceń (musi być zainstalowany *Apache Maven*!):
```powershell
mvn clean package
```
To także poprawnie skompiluje plik do `AiSD2025ZEx5-1.0.jar`.

Ten plik JAR jest już gotowy do uruchamiania z linii poleceń.

## Uruchamianie programu (wyłącznie **linia poleceń**):

Program uruchamia się wyłącznie przez `java -jar`, najlepiej w powershell'u, CMD (lub w terminalu bash na linuxie).

Przykładowe uruchomienie programu z lini poleceń:

```shell
java -jar AiSD2025ZEx5-1.0.jar -m comp -s input.txt -d output.comp -l 2
```

w którym argument:
* `-m` może przyjmować jedną z dwóch wartości: `comp`, `decomp` – w zależności od trybu działania programu, odpowiednio dla kompresji i dekompresji;
* `-s` ścieżka do pliku wejściowego (poddawanego kompresji lub dekompresji);
* `-d` ścieżka do pliku wyjściowego
* `-l` określa maksymalną długość łańcuchów, które powinny stanowić pojedynczą sekwencję odpowiadającą słowu kodowemu (np. `-l 2` oznacza, że zamiast wyznaczać słowa dla pojedynczych znaków, program będzie kodował pary znaków). Domyślna wartość tego argumentu to l=1.

## Komentarz odnośnie mojego programu:
* Użyłem `ISO-8859-1` (z biblioteki `java.nio.charset.StandardCharsets`), ponieważ jest to kodowanie 8-bitowe z mapowaniem 1:1 "bajt na znak", dzięki czemu mogę bezpiecznie traktować bajty pliku jako znaki w String, używać ich do porównań i haszowania, a następnie odzyskać dokładnie te same bajty bez utraty informacji (w przeciwieństwie do chociażby `UTF-8`).
[ISO-8859-1 (Wikipedia)](https://en.wikipedia.org/wiki/ISO/IEC_8859-1)


