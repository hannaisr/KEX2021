#### 2021-02-16
Försöker hitta ett sätt att göra bilderna till ett rimligt dataset.

#### 2021-02-17
Fortsättning på samma arbete som igår. Klar med YouTube-tutorial. Upptäcker problem med att csv inte sparar typen på värdena, utan gör om allt till sträng. Påbörjar fundering kring hur det bäst sparas istället. Pickle?

Diskussion kring hur bilderna ska roteras. Vad vill vi att testdata ska vara? Hanna röstar för bilder roterade med slumpmässig vinklel, Andreas röstar för bilder roterade med nx90$^\circ$ vinkel.

#### 2021-02-18
Gjort om så att datan sparas som pickle.
Gör en enkel random forest-klassificerare som ska kunna avgöra vä eller hö hand på personen.

Inget fungerar. Försöker analysera hur CIFAR-datasetten är sparade för att kunna utnyttja det.
