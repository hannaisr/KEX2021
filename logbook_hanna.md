#### 2021-02-16
Försöker hitta ett sätt att göra bilderna till ett rimligt dataset.

#### 2021-02-17
Fortsättning på samma arbete som igår. Klar med YouTube-tutorial. Upptäcker problem med att csv inte sparar typen på värdena, utan gör om allt till sträng. Påbörjar fundering kring hur det bäst sparas istället. Pickle?

Diskussion kring hur bilderna ska roteras. Vad vill vi att testdata ska vara? Hanna röstar för bilder roterade med slumpmässig vinklel, Andreas röstar för bilder roterade med nx90$^\circ$ vinkel.

#### 2021-02-18
Gjort om så att datan sparas som pickle.
Gör en enkel random forest-klassificerare som ska kunna avgöra vä eller hö hand på personen.

Inget fungerar. Försöker analysera hur CIFAR-datasetten är sparade för att kunna utnyttja det.

Möte med Linghui. Kommer fram till att csv är ett bra sätt att spara datan på. Slumpmässiga vinklar också bra, men kan behövas mer än 360 versioner av varje bild. Därför kanske bra att komplettera med förskjutningar i olika riktningar.

#### 2021-02-19
Lyckas importera datan och köra en enkel 'random forest classifier' på originaldatan. Funkar utmärkt för att förutspå vilken hand fingeravtrycket kommer ifrån! Intressant. Fungerar även bra på alla "altered"-fingeravtryck, men har inte testat våra rotationer.

Kön verkar vara en dålig kategoriseringsversion och vad gäller fingrar vore det kanske bättre att dela in i tre grupper: tumme, lillfinger och övriga tre mittenfingrar.

Bilder på confusion matrices laddas upp i en mapp i git.

Använde Pickle, trots allt.

Testar också med scikits MLPClassifier(), vilket inte ger några bra resultat alls. Vet dock inte hur denna funktion fungerar, så det kanske inte är så konstigt...

#### 2021-02-23
Skapat en extra fil för rotering av bilder, som sparar bilderna som .pkl och med alternativet att bara använda några få av personerna i databasen. Har även lagt till så att bilderna förskuts åt höger/vänster och upp/ned. Förskjutningen görs med normalfördelat antal pixlar. Rotationen kan väljas att göras uniformt eller normalfördelat.
