#### 2021-02-16
Försöker hitta ett sätt att göra bilderna till ett rimligt dataset.

#### 2021-02-17
Fortsättning på samma arbete som igår. Klar med YouTube-tutorial. Upptäcker problem med att csv inte sparar typen på värdena, utan gör om allt till sträng. Påbörjar fundering kring hur det bäst sparas istället. Pickle?

Diskussion kring hur bilderna ska roteras. Vad vill vi att testdata ska vara? Hanna röstar för bilder roterade med slumpmässig vinklel, Andreas röstar för bilder roterade med nx90&deg; vinkel.

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

#### 2021-02-24
Testat att identifiera 10 personer med 30 rotationer/förskjutningar per avtryck. Använde bara ett fingeravtryck (höger tumme) per person och fick en korrekthetsfaktor (?) på 0,7. Antar att detta inte är helt bra. Testade med 100 och fick faktorn 0.53, dvs inte alls bättre. 1000 rotationer per bild gav faktorn 0,88, så först här börjar lite resultat ses. 10000 funkar inte. Funkar inte att lagra så mycket i en lista.

Cross validation, check score for training.
Create curve for finding optimal number of trees.

#### 2021-02-25
Läser på om cross validation, https://scikit-learn.org/stable/modules/cross_validation.html.

Testar att använda 100 rot för att identifiera 10 pers. Får utskriften

"0.66 accuracy with a standard deviation of 0.03"

#### 2021-03-01
##### Möte:
<ul>
  <li>Testa vad effekten vlir av att bara ha svart och vitt, inte gråskala, på bilderna. </li>
  <li>Blir DNN bättre om man har lika mycket av varje kategori?</li>
</ul>

#### 2021-03-03
Söker information om metoderna validation_curve och learning_curve. Planerar att implementera det i våra beräkningar, för att ta reda på mer precis hur mycket utökning av datasetten som behövs (och hur väl det fungerar) samt för att få fram optimala "hyper parameters".

##### Möte med Linghui:
Try different dataset (mnist). För att verifiera metoden med att rotera bilderna.

Andreas:
<ul>
  <li> svårt med confusion matrix</li>
  <li> problem med pickle (DataSet)</li>
  <li> fortfarande problem med att koppla sanna "labeln" till rätt bild </li>
  <li> Kommer testa DNN med Tensorflow, sedan scikit. </li>
</ul>

Testa på annan data, t.ex. mnist. Se om rotationerna funkar där.

#### 2021-03-08
Testar rotationsmetoden på mnist. Se fil mnist_test.ipynb

sigma rot | sigma move | n rots | n estimators| score 1 | score 2 | score 3 | score 4 | score 5 |
--- | --- | --- | --- | --- | --- | --- | --- | --- |
10 | dim/10 | 100 | 100 | 0.5368561426904485 | 0.5785797632938823 | 0.5736622770461743 | 0.5455909318219704 | 0.5465410901816969 |
20 | dim/10 | 100 | 100 | 0.5591265210868478 | 0.5716452742123688 | 0.540006667777963 | 0.5508918153025504 | 0.56209368228038 |
30 | dim/10 | 100 | 100 | 0.518403067177863 | 0.540006667777963 | 0.5348224704117353 | 0.4895315885980997 | 0.5327054509084848 |
40 | dim/10 | 100 | 100 | 0.4995832638773129 | 0.4977496249374896 | 0.5354392398733122 | 0.5176862810468411 | 0.5226871145190866 |
50 | dim/10 | 100 | 100 | 0.4825804300716786 | 0.4944990831805301 | 0.4912652108684781 | 0.4489414902483747 | 0.46512752125354223 |

Funkar helt klart bättre än slumpen. Kan det antas att fingeravtryck varierar mindre än siffror?

Resultatet blir bättre ju fler rotationer som används - positivt! Testar att ta medel-score och plotta i en figur.

<i>Mean of 5 iterations. Sigma_rot=20</i><br>
List of number of rotations:
[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]<br>
List of scores for those rotations:
[0.1932022003667278, 0.36534422403733957, 0.4460810135022504, 0.5098649774962494, 0.49379563260543424, 0.5250175029171529, 0.5375629271545257, 0.5452342057009502, 0.533478913152192, 0.5602567094515754, 0.5606334389064844, 0.5716852808801467, 0.5778396399399901, 0.5768828138023003, 0.58864144024004, 0.5925754292382064, 0.5920653442240373, 0.5977929654942491, 0.5879479913318887, 0.5953258876479414, 0.5946857809634938, 0.6003567261210202, 0.6011568594765795, 0.5987997999666611, 0.6001500250041675, 0.5977329554925821]

#### 2021-03-09
Lagt till plot med resultaten från mnist-testningen. Ändrat åsikt om resultaten - tycker det verkar lovande! <i>Rätt siffra kan förutsägas med 60% säkerhet vid 1000 träningsbilder per siffra.</i> Visa Linghui.

Börjar undersöka hur många träningsbilder som behövs för fingeravtrycken och minsta antal träd som ändå ger rimligt resultat. Försökte skapa en fil med 1000 rotationer per fingeravtryck (10 st), vilket ledde till att datorn kraschade. Oklart om datorn klarar ens 1000. <i>Hur hanterar vi det här?</i>

Frågor:
<ul>
  <li>Kan vi hänvisa till (möjligt oseriösa) hemsidor som stöttat med idéer till koden i rapporten?</li>
  <li>Direktidentifiering tar mycket tid. Kan vi hoppa över det steget och gå direkt till flersteg, när vi sett att det funkar för en delmängd?</li>
</ul>

#### 2021-03-11
Vet inte vad jag kan göra. Väntar på Andreas bildbeskärningar.

#### 2021-03-17
##### Möte
Skapa graf för antal rotationer etc.
Jämför olika dataset för att ta reda på vad som är en godtagbar noggrannhet. MNIST säger t.ex. 60%.

#### 2021-03-23
Testade på mnist med "vanlig" slags random forest och fick en noggrannhet på 96,5%. Något bättre än vår metod, kan sägas.

#### 2021-03-24
Försöker skapa en graf för cv-score mot antal bilder per fingeravtryck. Går inte bra. Krasch som vanligt (Simple Random Forest.ipynb)

Går över till att bara köra 10 pers, ett finger. Funkar. Frågan är hur giltigt det är... Vet inte vad man ska tolka ur grafen över mean cval score vs number of images per fingerprint (finns i mappen figures).

#### 2021-03-25
100-200 rotations per fingerprint. Good enough according to the graph, or something. Test the performance for different number of fingerprints. Is multistage more successful?

#### 2021-03-28
Utkast till 2-stegs färdig. Finns i filen multi_stage_rf. Behöver modifieras lite för att vara användarvändlig, men funkar i alla fall som det är nu.

Användarvänligheten fixad. Ger dålig noggrannhet (som väntat).
