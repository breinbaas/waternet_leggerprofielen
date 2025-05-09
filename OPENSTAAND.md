## ArgGIS 

Momenteel wordt gebruik gemaakt van csv bestanden voor de volgende informatie;

* dijktafelhoogte
* maatgevend hoogwater
* onderhoudsdiepte
* IPO klasse

De invoer voor het bepalen van deze informatie bestaat uit de **dijktrajectcode** (bv A117) en **metrering** (bv 100). 

Deze aanvraag moet via de ArcGIS API lopen. De huidige manier is via de volgende functies;

```python
dth = dijktrajecten.get_by_code(dtcode).dth_2024_at(chainage) # geeft float terug
river_level = dijktrajecten.get_by_code(dtcode).mhw_2024_at(chainage) # geeft float terug
onderhoudsdiepte = dijktrajecten.get_by_code(dtcode).onderhoudsdiepte_at(chainage) # geeft float terug
ipo = ipo_search.get_ipo(dtcode, chainage) # geeft I, II, III, IV of V terug
```

Dit zou vervangen moeten worden door bv de volgende (pseudo)code;

```python
dth = arggis.get_dth_by_code_and_chainage(dtcode, chainage) # geeft float terug
river_level = arggis.get_river_level_by_code_and_chainage(dtcode, chainage) # geeft float terug
onderhoudsdiepte = arggis.get_onderhoudsdiepte_by_code_and_chainage(dtcode, chainage) # geeft float terug
ipo = arggis.get_ipo_by_code_and_chainage(dtcode, chainage) # geeft I, II, III, IV of V terug
```
 
## Berekeningen lezen en schrijven

De berekeningen staan momenteel lokaal, dit moet via Sharepoint gelezen én geschreven worden.

Voorbeeld van de huidige manier is;

```python
# zoek alle stix bestanden in de directory met de berekeningen
stix_files = case_insensitive_glob(Path(PATH_ALL_STIX_FILES) / MODELS_TO_RUN, ".stix")

# itereer over alle stix bestanden
for stix_file in stix_files:
    doe_al_het_werk()
``` 

Hierbij wordt in de opgegeven directory gezocht naar alle beschikbare stix bestanden. 

Sharepoint werkt als een netwerk protocol waarbij toegang nodig is tot een netwerk locatie. Dit is momenteel door security instellingen niet mogelijk. Wat gefixed moet worden is;
* het moet mogelijk zijn om de originele stix bestanden te kopieren naar een lokale directory
* het moet mogelijk zijn om gevonden oplossingen te kopieren naar een Sharepoint locatie (het gaat daarbij om .csv, .png en .stix bestanden)

De originele stix bestanden staan al op Sharepoint maar er is nog geen map voor de uitvoer van de gevonden leggerprofielen. 

## Dynamische profielen

Als uitgangspunten veranderen of er meer standaard berekeningen komen dan wijzigt het leggerprofiel. Dit kan onhandig zijn bij de vaststelling van de leggerprofielen. Er moet nagedacht worden over hiermee wordt omgegaan.

## Beperkingen script tov de handmatige manier

* Spencer berekeningen kunnen moeilijk geautomatiseerd worden, deze worden nu omgezet naar Bishop Brute Force berekeningen
* Er zijn situaties waarbij 'het ontgravingsbakje' van 2m diepte het pleistoceen raakt, in dit geval wordt de overgang naar de beschermingszone 10m voorbij het uittredepunt gelegd. 
* Het script kent maar 1 manier van 'versterken' en dat is door de helling van de taluds te verflauwen. Soms is een andere aanpak slimmer maar deze logica is niet geimplementeerd. 



