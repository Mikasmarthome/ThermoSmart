# ThermoSmart

**KI-gestützte, wetterbewusste Heizungssteuerung für Home Assistant**

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://hacs.xyz)
[![Version](https://img.shields.io/badge/version-v0.2.5b-blue.svg)](https://github.com/Mikasmarthome/ThermoSmart/releases)
[![HA min](https://img.shields.io/badge/HA-2024.1%2B-brightgreen.svg)](https://www.home-assistant.io)
[![License](https://img.shields.io/github/license/Mikasmarthome/ThermoSmart)](LICENSE)

ThermoSmart ist eine Custom Integration für Home Assistant, die deine Heizkörperthermostate (TRVs) intelligent steuert. Anders als einfache Zeitplan-Thermostate lernt ThermoSmart wie dein Haus heizt, passt sich an Außenbedingungen an und optimiert den Energieverbrauch automatisch.

---

## Funktionen

### Intelligente TRV-Steuerung
- **Ventil-Boost** – Bei kalten Räumen sendet ThermoSmart einen höheren Sollwert ans TRV damit das Ventil weiter öffnet und schneller heizt. Der Boost-Faktor wird automatisch gelernt und bei Überschießen reduziert.
- **Multi-Faktor Boost** – Außentemperatur, Windgeschwindigkeit und Außenluftfeuchtigkeit beeinflussen wie stark das Ventil geöffnet wird.
- **Lokale TRV-Kalibrierung** – ThermoSmart schreibt automatisch den korrekten Offset in `local_temperature_calibration`. TRVs messen oft die Heizkörpertemperatur statt die Raumtemperatur – das wird damit behoben.
- **Parallele Steuerung** – Mehrere TRVs in einer Zone werden gleichzeitig angesteuert.

### Lernalgorithmus
- **Zeitbasiertes Lernen** – Das System lernt wann welche Temperatur gewünscht wird (Wochentag, Uhrzeit).
- **Heizrate lernen** – Wie schnell heizt dein Haus unter verschiedenen Außenbedingungen? ThermoSmart misst und merkt es sich.
- **Abkühlrate lernen** – Wie schnell kühlt das Haus ab? Verbessert die Vorheizzeit-Berechnung.
- **Konfidenz** – Solange zu wenig Daten vorhanden sind, werden sichere Standardwerte verwendet.

### Wetterintegration
- **Wetterkorrektur** – Bei Kälte wird die Zieltemperatur leicht erhöht, bei Wärme reduziert.
- **Prognose-Unterdrückung** – Wird es heute warm genug? Dann heizt ThermoSmart weniger oder gar nicht.
- **Wind & Sonne** – Windchill und Sonneneinstrahlung beeinflussen die Heizleistung.
- **Sommer-Modus** – Automatische Erkennung über 72-Stunden-Außentemperatur-Durchschnitt. Über 18°C Schnitt: Heizung auf Frostschutz.

### Präsenz & Komfort
- **Fenster-Delay** – Heizung läuft noch kurz weiter wenn ein Fenster geöffnet wird (kein falsches Abschalten bei kurzem Lüften).
- **Sofort-Reaktion** – ThermoSmart reagiert sofort wenn sich Fenster, Personen oder Urlaubsmodus ändern.
- **Präsenzerkennung** – Konfigurierbare Personen-Entities für automatische Abwesenheitssteuerung.

### TRV-Quirks (Sonoff TRVZB & andere)
- **Interne Logik deaktivieren** – Schalter wie `switch.*_window_detection` werden automatisch deaktiviert damit das TRV nicht mit ThermoSmart konkurriert.
- **Ventil-Wartung** – Jeden Sonntag um 03:00 Uhr öffnet ThermoSmart das Ventil kurz vollständig und schließt es wieder. Verhindert Festklemmen nach dem Sommer.

---

## Installation über HACS

1. **HACS öffnen** → Integrationen → drei Punkte oben rechts → **Benutzerdefinierte Repositories**
2. URL eingeben: `https://github.com/Mikasmarthome/ThermoSmart`
3. Kategorie: **Integration** → Hinzufügen
4. ThermoSmart in der Liste suchen und **Installieren**
5. Home Assistant neu starten
6. **Einstellungen → Integrationen → Integration hinzufügen → ThermoSmart**

---

## Konfiguration

Pro Heizzone wird ein eigener Eintrag angelegt. Alle Felder sind über die UI konfigurierbar.

| Feld | Beschreibung |
|------|-------------|
| **Thermostate / TRVs** | Climate-Entities deiner Ventile (Pflichtfeld) |
| **TRV-Kalibrierung** | `number.*_local_temperature_calibration` Entities |
| **TRV-Eigenlogik deaktivieren** | Schalter die ThermoSmart auf AUS hält (z.B. `switch.*_window_detection`) |
| **Innensensoren** | Raumtemperatursensoren (Durchschnitt wird berechnet) |
| **Fenstersensoren** | Binary Sensors für automatisches Abschalten |
| **Personen** | Person-Entities für Präsenzerkennung |
| **Urlaubsmodus** | `input_boolean` für Urlaub |
| **Wetter-Entity** | HA Wetter-Entity für Prognosen |
| **Außensensoren** | Eigene Wetterstation (optional, wird bevorzugt) |
| **Ventil-Wartung** | Wöchentliche Ventilübung an/aus |

### Entities pro Zone

| Entity | Beschreibung |
|--------|-------------|
| `climate.*` | Virtuelle Climate-Entity – zeigt Modus, Ist/Soll-Temp |
| `sensor.*_zieltemperatur` | Berechnete Zieltemperatur |
| `sensor.*_trv_setpoint` | Tatsächlicher Wert der ans TRV gesendet wird (inkl. Boost) |
| `sensor.*_vorheizzeit` | Errechnete Vorheizzeit in Minuten |
| `sensor.*_konfidenz` | Lernfortschritt in Prozent |
| `sensor.*_wetterkorrektur` | Aktueller Temperatur-Offset durch Außenbedingungen |
| `sensor.*_status` | Übersichtsstatus (Heizt / Temperatur gehalten / Sommer / ...) |
| `switch.*_aktive_steuerung` | ThermoSmart aktiv oder Beobachtungsmodus |
| `switch.*_lernmodus` | Lernalgorithmus ein/aus |
| `select.*_heizmodus` | Auto / Komfort / Nacht / Abwesend / Urlaub |
| `number.*_override` | Manueller Temperatur-Override |

---


## Mithelfen & Contributing

Beiträge sind herzlich willkommen! So kannst du helfen:

### Bugs melden
→ [GitHub Issues](https://github.com/Mikasmarthome/ThermoSmart/issues/new) öffnen mit:
- HA-Version und ThermoSmart-Version
- Beschreibung was passiert ist
- Relevante Zeilen aus dem HA-Log (`Einstellungen → System → Protokoll`)

### Features vorschlagen
→ [GitHub Issues](https://github.com/Mikasmarthome/ThermoSmart/issues/new) mit Label `enhancement`

### Code beitragen (Pull Requests)
1. Repository **forken** (oben rechts auf GitHub)
2. Feature-Branch erstellen: `git checkout -b feature/mein-feature`
3. Änderungen committen: `git commit -m "feat: mein feature"`
4. Branch pushen: `git push origin feature/mein-feature`
5. **Pull Request** auf GitHub öffnen

#### Entwicklungsumgebung einrichten
```bash
git clone https://github.com/Mikasmarthome/ThermoSmart.git
# Dateien aus custom_components/thermosmart/ in dein HA config/custom_components/ kopieren
# HA neu starten
```

#### Was besonders gesucht wird
- **Tests** mit echten TRV-Modellen (Sonoff, Danfoss, Eurotronic, ...)
- **Übersetzungen** – `custom_components/thermosmart/translations/` – einfach neue Sprachdatei anlegen
- **TPI-Regler** Implementierung
- **Device Quirks** für weitere TRV-Modelle

### Fragen & Diskussion
→ [GitHub Discussions](https://github.com/Mikasmarthome/ThermoSmart/discussions) – für allgemeine Fragen, Ideen und Austausch

---

## Lizenz

MIT License – siehe [LICENSE](LICENSE)
