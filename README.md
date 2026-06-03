# ThermoSmart

**KI-gestützte, wetterbewusste Heizungssteuerung für Home Assistant**

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://hacs.xyz)
[![Version](https://img.shields.io/badge/version-v0.2.8b-blue.svg)](https://github.com/Mikasmarthome/ThermoSmart/releases)
[![HA min](https://img.shields.io/badge/HA-2024.1%2B-brightgreen.svg)](https://www.home-assistant.io)
[![License](https://img.shields.io/github/license/Mikasmarthome/ThermoSmart)](LICENSE)

ThermoSmart ist eine Custom Integration für Home Assistant die deine Heizkörperthermostate (TRVs) intelligent steuert. Statt starren Zeitplänen lernt ThermoSmart wie dein Haus heizt und kühlt, reagiert auf Außenbedingungen, Wetterprognoisen und Präsenz – und optimiert so Komfort und Energieverbrauch vollautomatisch.

**Ein Config-Eintrag = Eine Heizzone.** Mehrere Zonen möglich, jede unabhängig konfigurierbar.

---

## Funktionen im Überblick

### Intelligente TRV-Steuerung

**Ventil-Boost-Setpoint**
ThermoSmart berechnet einen höheren Sollwert als die Zieltemperatur um das Ventil stärker zu öffnen und schneller zu heizen. Der Setpoint berücksichtigt:
- Temperaturdifferenz (Ziel – Ist)
- Außentemperatur (kälter = mehr Boost)
- Windgeschwindigkeit (Wind erhöht Wärmeverlust)
- Außenluftfeuchtigkeit (feuchte Kälte kühlt stärker)
- Gelernter Boost-Faktor (passt sich automatisch an)

**Restwärme-Kompensation**
Heizkörper strahlen nach dem Schließen des Ventils noch Wärme ab. ThermoSmart beginnt den Setpoint bereits 1,5°C vor Erreichen der Zieltemperatur zu reduzieren – das Ventil schließt früher und die Restwärme deckt den Rest. Verhindert Überschwingen ohne auf schlechte Erfahrungen warten zu müssen.

**Bidirektionaler Lernfaktor**
- Überschießt die Temperatur → Boost-Faktor wird reduziert (×0,92, Minimum 0,5)
- Heizt zu langsam (nach 30 min noch >1°C unter Ziel) → Boost-Faktor wird erhöht (×1,05, Maximum 2,0)

**Parallele TRV-Steuerung**
Alle TRVs einer Zone werden gleichzeitig angesteuert, nicht nacheinander.

**Sofort-Korrektur bei manuellen Änderungen**
Wird ein TRV manuell verstellt (z.B. direkt am Gerät), erkennt ThermoSmart das sofort und korrigiert den Setpoint ohne auf den nächsten 5-Minuten-Zyklus zu warten.

---

### Lokale TRV-Kalibrierung

TRVs messen oft die Heizkörper- statt die Raumtemperatur. ThermoSmart berechnet den Offset zwischen Raumsensor und TRV-Sensor und schreibt ihn automatisch in die `local_temperature_calibration`-Entity des TRVs.

- **Auto-Erkennung** via Device Registry – keine manuelle Konfiguration nötig
- **EMA-Glättung** (α=0,25) – verhindert Jitter durch kurzfristige Schwankungen
- Kalibrierung wird übersprungen wenn der Heizkörper gerade stark heizt (Sensor verfälscht)
- Plausibilitätsprüfung: Offsets >7°C werden ignoriert

---

### Lernalgorithmus (Multi-Faktor)

ThermoSmart lernt das thermische Verhalten deines Hauses über einen langen Zeitraum.

**Was gelernt wird:**
| Datenpunkt | Wozu |
|---|---|
| Zieltemperatur nach Uhrzeit/Wochentag | Zeitplan-Anpassung |
| Heizrate (°C/min) | Vorheizzeit-Berechnung |
| Abkühlrate (°C/min) | Realistische Vorheizzeit (Haus kühlt während Vorheizen weiter) |
| Alle Außenbedingungen (Temp, Wind, Solar, Feuchte) | Thermische Ähnlichkeit für Prädiktionen |

**Gewichtung:**
- Neuere Beobachtungen zählen mehr (Zeitgewichtung, Halbwertszeit 180 Tage)
- Ähnliche Außenbedingungen zählen mehr (Gauß-Ähnlichkeit pro Faktor)
- Jahreszeitliche Gewichtung: Dezember-Daten zählen im Winter mehr als Sommer-Daten

**Konfidenz:**
Solange zu wenig Daten vorhanden sind, werden sichere Standardwerte verwendet. Die Konfidenz steigt mit Datenmenge und -vielfalt (0–100%).

---

### Wetterintegration

**Temperaturkorrektur (aktuell)**
| Außentemperatur | Korrektur |
|---|---|
| < 0°C | +1,5°C |
| 0–10°C | +0,5°C |
| 10–18°C | ±0°C |
| > 18°C | −1,0°C |
| Wind > 10 m/s + kalt | Zusätzlich +0,5°C |
| Sonne > 400 W/m² | Bis −0,5°C Reduktion |

**Prognose-Unterdrückung**
Wird die vorhergesagte Tageshöchsttemperatur höher als die Zieltemperatur, heizt ThermoSmart weniger oder gar nicht – das Haus erwärmt sich ohnehin selbst.

**Sommer-Modus**
Automatische Erkennung über 72-Stunden-Rollmittelwert der Außentemperatur:
- Ø > 18°C → Sommer: Heizung auf Frostschutz (12°C)
- Ø < 15°C → Winter: Heizung aktiv

**Eigene Wetterstation**
Eigene Sensoren (Temperatur, Feuchte, Wind, Solar, Regen) haben Vorrang vor der HA-Wetter-Entity und werden bevorzugt verwendet.

---

### Präsenz & Modi

**7 Heizmodi** – wählbar über die Climate-Entity, Select-Entity oder automatisch:

| Modus | Temperatur | Aktivierung |
|---|---|---|
| **Auto** | Zeitplan + Lernalgorithmus | Standard |
| **Boost** | Konfigurierbar (Standard 24°C) | Manuell – schnelles Aufheizen |
| **Komfort** | Konfigurierbar (Standard 21°C) | Manuell |
| **Eco** | Konfigurierbar (Standard 19°C) | Manuell – energiesparend |
| **Nacht** | Konfigurierbar (Standard 18°C) | Automatisch nach Zeitplan |
| **Abwesend** | Konfigurierbar (Standard 17°C) | Automatisch wenn alle Personen weg |
| **Urlaub** | Konfigurierbar (Standard 12°C) | Manuell oder via Urlaubsschalter |

**Zeitplan**
Werktag und Wochenende getrennt konfigurierbar. ThermoSmart heizt automatisch früher vor damit die Komforttemperatur pünktlich erreicht ist.

**Override mit Auto-Reset**
Wird die Temperatur manuell überschrieben, gilt der Override bis der Zeitplan-Slot wechselt – dann kehrt ThermoSmart selbständig zur Automatik zurück.

**Präsenzerkennung**
Konfigurierbare Person-Entities; unterstützt benutzerdefinierte HA-Zonen als Heimzone.

---

### Fenstererkennung

- Konfigurierbare Verzögerung: Heizung schaltet erst nach X Minuten ab (kein Fehlalarm beim kurzen Lüften)
- Bei geöffnetem Fenster: TRVs werden aktiv auf 5°C gesetzt statt auf dem letzten Sollwert zu bleiben
- Konfigurierbare Schließ-Toleranz: Heizung startet erst Y Minuten nach Schließen (Raum hat sich erst abgekühlt)
- Sofort-Reaktion: Keine Wartezeit bis zum nächsten Zyklus

---

### TRV-Quirk-Management

Viele TRVs haben interne Logiken die mit ThermoSmart konkurrieren. ThermoSmart erkennt und deaktiviert diese automatisch:

| Quirk-Pattern | Gerät | Problem |
|---|---|---|
| `*_window_detection` | Sonoff TRVZB, Danfoss | TRV erkennt Fenster selbst → Konflikt mit ThermoSmart-Sensoren |
| `*_child_lock` | Viele | Sperrt externe Setpoint-Änderungen → ThermoSmart kann nichts setzen |
| `*_frost_protection` | Verschiedene | Interner Frostschutz kollidiert mit ThermoSmart |

Erkennung automatisch via Device Registry – zusätzlich manuelle Konfiguration weiterer Switches möglich.

---

### Wartung & Zuverlässigkeit

**Ventil-Wartung**
Jeden Sonntag um 03:00 Uhr öffnet ThermoSmart alle Ventile kurz vollständig (28°C, 30 Sekunden) und schließt sie wieder. Verhindert Festklemmen nach dem Sommer durch Kalk oder Gummi.

**TRV-Watchdog**
Thermostate die ungewollt auf `off` schalten werden automatisch auf `heat` zurückgestellt.

**TRV-Offline-Erkennung**
Offline-TRVs werden erkannt und geloggt. Sobald sie wieder erreichbar sind, wird normal weitervorgegangen.

**Sensor-Noise-Filter**
EMA-Glättung (α=0,2) und Spike-Erkennung (>4°C Abweichung vom Mittelwert) für alle Temperatursensoren. Fehlerhafte Einzelmessungen werden ignoriert.

---

## Installation über HACS

1. **HACS öffnen** → Integrationen → drei Punkte oben rechts → **Benutzerdefinierte Repositories**
2. URL eingeben: `https://github.com/Mikasmarthome/ThermoSmart`
3. Kategorie: **Integration** → Hinzufügen
4. ThermoSmart in der Liste suchen und **Installieren**
5. Home Assistant neu starten
6. **Einstellungen → Integrationen → Integration hinzufügen → ThermoSmart**

---

## Einrichtung (4-Schritt-Wizard)

### Schritt 1 – Geräte & Sensoren
| Feld | Beschreibung |
|---|---|
| **Zonenname** | Frei wählbar, z.B. "Wohnzimmer" |
| **Thermostate / TRVs** | Climate-Entities (Pflichtfeld) |
| **Temperatursensoren** | Raumsensoren – Durchschnitt wird berechnet |
| **Luftfeuchtigkeitssensoren** | Für Lernalgorithmus (optional) |
| **Fenstersensoren** | Binary Sensors (optional) |
| **Fenster: Heizung aus nach** | Verzögerung in Minuten (Standard: 5 min) |
| **Fenster: Heizung an nach** | Verzögerung beim Schließen (Standard: 2 min) |
| **Ventil-Wartung** | Wöchentliche Ventilübung ein/aus |
| **Frostschutz bei geöffnetem Fenster** | TRVs auf 5°C setzen statt auf letztem Wert lassen |

### Schritt 2 – Temperaturen & Zeitplan
| Feld | Standard |
|---|---|
| Komforttemperatur | 21°C |
| Nachttemperatur | 18°C |
| Abwesenheitstemperatur | 17°C |
| Urlaubstemperatur | 12°C |
| Boost-Temperatur | 24°C |
| Eco-Temperatur | 19°C |
| Temperaturtoleranz | 0,5°C |
| Werktag: Komfort ab | 06:00 |
| Werktag: Nacht ab | 22:00 |
| Wochenende: Komfort ab | 08:00 |
| Wochenende: Nacht ab | 23:00 |

### Schritt 3 – Präsenz & Automatik
| Feld | Beschreibung |
|---|---|
| **Personen** | Person-Entities für automatischen Abwesenheitsmodus |
| **Heimzone** | Welche Zone gilt als "zuhause" (Standard: zone.home) |
| **Urlaubsschalter** | Beliebige Entity die auf `on` wechselt (input_boolean, binary_sensor, calendar …) |
| **Lernalgorithmus** | Ein/aus |

### Schritt 4 – Wetter & Außensensoren
| Feld | Beschreibung |
|---|---|
| **Wetter-Entity** | HA Wetter-Entity für Prognosen (z.B. weather.home) |
| **Außentemperatur** | Eigene Wetterstation – hat Vorrang vor Wetter-Entity |
| **Außenluftfeuchtigkeit** | Für präzisere Wetterkorrektur |
| **Windgeschwindigkeit** | m/s oder km/h |
| **Sonneneinstrahlung** | W/m² – für Solar-Kompensation |
| **Niederschlag** | Optional |

---

## Entities pro Zone

### Steuerung
| Entity | Typ | Beschreibung |
|---|---|---|
| `climate.thermosmart_*` | Climate | Virtuelle Thermostat-Entity – Ziel/Ist-Temp, Modus, Presets |
| `select.*_heizmodus` | Select | Auto / Boost / Komfort / Eco / Nacht / Abwesend / Urlaub |
| `switch.*_aktive_steuerung` | Switch | ThermoSmart aktiv (AN) oder Beobachtungsmodus (AUS) |
| `switch.*_lernmodus` | Switch | Lernalgorithmus ein/aus |

### Temperatursensoren
| Entity | Einheit | Beschreibung |
|---|---|---|
| `sensor.*_zieltemperatur` | °C | Berechnete Zieltemperatur (nach Zeitplan, Wetter, Override) |
| `sensor.*_trv_setpoint` | °C | Tatsächlicher Wert der ans TRV gesendet wird (inkl. Boost) |
| `sensor.*_temperatur_ema_1h` | °C | Geglättete Innentemperatur (60-Minuten-Trend) |

### Diagnose-Sensoren
| Entity | Einheit | Beschreibung |
|---|---|---|
| `sensor.*_temperatur_slope` | K/min | Aktuelle Aufheiz-/Abkühlrate des Raums |
| `sensor.*_heat_loss` | K/min | Durchschnittliche Wärmeverlust-Rate (gelernt) |
| `sensor.*_heating_power` | K/min | Durchschnittliche Aufheizrate der Heizung (gelernt) |
| `sensor.*_sun_intensity_heatup` | % | Solarer Wärmeeintrag – wie stark die Sonne den Heizbedarf gerade reduziert |
| `sensor.*_vorheizzeit` | min | Errechnete Vorlaufzeit bis Zieltemperatur |
| `sensor.*_vorhersage_konfidenz` | % | Lernfortschritt – Qualität der Vorhersagen |
| `sensor.*_wetterkorrektur` | °C | Aktueller Temperatur-Offset durch Außenbedingungen |
| `sensor.*_status` | – | Zusammenfassung: Heizt / Temperatur gehalten / Sommer / Urlaub / … |

---

## Beispiel-Automationen

### Boost nach dem Lüften
```yaml
automation:
  - alias: "ThermoSmart – Boost nach Lüften"
    trigger:
      - platform: state
        entity_id: binary_sensor.fenster_wohnzimmer
        to: "off"
    action:
      - service: select.select_option
        target:
          entity_id: select.thermosmart_wohnzimmer_heizmodus
        data:
          option: "Boost"
      - delay: "00:30:00"
      - service: select.select_option
        target:
          entity_id: select.thermosmart_wohnzimmer_heizmodus
        data:
          option: "Auto"
```

### Benachrichtigung bei langsamem Heizen
```yaml
automation:
  - alias: "ThermoSmart – Heizung zu langsam"
    trigger:
      - platform: numeric_state
        entity_id: sensor.thermosmart_wohnzimmer_heating_power
        below: 0.02
        for: "00:30:00"
    condition:
      - condition: state
        entity_id: switch.thermosmart_wohnzimmer_aktive_steuerung
        state: "on"
    action:
      - service: notify.mobile_app
        data:
          message: "Wohnzimmer heizt langsam – Ventil prüfen?"
```

---

## Wie funktioniert der Lernalgorithmus?

```
Beobachtung alle 5 Minuten:
  - Innentemperatur, Zieltemperatur, Differenz
  - Außentemperatur, Wind, Sonne, Feuchte
  - Gemessene Heizrate (falls Raum sich erwärmt)
  - Gemessene Abkühlrate (falls Raum sich abkühlt)

Vorhersage:
  1. Zeitplan gibt Basis-Zieltemperatur vor
  2. Lernalgorithmus gleicht ab: "Was war bei ähnlichen Bedingungen optimal?"
     → Multi-Faktor-Ähnlichkeit (Gauß-Gewichtung pro Außenbedingung)
  3. Wetter-Engine korrigiert für aktuelle und prognostizierte Bedingungen
  4. Ergebnis: angepasste Zieltemperatur → berechne TRV-Setpoint

Selbstkorrektur:
  - Überschießen erkannt → Boost-Faktor −8%
  - Zu langsames Heizen erkannt → Boost-Faktor +5%
```

---

## Mithelfen & Contributing

Beiträge sind herzlich willkommen!

### Bugs melden
→ [GitHub Issues](https://github.com/Mikasmarthome/ThermoSmart/issues/new) mit:
- HA-Version und ThermoSmart-Version
- Was passiert ist
- Relevante Zeilen aus dem HA-Log (`Einstellungen → System → Protokoll → thermosmart`)

### Features vorschlagen
→ [GitHub Issues](https://github.com/Mikasmarthome/ThermoSmart/issues/new) mit Label `enhancement`

### Code beitragen
1. Repository **forken**
2. Feature-Branch: `git checkout -b feature/mein-feature`
3. Commit: `git commit -m "feat: mein feature"`
4. Push: `git push origin feature/mein-feature`
5. **Pull Request** öffnen

#### Besonders gesucht
- Tests mit echten TRV-Modellen (Sonoff, Danfoss, Eurotronic, Tuya, …)
- Übersetzungen – einfach neue Datei in `custom_components/thermosmart/translations/` anlegen
- Device-Quirks für weitere TRV-Modelle

### Fragen & Diskussion
→ [GitHub Discussions](https://github.com/Mikasmarthome/ThermoSmart/discussions)

---

## Lizenz

MIT License – siehe [LICENSE](LICENSE)
