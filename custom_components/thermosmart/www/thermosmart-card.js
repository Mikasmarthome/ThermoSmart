/**
 * ThermoSmart Lovelace Card v1.1.0
 * https://github.com/Mikasmarthome/thermosmart-card
 */
const CARD_VERSION = '1.1.0';

// ── i18n ─────────────────────────────────────────────────────────────────────

const I18N = {
  de: {
    heating: 'Heizt', idle: 'Hält Temp', off: 'Aus',
    obs: 'Beobachtung',
    confident: 'Zuverlässig', learning: 'Lernend', collecting: 'Lernstatus',
    battery_low: 'Batterie schwach',
    not_found: 'Entity nicht gefunden',
    no_entity: 'Keine ThermoSmart-Zone gefunden',
    chart_actual: 'Ist', chart_target: 'Soll', chart_outdoor: 'Außen',
    learn_active: 'Lernmodus aktiv', learn_inactive: 'Beobachtungsmodus',
    label_invert_temps: 'Temperaturen tauschen (Ziel groß)',
    label_show_trv_setpoint: 'TRV-Sollwert anzeigen',
    label_show_learning_progress: 'Lernfortschritt anzeigen',
    label_entity:   'ThermoSmart Climate-Entity',
    label_name:     'Anzeigename (optional)',
    label_section_display:   'Anzeige',
    label_section_scale:     'Temperaturskala',
    label_section_warnings:  'Warnungen',
    label_disable_humidity: 'Luftfeuchtigkeit ausblenden',
    label_disable_chart:    'Verlaufsdiagramm ausblenden',
    label_disable_modes:    'Modus-Buttons ausblenden',
    label_disable_chips:    'Info-Chips ausblenden',
    label_section_chips:         'Info-Chips konfigurieren',
    label_hide_chip_weather:     'Wetterkorrektur ausblenden',
    label_hide_chip_boost:       'TRV-Boost ausblenden',
    label_hide_chip_suppression: 'Vorhersage-Unterdrückung ausblenden',
    label_hide_chip_preheat:     'Vorheizzeit ausblenden',
    label_min_temp: 'Skalenminimum (°C)',
    label_max_temp: 'Skalenmaximum (°C)',
    label_low_battery_threshold: 'Schwellenwert Batteriewarnung (%)',
    label_chart_hours: 'Diagramm-Zeitfenster (Stunden)',
    window_open:      'Fenster offen',
    heating_failure:  'Heizungsausfall erkannt',
    mode_auto:        'Auto',
    mode_comfort:     'Komfort',
    mode_eco:         'Eco',
    mode_sleep:       'Nacht',
    mode_away:        'Abwesend',
    summer_active:    'Sommermodus',
    preheat_active:   'Vorheizen',
    mode_vacation:    'Urlaub',
  },
  en: {
    heating: 'Heating', idle: 'Idle', off: 'Off',
    obs: 'Observation',
    confident: 'Reliable', learning: 'Learning', collecting: 'Learning status',
    battery_low: 'Low battery',
    not_found: 'Entity not found',
    no_entity: 'No ThermoSmart zone found',
    chart_actual: 'Actual', chart_target: 'Target', chart_outdoor: 'Outdoor',
    learn_active: 'Learning active', learn_inactive: 'Observation mode',
    label_invert_temps: 'Swap temperatures (target as primary)',
    label_show_trv_setpoint: 'Show TRV setpoint',
    label_show_learning_progress: 'Show learning progress',
    label_entity:   'ThermoSmart Climate Entity',
    label_name:     'Display name (optional)',
    label_section_display:   'Display',
    label_section_scale:     'Temperature scale',
    label_section_warnings:  'Warnings',
    label_disable_humidity: 'Hide humidity',
    label_disable_chart:    'Hide history chart',
    label_disable_modes:    'Hide mode buttons',
    label_disable_chips:    'Hide info chips',
    label_section_chips:         'Configure info chips',
    label_hide_chip_weather:     'Hide weather correction',
    label_hide_chip_boost:       'Hide TRV boost',
    label_hide_chip_suppression: 'Hide forecast suppression',
    label_hide_chip_preheat:     'Hide preheat time',
    label_min_temp: 'Scale minimum (°C)',
    label_max_temp: 'Scale maximum (°C)',
    label_low_battery_threshold: 'Low battery threshold (%)',
    label_chart_hours: 'Chart time window (hours)',
    window_open:      'Window open',
    heating_failure:  'Heating failure detected',
    mode_auto:        'Auto',
    mode_comfort:     'Comfort',
    mode_eco:         'Eco',
    mode_sleep:       'Night',
    mode_away:        'Away',
    summer_active:    'Summer mode',
    preheat_active:   'Preheating',
    mode_vacation:    'Vacation',
  },
  fr: {
    heating: 'Chauffe', idle: 'Temp atteinte', off: 'Éteint',
    obs: 'Observation',
    confident: 'Fiable', learning: 'Apprentissage', collecting: 'Statut d\'apprentissage',
    battery_low: 'Batterie faible',
    not_found: 'Entité introuvable',
    no_entity: 'Aucune zone ThermoSmart trouvée',
    chart_actual: 'Réel', chart_target: 'Cible', chart_outdoor: 'Extérieur',
    learn_active: 'Apprentissage actif', learn_inactive: 'Mode observation',
    label_invert_temps: 'Inverser températures (cible principale)',
    label_show_trv_setpoint: 'Afficher la consigne TRV',
    label_show_learning_progress: 'Afficher la progression d\'apprentissage',
    label_entity:   'Entité ThermoSmart',
    label_name:     "Nom d'affichage (optionnel)",
    label_section_display:   'Affichage',
    label_section_scale:     "Échelle de température",
    label_section_warnings:  'Avertissements',
    label_disable_humidity: "Masquer l'humidité",
    label_disable_chart:    "Masquer le graphique",
    label_disable_modes:    'Masquer les boutons de mode',
    label_disable_chips:    'Masquer les puces info',
    label_section_chips:         'Configurer les puces info',
    label_hide_chip_weather:     'Masquer la correction météo',
    label_hide_chip_boost:       'Masquer le boost TRV',
    label_hide_chip_suppression: 'Masquer la suppression des prévisions',
    label_hide_chip_preheat:     'Masquer le préchauffage',
    label_min_temp: "Minimum de l'échelle (°C)",
    label_max_temp: "Maximum de l'échelle (°C)",
    label_low_battery_threshold: 'Seuil batterie faible (%)',
    label_chart_hours: 'Fenêtre temporelle du graphique (heures)',
    window_open:      'Fenêtre ouverte',
    heating_failure:  'Défaillance du chauffage',
    mode_auto:        'Auto',
    mode_comfort:     'Confort',
    mode_eco:         'Eco',
    mode_sleep:       'Nuit',
    mode_away:        'Absent',
    summer_active:    'Mode été',
    preheat_active:   'Préchauffage',
    mode_vacation:    'Vacances',
  },
  nl: {
    heating: 'Verwarmt', idle: 'Temp bereikt', off: 'Uit',
    obs: 'Observatie',
    confident: 'Betrouwbaar', learning: 'Lerend', collecting: 'Leerstatus',
    battery_low: 'Batterij bijna leeg',
    not_found: 'Entiteit niet gevonden',
    no_entity: 'Geen ThermoSmart-zone gevonden',
    chart_actual: 'Actueel', chart_target: 'Doel', chart_outdoor: 'Buiten',
    learn_active: 'Leermodus actief', learn_inactive: 'Observatiemodus',
    label_invert_temps: 'Temperaturen wisselen (doel als primair)',
    label_show_trv_setpoint: 'TRV-instelpunt weergeven',
    label_show_learning_progress: 'Leervoortgang weergeven',
    label_entity:   'ThermoSmart klimaatentiteit',
    label_name:     'Weergavenaam (optioneel)',
    label_section_display:   'Weergave',
    label_section_scale:     'Temperatuurschaal',
    label_section_warnings:  'Waarschuwingen',
    label_disable_humidity: 'Luchtvochtigheid verbergen',
    label_disable_chart:    'Grafiek verbergen',
    label_disable_modes:    'Modusknoppen verbergen',
    label_disable_chips:    'Info-chips verbergen',
    label_section_chips:         'Info-chips configureren',
    label_hide_chip_weather:     'Weercorrectie verbergen',
    label_hide_chip_boost:       'TRV-boost verbergen',
    label_hide_chip_suppression: 'Voorspellingsonderdrukking verbergen',
    label_hide_chip_preheat:     'Voorverwarmingstijd verbergen',
    label_min_temp: 'Schaalminimum (°C)',
    label_max_temp: 'Schaalmaximum (°C)',
    label_low_battery_threshold: 'Drempel batterij bijna leeg (%)',
    label_chart_hours: 'Grafiek tijdvenster (uren)',
    window_open:      'Raam open',
    heating_failure:  'Verwarmingsstoring gedetecteerd',
    mode_auto:        'Auto',
    mode_comfort:     'Comfort',
    mode_eco:         'Eco',
    mode_sleep:       'Nacht',
    mode_away:        'Afwezig',
    summer_active:    'Zomermodus',
    preheat_active:   'Voorverwarmen',
    mode_vacation:    'Vakantie',
  },
  it: {
    heating: 'Riscaldamento', idle: 'Temp raggiunta', off: 'Spento',
    obs: 'Osservazione',
    confident: 'Affidabile', learning: 'Apprendimento', collecting: 'Stato di apprendimento',
    battery_low: 'Batteria scarica',
    not_found: 'Entità non trovata',
    no_entity: 'Nessuna zona ThermoSmart trovata',
    chart_actual: 'Reale', chart_target: 'Obiettivo', chart_outdoor: 'Esterno',
    learn_active: 'Apprendimento attivo', learn_inactive: 'Modalità osservazione',
    label_invert_temps: 'Inverti temperature (obiettivo principale)',
    label_show_trv_setpoint: 'Mostra setpoint TRV',
    label_show_learning_progress: 'Mostra avanzamento apprendimento',
    label_entity:   'Entità ThermoSmart',
    label_name:     'Nome (opzionale)',
    label_section_display:   'Visualizzazione',
    label_section_scale:     'Scala di temperatura',
    label_section_warnings:  'Avvisi',
    label_disable_humidity: "Nascondi umidità",
    label_disable_chart:    'Nascondi grafico',
    label_disable_modes:    'Nascondi pulsanti modalità',
    label_disable_chips:    'Nascondi chip info',
    label_section_chips:         'Configura chip info',
    label_hide_chip_weather:     'Nascondi correzione meteo',
    label_hide_chip_boost:       'Nascondi boost TRV',
    label_hide_chip_suppression: 'Nascondi soppressione previsioni',
    label_hide_chip_preheat:     'Nascondi tempo di preriscaldamento',
    label_min_temp: 'Minimo scala (°C)',
    label_max_temp: 'Massimo scala (°C)',
    label_low_battery_threshold: 'Soglia batteria scarica (%)',
    label_chart_hours: 'Finestra temporale grafico (ore)',
    window_open:      'Finestra aperta',
    heating_failure:  'Guasto riscaldamento rilevato',
    mode_auto:        'Auto',
    mode_comfort:     'Comfort',
    mode_eco:         'Eco',
    mode_sleep:       'Notte',
    mode_away:        'Assente',
    summer_active:    'Modalità estate',
    preheat_active:   'Preriscaldamento',
    mode_vacation:    'Vacanza',
  },
  pl: {
    heating: 'Grzeje', idle: 'Temp osiągnięta', off: 'Wyłączony',
    obs: 'Obserwacja',
    confident: 'Niezawodny', learning: 'Uczenie się', collecting: 'Status uczenia',
    battery_low: 'Niski poziom baterii',
    not_found: 'Encja nie znaleziona',
    no_entity: 'Nie znaleziono strefy ThermoSmart',
    chart_actual: 'Rzeczywista', chart_target: 'Cel', chart_outdoor: 'Zewnętrzna',
    learn_active: 'Nauka aktywna', learn_inactive: 'Tryb obserwacji',
    label_invert_temps: 'Zamień temperatury (cel jako główny)',
    label_show_trv_setpoint: 'Pokaż setpoint TRV',
    label_show_learning_progress: 'Pokaż postęp uczenia',
    label_entity:   'Encja ThermoSmart',
    label_name:     'Nazwa wyświetlania (opcjonalna)',
    label_section_display:   'Wyświetlanie',
    label_section_scale:     'Skala temperatury',
    label_section_warnings:  'Ostrzeżenia',
    label_disable_humidity: 'Ukryj wilgotność',
    label_disable_chart:    'Ukryj wykres',
    label_disable_modes:    'Ukryj przyciski trybu',
    label_disable_chips:    'Ukryj chipy info',
    label_section_chips:         'Konfiguruj chipy info',
    label_hide_chip_weather:     'Ukryj korektę pogody',
    label_hide_chip_boost:       'Ukryj boost TRV',
    label_hide_chip_suppression: 'Ukryj tłumienie prognozy',
    label_hide_chip_preheat:     'Ukryj czas podgrzewania',
    label_min_temp: 'Minimum skali (°C)',
    label_max_temp: 'Maksimum skali (°C)',
    label_low_battery_threshold: 'Próg niskiej baterii (%)',
    label_chart_hours: 'Okno czasowe wykresu (godziny)',
    window_open:      'Okno otwarte',
    heating_failure:  'Wykryto awarię ogrzewania',
    mode_auto:        'Auto',
    mode_comfort:     'Komfort',
    mode_eco:         'Eco',
    mode_sleep:       'Noc',
    mode_away:        'Nieobecny',
    summer_active:    'Tryb letni',
    preheat_active:   'Podgrzewanie',
    mode_vacation:    'Urlop',
  },
  sv: {
    heating: 'Värmer', idle: 'Temp nådd', off: 'Av',
    obs: 'Observation',
    confident: 'Tillförlitlig', learning: 'Lärande', collecting: 'Inlärningsstatus',
    battery_low: 'Lågt batteri',
    not_found: 'Entitet hittades inte',
    no_entity: 'Ingen ThermoSmart-zon hittades',
    chart_actual: 'Aktuell', chart_target: 'Mål', chart_outdoor: 'Utomhus',
    learn_active: 'Inlärning aktiv', learn_inactive: 'Observationsläge',
    label_invert_temps: 'Byt temperaturer (mål som primär)',
    label_show_trv_setpoint: 'Visa TRV-börvärde',
    label_show_learning_progress: 'Visa inlärningsförlopp',
    label_entity:   'ThermoSmart klimatenhet',
    label_name:     'Visningsnamn (valfritt)',
    label_section_display:   'Visning',
    label_section_scale:     'Temperaturskala',
    label_section_warnings:  'Varningar',
    label_disable_humidity: 'Dölj luftfuktighet',
    label_disable_chart:    'Dölj historikdiagram',
    label_disable_modes:    'Dölj lägesknappar',
    label_disable_chips:    'Dölj info-chips',
    label_section_chips:         'Konfigurera info-chips',
    label_hide_chip_weather:     'Dölj väderkorrektion',
    label_hide_chip_boost:       'Dölj TRV-boost',
    label_hide_chip_suppression: 'Dölj prognossuppression',
    label_hide_chip_preheat:     'Dölj förvärmingstid',
    label_min_temp: 'Skalminimum (°C)',
    label_max_temp: 'Skalmaximum (°C)',
    label_low_battery_threshold: 'Tröskel för lågt batteri (%)',
    label_chart_hours: 'Diagrammets tidsfönster (timmar)',
    window_open:      'Fönster öppet',
    heating_failure:  'Uppvärmningsfel detekterat',
    mode_auto:        'Auto',
    mode_comfort:     'Komfort',
    mode_eco:         'Eco',
    mode_sleep:       'Natt',
    mode_away:        'Borta',
    summer_active:    'Sommarläge',
    preheat_active:   'Förvärmning',
    mode_vacation:    'Semester',
  },
  es: {
    heating: 'Calentando', idle: 'Temp. alcanzada', off: 'Apagado',
    obs: 'Observación',
    confident: 'Fiable', learning: 'Aprendiendo', collecting: 'Estado de aprendizaje',
    battery_low: 'Batería baja',
    not_found: 'Entidad no encontrada',
    no_entity: 'No se encontró ninguna zona ThermoSmart',
    chart_actual: 'Real', chart_target: 'Objetivo', chart_outdoor: 'Exterior',
    learn_active: 'Aprendizaje activo', learn_inactive: 'Modo observación',
    label_invert_temps: 'Intercambiar temperaturas (objetivo principal)',
    label_show_trv_setpoint: 'Mostrar setpoint TRV',
    label_show_learning_progress: 'Mostrar progreso de aprendizaje',
    label_entity:   'Entidad ThermoSmart',
    label_name:     'Nombre de visualización (opcional)',
    label_section_display:   'Visualización',
    label_section_scale:     'Escala de temperatura',
    label_section_warnings:  'Avisos',
    label_disable_humidity: 'Ocultar humedad',
    label_disable_chart:    'Ocultar gráfico',
    label_disable_modes:    'Ocultar botones de modo',
    label_disable_chips:    'Ocultar chips de información',
    label_section_chips:         'Configurar chips de información',
    label_hide_chip_weather:     'Ocultar corrección meteorológica',
    label_hide_chip_boost:       'Ocultar boost TRV',
    label_hide_chip_suppression: 'Ocultar supresión de previsión',
    label_hide_chip_preheat:     'Ocultar temporizador de precalentamiento',
    label_min_temp: 'Mínimo de escala (°C)',
    label_max_temp: 'Máximo de escala (°C)',
    label_low_battery_threshold: 'Umbral de batería baja (%)',
    label_chart_hours: 'Ventana temporal del gráfico (horas)',
    window_open:      'Ventana abierta',
    heating_failure:  'Fallo de calefacción detectado',
    mode_auto:        'Auto',
    mode_comfort:     'Confort',
    mode_eco:         'Eco',
    mode_sleep:       'Noche',
    mode_away:        'Ausente',
    summer_active:    'Modo verano',
    preheat_active:   'Precalentamiento',
    mode_vacation:    'Vacaciones',
  },
  pt: {
    heating: 'Aquecendo', idle: 'Temp. atingida', off: 'Desligado',
    obs: 'Observação',
    confident: 'Confiável', learning: 'Aprendendo', collecting: 'Estado de aprendizagem',
    battery_low: 'Bateria fraca',
    not_found: 'Entidade não encontrada',
    no_entity: 'Nenhuma zona ThermoSmart encontrada',
    chart_actual: 'Real', chart_target: 'Alvo', chart_outdoor: 'Exterior',
    learn_active: 'Aprendizado ativo', learn_inactive: 'Modo observação',
    label_invert_temps: 'Trocar temperaturas (alvo principal)',
    label_show_trv_setpoint: 'Mostrar setpoint TRV',
    label_show_learning_progress: 'Mostrar progresso de aprendizagem',
    label_entity:   'Entidade ThermoSmart',
    label_name:     'Nome de exibição (opcional)',
    label_section_display:   'Exibição',
    label_section_scale:     'Escala de temperatura',
    label_section_warnings:  'Avisos',
    label_disable_humidity: 'Ocultar humidade',
    label_disable_chart:    'Ocultar gráfico',
    label_disable_modes:    'Ocultar botões de modo',
    label_disable_chips:    'Ocultar chips de informação',
    label_section_chips:         'Configurar chips de informação',
    label_hide_chip_weather:     'Ocultar correção meteorológica',
    label_hide_chip_boost:       'Ocultar boost TRV',
    label_hide_chip_suppression: 'Ocultar supressão de previsão',
    label_hide_chip_preheat:     'Ocultar tempo de pré-aquecimento',
    label_min_temp: 'Mínimo da escala (°C)',
    label_max_temp: 'Máximo da escala (°C)',
    label_low_battery_threshold: 'Limite de bateria fraca (%)',
    label_chart_hours: 'Janela temporal do gráfico (horas)',
    window_open:      'Janela aberta',
    heating_failure:  'Falha no aquecimento detectada',
    mode_auto:        'Auto',
    mode_comfort:     'Conforto',
    mode_eco:         'Eco',
    mode_sleep:       'Noite',
    mode_away:        'Ausente',
    summer_active:    'Modo verão',
    preheat_active:   'Pré-aquecimento',
    mode_vacation:    'Férias',
  },
  cs: {
    heating: 'Topí', idle: 'Teplota dosažena', off: 'Vypnuto',
    obs: 'Sledování',
    confident: 'Spolehlivý', learning: 'Učení', collecting: 'Stav učení',
    battery_low: 'Slabá baterie',
    not_found: 'Entita nenalezena',
    no_entity: 'Nebyla nalezena žádná zóna ThermoSmart',
    chart_actual: 'Skutečná', chart_target: 'Cíl', chart_outdoor: 'Venku',
    learn_active: 'Učení aktivní', learn_inactive: 'Režim sledování',
    label_invert_temps: 'Vyměnit teploty (cíl jako hlavní)',
    label_show_trv_setpoint: 'Zobrazit setpoint TRV',
    label_show_learning_progress: 'Zobrazit průběh učení',
    label_entity:   'ThermoSmart klimatická entita',
    label_name:     'Zobrazovaný název (volitelné)',
    label_section_display:   'Zobrazení',
    label_section_scale:     'Teplotní stupnice',
    label_section_warnings:  'Varování',
    label_disable_humidity: 'Skrýt vlhkost',
    label_disable_chart:    'Skrýt graf',
    label_disable_modes:    'Skrýt tlačítka režimů',
    label_disable_chips:    'Skrýt informační čipy',
    label_section_chips:         'Konfigurovat informační čipy',
    label_hide_chip_weather:     'Skrýt korekci počasí',
    label_hide_chip_boost:       'Skrýt TRV boost',
    label_hide_chip_suppression: 'Skrýt potlačení předpovědi',
    label_hide_chip_preheat:     'Skrýt čas předehřevu',
    label_min_temp: 'Minimum stupnice (°C)',
    label_max_temp: 'Maximum stupnice (°C)',
    label_low_battery_threshold: 'Práh slabé baterie (%)',
    label_chart_hours: 'Časové okno grafu (hodiny)',
    window_open:      'Okno otevřeno',
    heating_failure:  'Zjištěna porucha topení',
    mode_auto:        'Auto',
    mode_comfort:     'Komfort',
    mode_eco:         'Eco',
    mode_sleep:       'Noc',
    mode_away:        'Pryč',
    summer_active:    'Letní režim',
    preheat_active:   'Předehřev',
    mode_vacation:    'Dovolená',
  },
  da: {
    heating: 'Varmer', idle: 'Temp. nået', off: 'Fra',
    obs: 'Observation',
    confident: 'Pålidelig', learning: 'Lærer', collecting: 'Læringsstatus',
    battery_low: 'Lavt batteri',
    not_found: 'Enhed ikke fundet',
    no_entity: 'Ingen ThermoSmart-zone fundet',
    chart_actual: 'Aktuel', chart_target: 'Mål', chart_outdoor: 'Udendørs',
    learn_active: 'Læring aktiv', learn_inactive: 'Observationstilstand',
    label_invert_temps: 'Byt temperaturer (mål som primær)',
    label_show_trv_setpoint: 'Vis TRV-sætpunkt',
    label_show_learning_progress: 'Vis læringsforløb',
    label_entity:   'ThermoSmart klimaenhed',
    label_name:     'Vist navn (valgfrit)',
    label_section_display:   'Visning',
    label_section_scale:     'Temperaturskala',
    label_section_warnings:  'Advarsler',
    label_disable_humidity: 'Skjul luftfugtighed',
    label_disable_chart:    'Skjul historikdiagram',
    label_disable_modes:    'Skjul tilstandsknapper',
    label_disable_chips:    'Skjul info-chips',
    label_section_chips:         'Konfigurer info-chips',
    label_hide_chip_weather:     'Skjul vejrkorrektion',
    label_hide_chip_boost:       'Skjul TRV-boost',
    label_hide_chip_suppression: 'Skjul prognosesuppression',
    label_hide_chip_preheat:     'Skjul forvarmingstid',
    label_min_temp: 'Skalaminimum (°C)',
    label_max_temp: 'Skalamaximum (°C)',
    label_low_battery_threshold: 'Grænse for lavt batteri (%)',
    label_chart_hours: 'Diagrammets tidsvindue (timer)',
    window_open:      'Vindue åbent',
    heating_failure:  'Varmefejl registreret',
    mode_auto:        'Auto',
    mode_comfort:     'Komfort',
    mode_eco:         'Eco',
    mode_sleep:       'Nat',
    mode_away:        'Væk',
    summer_active:    'Sommertilstand',
    preheat_active:   'Forvarming',
    mode_vacation:    'Ferie',
  },
  fi: {
    heating: 'Lämmittää', idle: 'Lämpötila saavutettu', off: 'Pois päältä',
    obs: 'Tarkkailu',
    confident: 'Luotettava', learning: 'Oppii', collecting: 'Oppimisstatus',
    battery_low: 'Akku heikko',
    not_found: 'Entiteettiä ei löydy',
    no_entity: 'ThermoSmart-vyöhykettä ei löydy',
    chart_actual: 'Todellinen', chart_target: 'Tavoite', chart_outdoor: 'Ulkona',
    learn_active: 'Oppiminen aktiivinen', learn_inactive: 'Tarkkailutila',
    label_invert_temps: 'Vaihda lämpötilat (tavoite ensisijaisena)',
    label_show_trv_setpoint: 'Näytä TRV-asetusarvo',
    label_show_learning_progress: 'Näytä oppimisedistyminen',
    label_entity:   'ThermoSmart ilmastoentiteetti',
    label_name:     'Näyttönimi (valinnainen)',
    label_section_display:   'Näyttö',
    label_section_scale:     'Lämpötila-asteikko',
    label_section_warnings:  'Varoitukset',
    label_disable_humidity: 'Piilota kosteus',
    label_disable_chart:    'Piilota historiakaavio',
    label_disable_modes:    'Piilota tilapainikkeet',
    label_disable_chips:    'Piilota tietosirut',
    label_section_chips:         'Määritä tietosirut',
    label_hide_chip_weather:     'Piilota säänkorjaus',
    label_hide_chip_boost:       'Piilota TRV-tehostus',
    label_hide_chip_suppression: 'Piilota ennustesuppressio',
    label_hide_chip_preheat:     'Piilota esilämmitysaika',
    label_min_temp: 'Asteikon minimi (°C)',
    label_max_temp: 'Asteikon maksimi (°C)',
    label_low_battery_threshold: 'Heikon akun raja (%)',
    label_chart_hours: 'Kaavion aikaikkuna (tunnit)',
    window_open:      'Ikkuna auki',
    heating_failure:  'Lämmitysvika havaittu',
    mode_auto:        'Auto',
    mode_comfort:     'Mukavuus',
    mode_eco:         'Eco',
    mode_sleep:       'Yö',
    mode_away:        'Poissa',
    summer_active:    'Kesätila',
    preheat_active:   'Esilämmitys',
    mode_vacation:    'Loma',
  },
  nb: {
    heating: 'Varmer', idle: 'Temp. nådd', off: 'Av',
    obs: 'Observasjon',
    confident: 'Pålitelig', learning: 'Lærer', collecting: 'Læringsstatus',
    battery_low: 'Lavt batteri',
    not_found: 'Enhet ikke funnet',
    no_entity: 'Ingen ThermoSmart-sone funnet',
    chart_actual: 'Aktuell', chart_target: 'Mål', chart_outdoor: 'Utendørs',
    learn_active: 'Læring aktiv', learn_inactive: 'Observasjonsmodus',
    label_invert_temps: 'Bytt temperaturer (mål som primær)',
    label_show_trv_setpoint: 'Vis TRV-settpunkt',
    label_show_learning_progress: 'Vis læringsfremdrift',
    label_entity:   'ThermoSmart klimaenhet',
    label_name:     'Visningsnavn (valgfritt)',
    label_section_display:   'Visning',
    label_section_scale:     'Temperaturskala',
    label_section_warnings:  'Advarsler',
    label_disable_humidity: 'Skjul luftfuktighet',
    label_disable_chart:    'Skjul historikk-diagram',
    label_disable_modes:    'Skjul modus-knapper',
    label_disable_chips:    'Skjul info-chips',
    label_section_chips:         'Konfigurer info-chips',
    label_hide_chip_weather:     'Skjul værkorreksjon',
    label_hide_chip_boost:       'Skjul TRV-boost',
    label_hide_chip_suppression: 'Skjul prognose-suppresjon',
    label_hide_chip_preheat:     'Skjul forvarmningstid',
    label_min_temp: 'Skalaminimum (°C)',
    label_max_temp: 'Skalamaximum (°C)',
    label_low_battery_threshold: 'Terskel for lavt batteri (%)',
    label_chart_hours: 'Diagrammets tidsvindu (timer)',
    window_open:      'Vindu åpent',
    heating_failure:  'Varmefeil oppdaget',
    mode_auto:        'Auto',
    mode_comfort:     'Komfort',
    mode_eco:         'Eco',
    mode_sleep:       'Natt',
    mode_away:        'Borte',
    summer_active:    'Sommermodus',
    preheat_active:   'Forvarmning',
    mode_vacation:    'Ferie',
  },
  bg: {
    heating: 'Отоплява', idle: 'Темп. достигната', off: 'Изключено',
    obs: 'Наблюдение',
    confident: 'Надежден', learning: 'Учи', collecting: 'Статус на обучение',
    battery_low: 'Слаба батерия',
    not_found: 'Обектът не е намерен',
    no_entity: 'Не е намерена ThermoSmart зона',
    chart_actual: 'Реална', chart_target: 'Целева', chart_outdoor: 'Навън',
    learn_active: 'Обучението активно', learn_inactive: 'Режим наблюдение',
    label_invert_temps: 'Смени температурите (цел като основна)',
    label_show_trv_setpoint: 'Покажи TRV уставка',
    label_show_learning_progress: 'Покажи напредъка на обучението',
    label_entity:   'ThermoSmart климатичен обект',
    label_name:     'Показвано име (незадължително)',
    label_section_display:   'Дисплей',
    label_section_scale:     'Температурна скала',
    label_section_warnings:  'Предупреждения',
    label_disable_humidity: 'Скрий влажността',
    label_disable_chart:    'Скрий графиката',
    label_disable_modes:    'Скрий бутоните за режим',
    label_disable_chips:    'Скрий информационните чипове',
    label_section_chips:         'Конфигурирай чиповете',
    label_hide_chip_weather:     'Скрий метео корекция',
    label_hide_chip_boost:       'Скрий TRV boost',
    label_hide_chip_suppression: 'Скрий потискане на прогнозата',
    label_hide_chip_preheat:     'Скрий времето за предварително загряване',
    label_min_temp: 'Минимум на скалата (°C)',
    label_max_temp: 'Максимум на скалата (°C)',
    label_low_battery_threshold: 'Праг за слаба батерия (%)',
    label_chart_hours: 'Времево прозорче на графиката (часове)',
    window_open:      'Прозорец отворен',
    heating_failure:  'Открита неизправност на отоплението',
    mode_auto:        'Авто',
    mode_comfort:     'Комфорт',
    mode_eco:         'Еко',
    mode_sleep:       'Нощ',
    mode_away:        'Отсъстващ',
    summer_active:    'Летен режим',
    preheat_active:   'Предогряване',
    mode_vacation:    'Ваканция',
  },
  ca: {
    heating: 'Escalfant', idle: 'Temp. assolida', off: 'Apagat',
    obs: 'Observació',
    confident: 'Fiable', learning: 'Aprenent', collecting: 'Estat d\'aprenentatge',
    battery_low: 'Bateria baixa',
    not_found: 'Entitat no trobada',
    no_entity: 'No s\'ha trobat cap zona ThermoSmart',
    chart_actual: 'Real', chart_target: 'Objectiu', chart_outdoor: 'Exterior',
    learn_active: 'Aprenentatge actiu', learn_inactive: 'Mode observació',
    label_invert_temps: 'Intercanviar temperatures (objectiu principal)',
    label_show_trv_setpoint: 'Mostra setpoint TRV',
    label_show_learning_progress: 'Mostra el progrés d\'aprenentatge',
    label_entity:   'Entitat ThermoSmart',
    label_name:     "Nom de visualització (opcional)",
    label_section_display:   'Visualització',
    label_section_scale:     "Escala de temperatura",
    label_section_warnings:  'Avisos',
    label_disable_humidity: "Amagar humitat",
    label_disable_chart:    'Amagar gràfic',
    label_disable_modes:    'Amagar botons de mode',
    label_disable_chips:    "Amagar xips d'informació",
    label_section_chips:         "Configurar xips d'informació",
    label_hide_chip_weather:     'Amagar correcció meteorològica',
    label_hide_chip_boost:       'Amagar boost TRV',
    label_hide_chip_suppression: 'Amagar supressió de previsió',
    label_hide_chip_preheat:     'Amagar temps de preescalfament',
    label_min_temp: "Mínim de l'escala (°C)",
    label_max_temp: "Màxim de l'escala (°C)",
    label_low_battery_threshold: 'Llindar de bateria baixa (%)',
    label_chart_hours: 'Finestra temporal del gràfic (hores)',
    window_open:      'Finestra oberta',
    heating_failure:  'Error de calefacció detectat',
    mode_auto:        'Auto',
    mode_comfort:     'Confort',
    mode_eco:         'Eco',
    mode_sleep:       'Nit',
    mode_away:        'Absent',
    summer_active:    'Mode estiu',
    preheat_active:   'Preescalfament',
    mode_vacation:    'Vacances',
  },
  el: {
    heating: 'Θερμαίνει', idle: 'Θερμ. επιτεύχθηκε', off: 'Απενεργ.',
    obs: 'Παρακολούθηση',
    confident: 'Αξιόπιστο', learning: 'Μάθηση', collecting: 'Κατάσταση μάθησης',
    battery_low: 'Χαμηλή μπαταρία',
    not_found: 'Η οντότητα δεν βρέθηκε',
    no_entity: 'Δεν βρέθηκε ζώνη ThermoSmart',
    chart_actual: 'Πραγματική', chart_target: 'Στόχος', chart_outdoor: 'Εξωτερικό',
    learn_active: 'Μάθηση ενεργή', learn_inactive: 'Λειτουργία παρακολούθησης',
    label_invert_temps: 'Εναλλαγή θερμοκρασιών (στόχος κύριος)',
    label_show_trv_setpoint: 'Εμφάνιση τιμής TRV',
    label_show_learning_progress: 'Εμφάνιση προόδου εκμάθησης',
    label_entity:   'Οντότητα ThermoSmart',
    label_name:     'Εμφανιζόμενο όνομα (προαιρετικό)',
    label_section_display:   'Εμφάνιση',
    label_section_scale:     'Κλίμακα θερμοκρασίας',
    label_section_warnings:  'Προειδοποιήσεις',
    label_disable_humidity: 'Απόκρυψη υγρασίας',
    label_disable_chart:    'Απόκρυψη γραφήματος',
    label_disable_modes:    'Απόκρυψη κουμπιών λειτουργίας',
    label_disable_chips:    'Απόκρυψη τσιπ πληροφοριών',
    label_section_chips:         'Διαμόρφωση τσιπ πληροφοριών',
    label_hide_chip_weather:     'Απόκρυψη διόρθωσης καιρού',
    label_hide_chip_boost:       'Απόκρυψη boost TRV',
    label_hide_chip_suppression: 'Απόκρυψη καταστολής πρόβλεψης',
    label_hide_chip_preheat:     'Απόκρυψη χρόνου προθέρμανσης',
    label_min_temp: 'Ελάχιστο κλίμακας (°C)',
    label_max_temp: 'Μέγιστο κλίμακας (°C)',
    label_low_battery_threshold: 'Κατώφλι χαμηλής μπαταρίας (%)',
    label_chart_hours: 'Χρονικό παράθυρο γραφήματος (ώρες)',
    window_open:      'Παράθυρο ανοιχτό',
    heating_failure:  'Εντοπίστηκε βλάβη θέρμανσης',
    mode_auto:        'Αυτόματο',
    mode_comfort:     'Άνεση',
    mode_eco:         'Eco',
    mode_sleep:       'Νύχτα',
    mode_away:        'Απουσία',
    summer_active:    'Καλοκαιρινή λειτουργία',
    preheat_active:   'Προθέρμανση',
    mode_vacation:    'Διακοπές',
  },
  hu: {
    heating: 'Fűt', idle: 'Hőm. elérve', off: 'Kikapcsolt',
    obs: 'Megfigyelés',
    confident: 'Megbízható', learning: 'Tanulás', collecting: 'Tanulási állapot',
    battery_low: 'Gyenge akkumulátor',
    not_found: 'Az entitás nem található',
    no_entity: 'Nem található ThermoSmart zóna',
    chart_actual: 'Tényleges', chart_target: 'Célérték', chart_outdoor: 'Kültéri',
    learn_active: 'Tanulás aktív', learn_inactive: 'Megfigyelési mód',
    label_invert_temps: 'Hőmérsékletek cseréje (célérték elsődleges)',
    label_show_trv_setpoint: 'TRV beállítási érték mutatása',
    label_show_learning_progress: 'Tanulási folyamat mutatása',
    label_entity:   'ThermoSmart klímaentitás',
    label_name:     'Megjelenített név (opcionális)',
    label_section_display:   'Megjelenítés',
    label_section_scale:     'Hőmérsékleti skála',
    label_section_warnings:  'Figyelmeztetések',
    label_disable_humidity: 'Páratartalom elrejtése',
    label_disable_chart:    'Grafikon elrejtése',
    label_disable_modes:    'Mód gombok elrejtése',
    label_disable_chips:    'Infochipek elrejtése',
    label_section_chips:         'Infochipek konfigurálása',
    label_hide_chip_weather:     'Időjárás-korrekció elrejtése',
    label_hide_chip_boost:       'TRV boost elrejtése',
    label_hide_chip_suppression: 'Előrejelzés-elnyomás elrejtése',
    label_hide_chip_preheat:     'Előmelegítési idő elrejtése',
    label_min_temp: 'Skála minimum (°C)',
    label_max_temp: 'Skála maximum (°C)',
    label_low_battery_threshold: 'Gyenge akkumulátor küszöb (%)',
    label_chart_hours: 'Grafikon időablak (óra)',
    window_open:      'Ablak nyitva',
    heating_failure:  'Fűtési hiba észlelve',
    mode_auto:        'Auto',
    mode_comfort:     'Kényelmi',
    mode_eco:         'Eco',
    mode_sleep:       'Éjszaka',
    mode_away:        'Távol',
    summer_active:    'Nyári mód',
    preheat_active:   'Előmelegítés',
    mode_vacation:    'Vakáció',
  },
  ro: {
    heating: 'Încălzește', idle: 'Temp. atinsă', off: 'Oprit',
    obs: 'Observare',
    confident: 'Fiabil', learning: 'Învățare', collecting: 'Status de învățare',
    battery_low: 'Baterie slabă',
    not_found: 'Entitatea nu a fost găsită',
    no_entity: 'Nu s-a găsit nicio zonă ThermoSmart',
    chart_actual: 'Reală', chart_target: 'Țintă', chart_outdoor: 'Exterior',
    learn_active: 'Învățare activă', learn_inactive: 'Mod observare',
    label_invert_temps: 'Schimbă temperaturile (țintă principală)',
    label_show_trv_setpoint: 'Afișează setpoint TRV',
    label_show_learning_progress: 'Afișează progresul de învățare',
    label_entity:   'Entitate ThermoSmart',
    label_name:     'Nume afișat (opțional)',
    label_section_display:   'Afișaj',
    label_section_scale:     'Scală temperatură',
    label_section_warnings:  'Avertizări',
    label_disable_humidity: 'Ascunde umiditatea',
    label_disable_chart:    'Ascunde graficul',
    label_disable_modes:    'Ascunde butoanele de mod',
    label_disable_chips:    'Ascunde cipurile info',
    label_section_chips:         'Configurează cipurile info',
    label_hide_chip_weather:     'Ascunde corecția meteo',
    label_hide_chip_boost:       'Ascunde boost TRV',
    label_hide_chip_suppression: 'Ascunde suprimarea prognozei',
    label_hide_chip_preheat:     'Ascunde timpul de preîncălzire',
    label_min_temp: 'Minimum scală (°C)',
    label_max_temp: 'Maximum scală (°C)',
    label_low_battery_threshold: 'Prag baterie slabă (%)',
    label_chart_hours: 'Fereastră temporală grafic (ore)',
    window_open:      'Fereastră deschisă',
    heating_failure:  'Defecțiune încălzire detectată',
    mode_auto:        'Auto',
    mode_comfort:     'Confort',
    mode_eco:         'Eco',
    mode_sleep:       'Noapte',
    mode_away:        'Absent',
    summer_active:    'Mod vară',
    preheat_active:   'Preîncălzire',
    mode_vacation:    'Vacanță',
  },
  ru: {
    heating: 'Нагрев', idle: 'Темп. достигнута', off: 'Выключено',
    obs: 'Наблюдение',
    confident: 'Надёжный', learning: 'Обучение', collecting: 'Статус обучения',
    battery_low: 'Слабая батарея',
    not_found: 'Объект не найден',
    no_entity: 'Зона ThermoSmart не найдена',
    chart_actual: 'Фактическая', chart_target: 'Целевая', chart_outdoor: 'На улице',
    learn_active: 'Обучение активно', learn_inactive: 'Режим наблюдения',
    label_invert_temps: 'Поменять температуры (цель главная)',
    label_show_trv_setpoint: 'Показать уставку TRV',
    label_show_learning_progress: 'Показать прогресс обучения',
    label_entity:   'Объект ThermoSmart',
    label_name:     'Отображаемое имя (необязательно)',
    label_section_display:   'Отображение',
    label_section_scale:     'Шкала температуры',
    label_section_warnings:  'Предупреждения',
    label_disable_humidity: 'Скрыть влажность',
    label_disable_chart:    'Скрыть график',
    label_disable_modes:    'Скрыть кнопки режимов',
    label_disable_chips:    'Скрыть информационные чипы',
    label_section_chips:         'Настроить чипы',
    label_hide_chip_weather:     'Скрыть метеокоррекцию',
    label_hide_chip_boost:       'Скрыть boost TRV',
    label_hide_chip_suppression: 'Скрыть подавление прогноза',
    label_hide_chip_preheat:     'Скрыть время предварительного нагрева',
    label_min_temp: 'Минимум шкалы (°C)',
    label_max_temp: 'Максимум шкалы (°C)',
    label_low_battery_threshold: 'Порог слабой батареи (%)',
    label_chart_hours: 'Временное окно графика (часы)',
    window_open:      'Окно открыто',
    heating_failure:  'Обнаружена неисправность отопления',
    mode_auto:        'Авто',
    mode_comfort:     'Комфорт',
    mode_eco:         'Эко',
    mode_sleep:       'Ночь',
    mode_away:        'Отсутствие',
    summer_active:    'Летний режим',
    preheat_active:   'Предогрев',
    mode_vacation:    'Отпуск',
  },
  sk: {
    heating: 'Kúri', idle: 'Teplota dosiahnutá', off: 'Vypnuté',
    obs: 'Pozorovanie',
    confident: 'Spoľahlivý', learning: 'Učenie', collecting: 'Stav učenia',
    battery_low: 'Slabá batéria',
    not_found: 'Entita nenájdená',
    no_entity: 'Nebola nájdená žiadna zóna ThermoSmart',
    chart_actual: 'Skutočná', chart_target: 'Cieľ', chart_outdoor: 'Vonku',
    learn_active: 'Učenie aktívne', learn_inactive: 'Režim pozorovania',
    label_invert_temps: 'Vymeniť teploty (cieľ ako hlavná)',
    label_show_trv_setpoint: 'Zobraziť setpoint TRV',
    label_show_learning_progress: 'Zobraziť priebeh učenia',
    label_entity:   'ThermoSmart klimatická entita',
    label_name:     'Zobrazovaný názov (voliteľné)',
    label_section_display:   'Zobrazenie',
    label_section_scale:     'Teplotná stupnica',
    label_section_warnings:  'Varovania',
    label_disable_humidity: 'Skryť vlhkosť',
    label_disable_chart:    'Skryť graf',
    label_disable_modes:    'Skryť tlačidlá režimov',
    label_disable_chips:    'Skryť informačné čipy',
    label_section_chips:         'Konfigurovať informačné čipy',
    label_hide_chip_weather:     'Skryť korekciu počasia',
    label_hide_chip_boost:       'Skryť TRV boost',
    label_hide_chip_suppression: 'Skryť potlačenie predpovede',
    label_hide_chip_preheat:     'Skryť čas predohrevu',
    label_min_temp: 'Minimum stupnice (°C)',
    label_max_temp: 'Maximum stupnice (°C)',
    label_low_battery_threshold: 'Prah slabej batérie (%)',
    label_chart_hours: 'Časové okno grafu (hodiny)',
    window_open:      'Okno otvorené',
    heating_failure:  'Zistená porucha kúrenia',
    mode_auto:        'Auto',
    mode_comfort:     'Komfort',
    mode_eco:         'Eco',
    mode_sleep:       'Noc',
    mode_away:        'Preč',
    summer_active:    'Letný režim',
    preheat_active:   'Predohrev',
    mode_vacation:    'Dovolenka',
  },
  sl: {
    heating: 'Ogrevanje', idle: 'Temp. dosežena', off: 'Izklopljeno',
    obs: 'Opazovanje',
    confident: 'Zanesljiv', learning: 'Učenje', collecting: 'Status učenja',
    battery_low: 'Slaba baterija',
    not_found: 'Entiteta ni najdena',
    no_entity: 'Nobena cona ThermoSmart ni bila najdena',
    chart_actual: 'Dejanska', chart_target: 'Cilj', chart_outdoor: 'Zunaj',
    learn_active: 'Učenje aktivno', learn_inactive: 'Način opazovanja',
    label_invert_temps: 'Zamenjaj temperature (cilj kot primarni)',
    label_show_trv_setpoint: 'Prikaži setpoint TRV',
    label_show_learning_progress: 'Prikaži napredek učenja',
    label_entity:   'ThermoSmart podnebna entiteta',
    label_name:     'Prikazano ime (neobvezno)',
    label_section_display:   'Prikaz',
    label_section_scale:     'Temperaturna lestvica',
    label_section_warnings:  'Opozorila',
    label_disable_humidity: 'Skrij vlažnost',
    label_disable_chart:    'Skrij grafikon',
    label_disable_modes:    'Skrij gumbe načina',
    label_disable_chips:    'Skrij informacijske čipe',
    label_section_chips:         'Konfiguriraj informacijske čipe',
    label_hide_chip_weather:     'Skrij vremensko korekcijo',
    label_hide_chip_boost:       'Skrij TRV boost',
    label_hide_chip_suppression: 'Skrij zatiranje napovedi',
    label_hide_chip_preheat:     'Skrij čas predogrevanja',
    label_min_temp: 'Minimum lestvice (°C)',
    label_max_temp: 'Maksimum lestvice (°C)',
    label_low_battery_threshold: 'Prag slabe baterije (%)',
    label_chart_hours: 'Časovno okno grafikona (ure)',
    window_open:      'Okno odprto',
    heating_failure:  'Odkrita napaka ogrevanja',
    mode_auto:        'Auto',
    mode_comfort:     'Udobje',
    mode_eco:         'Eco',
    mode_sleep:       'Noč',
    mode_away:        'Odsoten',
    summer_active:    'Poletni način',
    preheat_active:   'Predogrevanje',
    mode_vacation:    'Dopust',
  },
  tr: {
    heating: 'Isıtıyor', idle: 'Sıcaklık sağlandı', off: 'Kapalı',
    obs: 'Gözlem',
    confident: 'Güvenilir', learning: 'Öğreniyor', collecting: 'Öğrenme durumu',
    battery_low: 'Düşük pil',
    not_found: 'Varlık bulunamadı',
    no_entity: 'ThermoSmart bölgesi bulunamadı',
    chart_actual: 'Gerçek', chart_target: 'Hedef', chart_outdoor: 'Dış mekan',
    learn_active: 'Öğrenme aktif', learn_inactive: 'Gözlem modu',
    label_invert_temps: 'Sıcaklıkları değiştir (hedef birincil)',
    label_show_trv_setpoint: 'TRV ayar değerini göster',
    label_show_learning_progress: 'Öğrenme ilerlemesini göster',
    label_entity:   'ThermoSmart iklim varlığı',
    label_name:     'Görünen ad (isteğe bağlı)',
    label_section_display:   'Görünüm',
    label_section_scale:     'Sıcaklık ölçeği',
    label_section_warnings:  'Uyarılar',
    label_disable_humidity: 'Nemi gizle',
    label_disable_chart:    'Grafiği gizle',
    label_disable_modes:    'Mod düğmelerini gizle',
    label_disable_chips:    'Bilgi çiplerini gizle',
    label_section_chips:         'Bilgi çiplerini yapılandır',
    label_hide_chip_weather:     'Hava durumu düzeltmesini gizle',
    label_hide_chip_boost:       'TRV boost gizle',
    label_hide_chip_suppression: 'Tahmin baskılamayı gizle',
    label_hide_chip_preheat:     'Ön ısıtma süresini gizle',
    label_min_temp: 'Ölçek minimum (°C)',
    label_max_temp: 'Ölçek maksimum (°C)',
    label_low_battery_threshold: 'Düşük pil eşiği (%)',
    label_chart_hours: 'Grafik zaman penceresi (saat)',
    window_open:      'Pencere açık',
    heating_failure:  'Isıtma arızası tespit edildi',
    mode_auto:        'Otomatik',
    mode_comfort:     'Konfor',
    mode_eco:         'Eco',
    mode_sleep:       'Gece',
    mode_away:        'Uzakta',
    summer_active:    'Yaz modu',
    preheat_active:   'Ön ısıtma',
    mode_vacation:    'Tatil',
  },
  uk: {
    heating: 'Нагріває', idle: 'Темп. досягнута', off: 'Вимкнено',
    obs: 'Спостереження',
    confident: 'Надійний', learning: 'Навчання', collecting: 'Статус навчання',
    battery_low: 'Слабка батарея',
    not_found: 'Об\'єкт не знайдено',
    no_entity: 'Зону ThermoSmart не знайдено',
    chart_actual: 'Фактична', chart_target: 'Цільова', chart_outdoor: 'На вулиці',
    learn_active: 'Навчання активне', learn_inactive: 'Режим спостереження',
    label_invert_temps: 'Змінити температури (ціль головна)',
    label_show_trv_setpoint: 'Показати уставку TRV',
    label_show_learning_progress: 'Показати прогрес навчання',
    label_entity:   'Об\'єкт ThermoSmart',
    label_name:     'Відображувана назва (необов\'язково)',
    label_section_display:   'Відображення',
    label_section_scale:     'Шкала температури',
    label_section_warnings:  'Попередження',
    label_disable_humidity: 'Сховати вологість',
    label_disable_chart:    'Сховати графік',
    label_disable_modes:    'Сховати кнопки режимів',
    label_disable_chips:    'Сховати інформаційні чіпи',
    label_section_chips:         'Налаштувати чіпи',
    label_hide_chip_weather:     'Сховати метеокорекцію',
    label_hide_chip_boost:       'Сховати boost TRV',
    label_hide_chip_suppression: 'Сховати придушення прогнозу',
    label_hide_chip_preheat:     'Сховати час попереднього нагрівання',
    label_min_temp: 'Мінімум шкали (°C)',
    label_max_temp: 'Максимум шкали (°C)',
    label_low_battery_threshold: 'Поріг слабкої батареї (%)',
    label_chart_hours: 'Часове вікно графіка (годин)',
    window_open:      'Вікно відкрито',
    heating_failure:  'Виявлена несправність опалення',
    mode_auto:        'Авто',
    mode_comfort:     'Комфорт',
    mode_eco:         'Еко',
    mode_sleep:       'Ніч',
    mode_away:        'Відсутність',
    summer_active:    'Літній режим',
    preheat_active:   'Попередній нагрів',
    mode_vacation:    'Відпустка',
  },
  zh: {
    heating: '加热中', idle: '已达温度', off: '关闭',
    obs: '观测',
    confident: '可靠', learning: '学习中', collecting: '学习状态',
    battery_low: '电池电量低',
    not_found: '未找到实体',
    no_entity: '未找到 ThermoSmart 区域',
    chart_actual: '实际', chart_target: '目标', chart_outdoor: '室外',
    learn_active: '学习模式激活', learn_inactive: '观测模式',
    label_invert_temps: '交换温度显示（目标为主）',
    label_show_trv_setpoint: '显示 TRV 设定值',
    label_show_learning_progress: '显示学习进度',
    label_entity:   'ThermoSmart 气候实体',
    label_name:     '显示名称（可选）',
    label_section_display:   '显示',
    label_section_scale:     '温度刻度',
    label_section_warnings:  '警告',
    label_disable_humidity: '隐藏湿度',
    label_disable_chart:    '隐藏历史图表',
    label_disable_modes:    '隐藏模式按钮',
    label_disable_chips:    '隐藏信息标签',
    label_section_chips:         '配置信息标签',
    label_hide_chip_weather:     '隐藏天气修正',
    label_hide_chip_boost:       '隐藏 TRV 增益',
    label_hide_chip_suppression: '隐藏预测抑制',
    label_hide_chip_preheat:     '隐藏预热时间',
    label_min_temp: '刻度最小值 (°C)',
    label_max_temp: '刻度最大值 (°C)',
    label_low_battery_threshold: '低电量报警阈值 (%)',
    label_chart_hours: '图表时间窗口（小时）',
    window_open:      '窗户已开',
    heating_failure:  '检测到供暖故障',
    mode_auto:        '自动',
    mode_comfort:     '舒适',
    mode_eco:         '节能',
    mode_sleep:       '睡眠',
    mode_away:        '离家',
    summer_active:    '夏季模式',
    preheat_active:   '预热',
    mode_vacation:    '假期',
  },
};

function tr(hass, key) {
  const lang = (hass?.language || 'de').split('-')[0];
  return (I18N[lang] || I18N.de)[key] ?? I18N.de[key] ?? key;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function debounce(fn, delay) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

function polarToXY(cx, cy, r, deg) {
  const rad = (deg - 90) * Math.PI / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(cx, cy, r, fromDeg, sweepDeg) {
  if (sweepDeg < 0.5) return '';
  const s = polarToXY(cx, cy, r, fromDeg);
  const e = polarToXY(cx, cy, r, fromDeg + sweepDeg);
  return `M${s.x.toFixed(1)} ${s.y.toFixed(1)} A${r} ${r} 0 ${sweepDeg > 180 ? 1 : 0} 1 ${e.x.toFixed(1)} ${e.y.toFixed(1)}`;
}

// ── Ring geometry constants ────────────────────────────────────────────────────
const CX = 140, CY = 126, R = 112, AF = 240, AS = 240;

// ── MDI icon paths for editor sections ────────────────────────────────────────
const MDI_EYE  = 'M12,9A3,3 0 0,0 9,12A3,3 0 0,0 12,15A3,3 0 0,0 15,12A3,3 0 0,0 12,9M12,17A5,5 0 0,1 7,12A5,5 0 0,1 12,7A5,5 0 0,1 17,12A5,5 0 0,1 12,17M12,4.5C7,4.5 2.73,7.61 1,12C2.73,16.39 7,19.5 12,19.5C17,19.5 21.27,16.39 23,12C21.27,7.61 17,4.5 12,4.5Z';
const MDI_TUNE = 'M3,17V19H9V17H3M3,5V7H13V5H3M13,21V19H21V17H13V15H11V21H13M7,9V11H3V13H7V15H9V9H7M21,13V11H11V13H21M15,9H17V7H21V5H17V3H15V9Z';
const MDI_ALERT = 'M13,14H11V10H13M13,18H11V16H13M1,21H23L12,2L1,21Z';
const MDI_FILTER= 'M6,13H18V11H6M3,6V8H21V6M10,18H14V16H10V18Z';

// ── Preset modes (must match ThermoSmart integration const.py) ─────────────────
const MODES = [
  { preset: 'auto',     icon: 'mdi:home-thermometer-outline' },
  { preset: 'comfort',  icon: 'mdi:sun-thermometer-outline'  },
  { preset: 'eco',      icon: 'mdi:leaf'                     },
  { preset: 'sleep',    icon: 'mdi:weather-night'            },
  { preset: 'away',     icon: 'mdi:walk'                     },
  { preset: 'vacation', icon: 'mdi:airplane'                 },
];

const STATE_COLORS = {
  heating: { main: '#e8573f', glow: 0.24, text: '#ff8a70' },
  idle:    { main: '#4caf50', glow: 0.20, text: '#81c784' },
  obs:     { main: '#ff9800', glow: 0.16, text: '#ffb74d' },
  off:     { main: '#555555', glow: 0.06, text: '#888888' },
};

const PRESET_COLORS = {
  comfort:  { main: '#ff7043', glow: 0.22, text: '#ff8a65' },
  eco:      { main: '#43a047', glow: 0.21, text: '#66bb6a' },
  sleep:    { main: '#5c6bc0', glow: 0.20, text: '#7986cb' },
  away:     { main: '#78909c', glow: 0.16, text: '#90a4ae' },
  vacation: { main: '#7986cb', glow: 0.18, text: '#9fa8da' },
  auto:     { main: '#cc8800', glow: 0.18, text: '#f0a500' },
};

// chart line colors (fixed, independent of preset)
const CHART_COL_IST    = '#2196F3';  // blue   – actual
const CHART_COL_TGT    = '#ff9800';  // orange – target
const CHART_COL_OUTDOOR= '#ef5350';  // red    – outdoor

// ── Config-Editor ─────────────────────────────────────────────────────────────

const EDITOR_SCHEMA = [
  { name: 'entity', required: true, selector: { entity: { domain: 'climate' } } },
  { name: 'name',   selector: { text: {} } },
  {
    name: 'section_display',
    type: 'expandable',
    flatten: true,
    expanded: true,
    iconPath: MDI_EYE,
    schema: [
      { type: 'grid', name: '', schema: [
        { name: 'invert_temps',             selector: { boolean: {} } },
        { name: 'show_trv_setpoint',        selector: { boolean: {} } },
        { name: 'show_learning_progress',   selector: { boolean: {} } },
        { name: 'disable_humidity',         selector: { boolean: {} } },
        { name: 'disable_modes',            selector: { boolean: {} } },
        { name: 'disable_chart',            selector: { boolean: {} } },
      ]},
      { name: 'chart_hours', selector: { number: { min: 1, max: 24, step: 1, mode: 'box', unit_of_measurement: 'h' } } },
    ],
  },
  {
    name: 'section_scale',
    type: 'expandable',
    flatten: true,
    iconPath: MDI_TUNE,
    schema: [
      { type: 'grid', name: '', schema: [
        { name: 'min_temp', selector: { number: { min: 5,  max: 20, step: 1, mode: 'box', unit_of_measurement: '°C' } } },
        { name: 'max_temp', selector: { number: { min: 20, max: 35, step: 1, mode: 'box', unit_of_measurement: '°C' } } },
      ]},
    ],
  },
  {
    name: 'section_warnings',
    type: 'expandable',
    flatten: true,
    iconPath: MDI_ALERT,
    schema: [
      { name: 'low_battery_threshold', selector: { number: { min: 0, max: 100, step: 1, mode: 'box', unit_of_measurement: '%' } } },
    ],
  },
];

class ThermosmartCardEditor extends HTMLElement {
  constructor() {
    super();
    this._config = {};
    this._hass   = null;
    this._form   = null;
  }

  setConfig(config) { this._config = config; this._update(); }
  set hass(hass)    { this._hass = hass; this._update(); }

  // ha-form created once and reused to prevent focus loss on HA state updates
  _update() {
    if (!this._hass) return;

    if (!this._form) {
      const form = document.createElement('ha-form');
      form.schema       = EDITOR_SCHEMA;
      form.computeLabel = (s) => tr(this._hass, `label_${s.name}`);
      form.addEventListener('value-changed', (ev) => {
        ev.stopPropagation();
        this._config = ev.detail.value;
        this.dispatchEvent(new CustomEvent('config-changed', {
          detail: { config: this._config }, bubbles: true, composed: true,
        }));
      });
      this._form = form;
      this.replaceChildren(form);
    }

    this._form.hass = this._hass;
    this._form.data = { chart_hours: 24, low_battery_threshold: 15, show_learning_progress: true, ...this._config };
  }
}
if (!customElements.get('thermosmart-card-editor'))
  customElements.define('thermosmart-card-editor', ThermosmartCardEditor);

// ── Main Card ─────────────────────────────────────────────────────────────────

class ThermosmartCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._config           = {};
    this._hass             = null;
    this._isDragging       = false;
    this._dragTemp              = null;
    this._optimisticTemp        = null;   // holds target after set until HA confirms
    this._optimisticTimer       = null;
    this._lastKnownTargetTemp   = null;
    this._listeners        = [];
    this._historyData      = null;
    this._historyFetching  = false;
    this._lastHistoryFetch = 0;
    this._sparklineCache   = null;
    this._minTemp             = 15;
    this._maxTemp             = 30;
    this._learnSwitch         = null;
    this._activeSwitch        = null;
    this._switchesDetected    = false;
    this._optimisticPreset    = null;
    this._optimisticPresetTimer = null;
  }

  static async getStubConfig(hass) {
    const found = Object.entries(hass?.states || {})
      .find(([eid, st]) => {
        if (!eid.startsWith('climate.')) return false;
        const a = st?.attributes || {};
        return eid.startsWith('climate.thermosmart') ||
          ('learning_confidence' in a && 'observation_mode' in a);
      })?.[0];
    return { entity: found || '' };
  }
  static getConfigElement() { return document.createElement('thermosmart-card-editor'); }

  setConfig(config) {
    if (!('entity' in config)) throw new Error('ThermoSmart Card: required "entity" field is missing.');
    this._config             = config;
    this._lastHistoryFetch   = 0;
    this._switchesDetected   = false;
    this._switchDetectAttempts = 0;
    this._learnSwitch        = null;
    this._activeSwitch       = null;
    this._optimisticPreset   = null;
    this._optimisticTemp     = null;
    this._dragTemp           = null;
    clearTimeout(this._optimisticPresetTimer);
    clearTimeout(this._optimisticTimer);
    this._render();
  }

  set hass(hass) {
    const prev = this._hass;
    this._hass = hass;
    if (this._isDragging) return;
    if (prev && this._config?.entity) {
      const wasUnavailable = prev.states[this._config.entity]?.state === 'unavailable';
      const isAvailable    = hass.states[this._config.entity]?.state !== 'unavailable';
      if (wasUnavailable && isAvailable) this._lastHistoryFetch = 0;
    }
    // Clear optimistic temp once HA confirms the new value
    if (this._optimisticTemp !== null) {
      const haTemp = hass.states[this._config?.entity]?.attributes?.temperature;
      if (haTemp != null && Math.abs(haTemp - this._optimisticTemp) < 0.3) {
        clearTimeout(this._optimisticTimer);
        this._optimisticTemp  = null;
        this._optimisticTimer = null;
      }
    }
    if (!this._switchesDetected) this._detectSwitches();
    // Clear optimistic preset when HA confirms the new value
    if (this._optimisticPreset !== null) {
      const haPreset = hass.states[this._config?.entity]?.attributes?.preset_mode;
      if (haPreset === this._optimisticPreset) {
        clearTimeout(this._optimisticPresetTimer);
        this._optimisticPreset = null;
        this._optimisticPresetTimer = null;
      }
    }
    const entityChanged = !prev ||
      prev.states[this._config?.entity] !== hass.states[this._config?.entity];
    const learnChanged  = !!(this._learnSwitch  && prev?.states[this._learnSwitch]  !== hass.states[this._learnSwitch]);
    const activeChanged = !!(this._activeSwitch && prev?.states[this._activeSwitch] !== hass.states[this._activeSwitch]);
    if (!entityChanged && !learnChanged && !activeChanged) return;
    this._fetchHistory();
    this._render();
  }

  _detectSwitches() {
    if (!this._hass || !this._config?.entity) return;
    const entity = this._config.entity;
    if (!entity.startsWith('climate.thermosmart_')) return;
    const zone   = entity.slice('climate.thermosmart_'.length);
    const prefix = `switch.thermosmart_${zone}_`;
    const ids    = Object.keys(this._hass.states);
    this._learnSwitch  = ids.find(id => id.startsWith(prefix) && /learn|lern/i.test(id))  ?? null;
    this._activeSwitch = ids.find(id => id.startsWith(prefix) && /active|aktiv|steuer/i.test(id)) ?? null;
    this._switchDetectAttempts = (this._switchDetectAttempts || 0) + 1;
    // Mark done if switches found OR after 10 attempts (avoid endless retrying)
    this._switchesDetected = !!(this._learnSwitch || this._activeSwitch) || this._switchDetectAttempts >= 10;
  }

  // Keep displayed target temp optimistically after a set call until HA confirms
  _setOptimistic(temp) {
    this._optimisticTemp = temp;
    clearTimeout(this._optimisticTimer);
    this._optimisticTimer = setTimeout(() => {
      this._optimisticTemp  = null;
      this._optimisticTimer = null;
    }, 30000);
  }

  getCardSize() { return 8; }

  // ── Cleanup ──────────────────────────────────────────────────────────────

  disconnectedCallback() { this._cleanup(); }

  _cleanup() {
    this._listeners.forEach(fn => { try { fn(); } catch (_) {} });
    this._listeners = [];
    this._isDragging = false;
  }

  // ── History fetch ─────────────────────────────────────────────────────────

  _fetchHistory() {
    if (this._config.disable_chart) return;
    if (!this._hass?.callApi || this._historyFetching) return;
    if (Date.now() - this._lastHistoryFetch < 5 * 60 * 1000) return;

    this._historyFetching = true;
    const hours = this._config.chart_hours ?? 24;
    const start = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();

    this._hass.callApi(
      'GET',
      `history/period/${start}?filter_entity_id=${encodeURIComponent(this._config.entity)}&significant_changes_only=false&minimal_response=true`
    ).then(data => {
      this._historyFetching  = false;
      this._lastHistoryFetch = Date.now();

      const states = data?.[0];
      if (!states?.length) return;

      this._historyData = states.map(s => {
        const a = s.a || s.attributes || {};
        const t = s.lu ? s.lu * 1000 : new Date(s.last_updated).getTime();
        return {
          t,
          curr:    parseFloat(a.current_temperature),
          tgt:     parseFloat(a.temperature),
          outdoor: parseFloat(a.outdoor_temperature),
        };
      }).filter(p => !isNaN(p.curr));

      if (this._lastKnownTargetTemp == null) {
        for (let i = this._historyData.length - 1; i >= 0; i--) {
          if (!isNaN(this._historyData[i].tgt)) {
            this._lastKnownTargetTemp = this._historyData[i].tgt;
            break;
          }
        }
      }

      if (!this._isDragging) this._render();
    }).catch(() => {
      this._historyFetching  = false;
      this._lastHistoryFetch = Date.now();
    });
  }

  // ── Data ─────────────────────────────────────────────────────────────────

  _getData() {
    if (!this._hass || !this._config.entity) return null;
    const st = this._hass.states[this._config.entity];
    if (!st) return null;
    const a = st.attributes;

    // Parses both raw numbers and string values like "16%", "5 min", "+2.5°C"
    const num = (v, fb = null) => {
      const n = parseFloat(String(v ?? '').replace(/[^0-9.+-]/g, ''));
      return isNaN(n) ? fb : n;
    };

    const isObs      = a.observation_mode === true;
    const hvacAction = a.hvac_action || 'off';

    let stateKey = 'off';
    if (isObs)                         stateKey = 'obs';
    else if (hvacAction === 'heating') stateKey = 'heating';
    else if (hvacAction === 'idle')    stateKey = 'idle';
    else if (st.state !== 'off')       stateKey = 'idle';

    const raw  = this._config.name || a.friendly_name || this._config.entity;
    const name = raw.replace(/^ThermoSmart\s*[–\-]\s*/i, '');

    const activeOn = this._activeSwitch
      ? (this._hass.states[this._activeSwitch]?.state === 'on')
      : !isObs;
    const learnOn = this._learnSwitch
      ? (this._hass.states[this._learnSwitch]?.state === 'on')
      : !isObs;

    const preset = this._optimisticPreset ?? a.preset_mode ?? 'auto';
    const presetCol = stateKey !== 'off' ? (PRESET_COLORS[preset] ?? null) : null;
    const col = presetCol ?? STATE_COLORS[stateKey];

    const rawTarget  = a.temperature ?? null;
    const windowOpen = a.window_open === true;
    if (rawTarget != null) this._lastKnownTargetTemp = rawTarget;
    const targetTemp = this._dragTemp ?? this._optimisticTemp ?? rawTarget
                       ?? (windowOpen ? (this._lastKnownTargetTemp ?? null) : null);

    return {
      entityId:    this._config.entity,
      name,
      currentTemp: a.current_temperature ?? null,
      targetTemp,
      hvacMode:    st.state,
      hvacAction,
      preset,
      isObs,
      activeOn,
      learnOn,
      stateKey,
      col,
      outdoorTemp: num(a.outdoor_temperature),
      weatherOff:  num(a.weather_correction),
      confidence:  num(a.learning_confidence, 0),
      preheat:     num(a.preheat_time, 0),
      trvSetpoint: num(a.trv_setpoint),
      boostDelta:  num(a.boost_delta, 0),
      boostFactor: num(a.boost_factor, 1.0),
      suppression: num(a.forecast_suppression, 0),
      humidity:         num(a.current_humidity),
      deviceBatteries:  a.device_batteries || {},
      windowOpen,
      heatingFailure: a.heating_failure === true,
      summerMode:     a.summer_mode === true,
      preheatActive:  a.preheat_active === true,
      entityMinTemp:  a.min_temp ?? 15,
      entityMaxTemp:  a.max_temp ?? 30,
    };
  }

  _statusLabel(d) {
    if (d.heatingFailure)                 return tr(this._hass, 'heating_failure');
    if (d.summerMode)                     return tr(this._hass, 'summer_active');
    if (d.windowOpen)                     return tr(this._hass, 'window_open');
    if (d.preset === 'vacation')          return tr(this._hass, 'mode_vacation');
    if (d.isObs)                          return tr(this._hass, 'obs');
    if (d.hvacAction === 'heating')       return tr(this._hass, 'heating');
    if (d.hvacAction === 'idle')          return tr(this._hass, 'idle');
    return tr(this._hass, 'off');
  }

  // ── SVG Ring ─────────────────────────────────────────────────────────────

  _buildRing(d) {
    const MIN_T = this._minTemp;
    const MAX_T = this._maxTemp;
    const frac  = t => Math.max(0, Math.min(1, (t - MIN_T) / (MAX_T - MIN_T)));

    const arcInactive = !d.activeOn || d.hvacMode === 'off';
    const fillColor   = arcInactive ? 'var(--secondary-background-color,rgba(120,120,120,.25))' : d.col.main;

    let fillSvg = '';
    if (d.currentTemp != null) {
      const sw = frac(d.currentTemp) * AS;
      if (sw > 0.5) {
        fillSvg = `
          <path d="${arcPath(CX, CY, R, AF, sw)}"
            fill="none" stroke="${fillColor}" stroke-width="18" stroke-linecap="round"/>
          <path d="${arcPath(CX, CY, R - 16, AF, sw)}"
            fill="none" stroke="${fillColor}" stroke-width="3" stroke-linecap="round" opacity="0.28"/>`;
      }
    }

    let dotSvg = '';
    if (d.targetTemp != null) {
      const p = polarToXY(CX, CY, R, AF + frac(d.targetTemp) * AS);
      dotSvg = `<circle class="ring-dot" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="7"
        style="fill:var(--ha-card-background,var(--card-background-color,#fff))" stroke="${d.col.main}" stroke-width="2.5"/>`;
    }

    return `
      <svg class="ring-svg" viewBox="0 0 280 196"
        style="width:100%;max-width:360px;display:block;margin:0 auto;overflow:visible;touch-action:none">
        <defs>
          <radialGradient id="grd_${d.stateKey}" cx="50%" cy="50%" r="50%">
            <stop offset="0%"   stop-color="${d.col.main}" stop-opacity="${d.col.glow}"/>
            <stop offset="100%" stop-color="${d.col.main}" stop-opacity="0"/>
          </radialGradient>
        </defs>
        <circle cx="${CX}" cy="${CY}" r="126" fill="url(#grd_${d.stateKey})"/>
        <path d="${arcPath(CX, CY, R, AF, AS)}"
          fill="none" stroke="var(--secondary-background-color,rgba(120,120,120,.15))" stroke-width="18" stroke-linecap="round"/>
        <path d="${arcPath(CX, CY, R - 16, AF, AS)}"
          fill="none" stroke="var(--divider-color,rgba(120,120,120,.12))" stroke-width="3" stroke-linecap="round"/>
        ${fillSvg}
        ${dotSvg}
        <path class="arc-interact" d="${arcPath(CX, CY, R, AF, AS)}"
          fill="none" stroke="transparent" stroke-width="44" stroke-linecap="round"/>
      </svg>`;
  }

  // ── Overlays & widgets ───────────────────────────────────────────────────

  _buildTempOverlay(d) {
    const invert  = !!this._config.invert_temps;
    const bigTemp = invert ? d.targetTemp  : d.currentTemp;
    const secTemp = invert ? d.currentTemp : null;

    const intPart = bigTemp != null ? Math.floor(Math.abs(bigTemp)) * (bigTemp < 0 ? -1 : 1) : null;
    const decPart = bigTemp != null ? Math.round(Math.abs(bigTemp - Math.trunc(bigTemp)) * 10) : null;

    const contextPill = d.windowOpen
      ? `<div class="ov-icon-badge ov-icon--window"><ha-icon icon="mdi:window-open-variant" style="--mdc-icon-size:28px"></ha-icon></div>`
      : !d.activeOn
      ? `<div class="ov-icon-badge ov-icon--obs${d.learnOn ? ' ov-icon--obs-learn' : ''}"><ha-icon icon="mdi:eye" style="--mdc-icon-size:28px"></ha-icon></div>`
      : d.summerMode
      ? `<div class="ov-icon-badge ov-icon--summer"><ha-icon icon="mdi:white-balance-sunny" style="--mdc-icon-size:28px"></ha-icon></div>`
      : d.preset === 'vacation'
      ? `<div class="ov-icon-badge ov-icon--vacation"><ha-icon icon="mdi:airplane" style="--mdc-icon-size:28px"></ha-icon></div>`
      : '';

    return `
      <div class="ring-overlay">
        <div class="ov-pill-slot">${contextPill}</div>
        <div class="ov-status" style="color:${d.col.text}">${this._statusLabel(d)}</div>
        <div class="ov-temp">
          ${intPart != null
            ? `<span class="ov-int">${intPart}</span><span class="ov-frac">.${decPart}<span class="ov-unit">°C</span></span>`
            : `<span class="ov-int">--</span>`
          }
        </div>
        ${invert
          ? (d.summerMode || d.windowOpen)
            ? `<div class="ov-target" style="opacity:0.45${secTemp == null ? ';visibility:hidden' : ''}"><ha-icon icon="mdi:pause-circle-outline" style="--mdc-icon-size:16px"></ha-icon>${secTemp != null ? secTemp.toFixed(1) + '°' : '--'}</div>`
            : `<div class="ov-target" style="${secTemp == null ? 'visibility:hidden' : ''}"><ha-icon icon="mdi:thermometer" style="--mdc-icon-size:16px"></ha-icon>${secTemp != null ? secTemp.toFixed(1) + '°' : '--'}</div>`
          : (d.summerMode || d.windowOpen)
          ? `<div class="ov-target" style="opacity:0.45${d.targetTemp == null ? ';visibility:hidden' : ''}"><ha-icon icon="mdi:pause-circle-outline" style="--mdc-icon-size:16px"></ha-icon>${d.targetTemp != null ? d.targetTemp.toFixed(1) + '°' : '--'}${this._config.show_trv_setpoint && d.trvSetpoint != null ? `<ha-icon icon="mdi:valve" style="--mdc-icon-size:16px;margin-left:10px"></ha-icon>${d.trvSetpoint.toFixed(1)}°` : ''}</div>`
          : (this._config.show_trv_setpoint && d.trvSetpoint != null)
          ? `<div class="ov-target" style="${d.targetTemp == null ? 'visibility:hidden' : ''}"><ha-icon icon="mdi:thermostat" style="--mdc-icon-size:16px"></ha-icon>${d.targetTemp != null ? d.targetTemp.toFixed(1) + '°' : '--'}<ha-icon icon="mdi:valve" style="--mdc-icon-size:16px;margin-left:10px"></ha-icon>${d.trvSetpoint.toFixed(1)}°</div>`
          : `<div class="ov-target" style="${d.targetTemp == null ? 'visibility:hidden' : ''}"><ha-icon icon="mdi:thermostat" style="--mdc-icon-size:16px"></ha-icon>${d.targetTemp != null ? d.targetTemp.toFixed(1) + '°' : '--'}</div>`}
        ${d.humidity != null
          ? `<div class="ov-humidity"${this._config.disable_humidity ? ' style="visibility:hidden"' : ''}><ha-icon icon="mdi:water-percent" style="--mdc-icon-size:16px"></ha-icon>${d.humidity.toFixed(0)}%</div>`
          : ''}
      </div>`;
  }

  _buildBanners(d) {
    let html = '';
    if (d.heatingFailure) {
      html += `<div class="banner failure">
        <ha-icon icon="mdi:alert" style="--mdc-icon-size:14px"></ha-icon>
        ${tr(this._hass, 'heating_failure')}
      </div>`;
    }
    const thr = this._config.low_battery_threshold ?? 15;
    const lowDevices = Object.entries(d.deviceBatteries)
      .filter(([, pct]) => pct <= thr)
      .sort(([, a], [, b]) => a - b);
    for (const [entityId, pct] of lowDevices) {
      const name = this._hass.states[entityId]?.attributes?.friendly_name
        || entityId.split('.').pop().replace(/_/g, ' ');
      html += `<div class="banner battery">
        <ha-icon icon="mdi:battery-alert-variant-outline" style="--mdc-icon-size:14px"></ha-icon>
        ${tr(this._hass, 'battery_low')}: ${esc(name)} · ${pct}%
      </div>`;
    }
    return html;
  }

  _buildModes(d) {
    return `<div class="modes">${MODES.map(m => {
      const active = m.preset === d.preset;
      return `<button class="mode-btn${active ? ' active' : ''}" data-preset="${m.preset}"
        title="${tr(this._hass, 'mode_' + m.preset)}" style="${active ? `--mc:${d.col.main};--mt:${d.col.text}` : ''}">
        <ha-icon icon="${m.icon}"></ha-icon>
      </button>`;
    }).join('')}</div>`;
  }

  _buildChips(d) {
    if (d.summerMode) return '';
    const cfg = this._config;
    const c = [];
    if (!cfg.hide_chip_weather && d.weatherOff != null && Math.abs(d.weatherOff) >= 0.1)
      c.push(`<span class="chip ${d.weatherOff > 0 ? 'pos' : 'neg'}">💨 ${d.weatherOff > 0 ? '+' : ''}${d.weatherOff.toFixed(1)}°</span>`);
    if (!cfg.hide_chip_boost && d.boostDelta > 0.05 && d.trvSetpoint != null)
      c.push(`<span class="chip warn">↑ ${d.trvSetpoint.toFixed(1)}° TRV ×${d.boostFactor.toFixed(2)}</span>`);
    if (!cfg.hide_chip_suppression && d.suppression > 0)
      c.push(`<span class="chip info">🌤️ −${d.suppression.toFixed(0)}%</span>`);
    if (!cfg.hide_chip_preheat && d.preheat > 0)
      c.push(`<span class="chip ${d.preheatActive ? 'pos' : 'warn'}">${d.preheatActive ? '🔥' : '⏱'} ${d.preheat.toFixed(0)} min</span>`);
    return c.length ? `<div class="chips">${c.join('')}</div>` : '';
  }

  _buildConf(d) {
    const pct = Math.min(100, Math.max(0, d.confidence)).toFixed(0);
    const lbl = d.confidence >= 80
      ? tr(this._hass, 'confident')
      : d.confidence >= 50
      ? tr(this._hass, 'learning')
      : tr(this._hass, 'collecting');
    const bar = d.stateKey === 'off' ? '#4fc3f7' : d.col.main;
    return `<div class="conf-row">
      <ha-icon icon="mdi:brain" style="--mdc-icon-size:13px;opacity:.5"></ha-icon>
      <span class="conf-lbl">${lbl}</span>
      <div class="conf-bg"><div class="conf-fill" style="width:${pct}%;background:${bar}"></div></div>
      <span class="conf-pct">${pct}%</span>
    </div>`;
  }

  // ── Sparkline chart ───────────────────────────────────────────────────────

  _buildSparkline(d) {
    if (this._historyFetching && !this._historyData) {
      return `<div class="spark-wrap spark-loading"><div class="spark-spinner"></div></div>`;
    }

    const data = this._historyData;
    if (!data || data.length < 3) return '';

    if (this._sparklineCache?.dataRef === this._historyData &&
        this._sparklineCache?.minTemp === this._config.min_temp &&
        this._sparklineCache?.maxTemp === this._config.max_temp &&
        Date.now() - (this._sparklineCache?.cachedAt ?? 0) < 60000) {
      return this._sparklineCache.html;
    }

    const W = 260, H = 90;
    const PL = 24, PR = 24, PT = 6, PB = 16;
    const vW = W - PL - PR, vH = H - PT - PB;

    const step = Math.max(1, Math.ceil(data.length / 100));
    const pts  = data.filter((_, i) => i % step === 0);
    if (pts.length < 2) return '';

    const allT = [
      ...pts.map(p => p.curr),
      ...pts.map(p => p.tgt),
      ...pts.map(p => p.outdoor),
    ].filter(t => !isNaN(t));
    if (!allT.length) return '';

    const rawMin = Math.min(...allT);
    const rawMax = Math.max(...allT);
    const minT = this._config.min_temp != null
      ? this._config.min_temp
      : Math.floor(rawMin / 5) * 5;
    const maxT = this._config.max_temp != null
      ? this._config.max_temp
      : (Math.ceil(rawMax / 5) * 5 || minT + 5);
    const range = maxT - minT || 1;

    // Time-based X axis: right = now, left = now - chart_hours
    const tNow   = Date.now();
    const tStart = tNow - (this._config.chart_hours ?? 24) * 3600 * 1000;
    const tRange = tNow - tStart;

    const tx = p  => (PL + Math.max(0, Math.min(1, (p.t - tStart) / tRange)) * vW).toFixed(1);
    const ty = tv => (PT + vH - ((tv - minT) / range) * vH).toFixed(1);

    const makePath = (get, locf = false) => {
      let last = NaN;
      return pts.reduce((acc, p) => {
        let v = get(p);
        if (isNaN(v) && locf) v = last;
        if (isNaN(v)) return acc;
        last = v;
        return acc + (acc === '' ? `M${tx(p)} ${ty(v)}` : ` L${tx(p)} ${ty(v)}`);
      }, '');
    };

    // Grid lines – Y-axis labels float inside chart (left-aligned, low opacity)
    let gridLines = '';
    for (let t = minT; t <= maxT; t += 5) {
      const y = parseFloat(ty(t)).toFixed(1);
      gridLines += `<line x1="${PL}" y1="${y}" x2="${W - PR}" y2="${y}"
        style="stroke:var(--divider-color,rgba(0,0,0,.1))" stroke-width="1"/>`;
      gridLines += `<text x="${PL - 3}" y="${y}" dominant-baseline="middle"
        text-anchor="end" style="font-size:8.5px;font-weight:500;fill:var(--secondary-text-color,#888);opacity:0.8">${t}°</text>`;
    }

    const currPath    = makePath(p => p.curr);
    const tgtPath     = makePath(p => p.tgt, true);  // LOCF: hold last valid target during window_open
    const outdoorPath = makePath(p => p.outdoor);

    const fmt = ts => {
      const d = new Date(ts);
      return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
    };

    // Legend
    const legendItems = [
      { col: CHART_COL_IST,    label: tr(this._hass, 'chart_actual')  },
      { col: CHART_COL_TGT,    label: tr(this._hass, 'chart_target')  },
    ];
    if (outdoorPath) {
      legendItems.push({ col: CHART_COL_OUTDOOR, label: tr(this._hass, 'chart_outdoor') });
    }
    // Legend rendered as HTML for proper flex-centering inside the card
    const legendHtml = legendItems.map(item => `
      <span class="spark-legend-item">
        <svg width="14" height="8" style="display:inline-block;vertical-align:middle;overflow:visible">
          <line x1="0" y1="4" x2="14" y2="4" stroke="${item.col}" stroke-width="2"/>
        </svg>
        <span>${item.label}</span>
      </span>`).join('');

    const result = `
      <div class="spark-wrap">
        <svg viewBox="0 0 ${W} ${H + 4}" style="width:100%;display:block;overflow:visible">
          <defs>
            <clipPath id="ts-spark-clip">
              <rect x="${PL}" y="${PT}" width="${vW}" height="${vH}"/>
            </clipPath>
          </defs>
          ${gridLines}
          <g clip-path="url(#ts-spark-clip)">
            ${tgtPath     ? `<path d="${tgtPath}"     fill="none" stroke="${CHART_COL_TGT}"     stroke-width="1.0" stroke-linecap="round" stroke-linejoin="round"/>` : ''}
            ${outdoorPath ? `<path d="${outdoorPath}" fill="none" stroke="${CHART_COL_OUTDOOR}" stroke-width="1.0" stroke-linecap="round" stroke-linejoin="round"/>` : ''}
            ${currPath    ? `<path d="${currPath}"    fill="none" stroke="${CHART_COL_IST}"     stroke-width="1.0" stroke-linecap="round" stroke-linejoin="round"/>` : ''}
          </g>
          <text class="spark-t-start" x="${PL}"     y="${H + 1}" text-anchor="start" style="font-size:9px;font-weight:500;fill:var(--secondary-text-color,#888)">${fmt(tStart)}</text>
          <text class="spark-t-end"   x="${W - PR}" y="${H + 1}" text-anchor="end"   style="font-size:9px;font-weight:500;fill:var(--secondary-text-color,#888)">${fmt(tNow)}</text>
        </svg>
        <div class="spark-legend">${legendHtml}</div>
      </div>`;

    this._sparklineCache = { dataRef: this._historyData, minTemp: this._config.min_temp, maxTemp: this._config.max_temp, cachedAt: Date.now(), html: result };
    return result;
  }

  _startChartTimer() {
    if (this._config.disable_chart) return;
    const hours = this._config.chart_hours ?? 24;
    const fmt = ts => {
      const d = new Date(ts);
      return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
    };
    const timer = setInterval(() => {
      const tNow   = Date.now();
      const tStart = tNow - hours * 3600 * 1000;
      const startEl = this.shadowRoot.querySelector('.spark-t-start');
      const endEl   = this.shadowRoot.querySelector('.spark-t-end');
      if (startEl) startEl.textContent = fmt(tStart);
      if (endEl)   endEl.textContent   = fmt(tNow);
    }, 60000);
    this._listeners.push(() => clearInterval(timer));
  }

  // ── Layout ───────────────────────────────────────────────────────────────

  _renderNormal(d) {
    return `
      <ha-card>
        <div class="card-body">
          <div class="hdr">
            <span class="hdr-name">${esc(d.name)}</span>
            <span class="hdr-readouts">
              ${d.outdoorTemp != null
                ? `<span class="hdr-out"><ha-icon icon="mdi:thermometer" style="--mdc-icon-size:14px"></ha-icon>${d.outdoorTemp.toFixed(1)}°C</span>` : ''}
            </span>
          </div>

          <div class="ring-wrap">
            ${this._buildRing(d)}
            ${this._buildTempOverlay(d)}
          </div>

          ${this._buildBanners(d)}

          <div class="ctrl-row">
            <button class="ctrl-btn sm${d.learnOn ? ' learn-active' : ' learn-off'}" data-action="learn"
              title="${d.learnOn ? tr(this._hass, 'learn_active') : tr(this._hass, 'learn_inactive')}">
              <ha-icon icon="mdi:brain"></ha-icon>
            </button>
            <button class="ctrl-btn" data-action="dec"><ha-icon icon="mdi:minus"></ha-icon></button>
            <span class="ctrl-val">${d.targetTemp != null ? d.targetTemp.toFixed(1) : '--'}</span>
            <button class="ctrl-btn" data-action="inc"><ha-icon icon="mdi:plus"></ha-icon></button>
            <button class="ctrl-btn sm${d.activeOn ? ' ctrl-active' : ' pwr-off'}" data-action="power">
              <ha-icon icon="mdi:power"></ha-icon>
            </button>
          </div>

          ${this._config.disable_modes ? '' : this._buildModes(d)}
          ${this._config.show_learning_progress === false ? '' : this._buildConf(d)}
        </div>
        ${this._config.disable_chart ? '<div style="padding-bottom:12px"></div>' : this._buildSparkline(d)}
      </ha-card>`;
  }


  // ── CSS ───────────────────────────────────────────────────────────────────

  _css() {
    return `<style>
      :host { display: block; }
      ha-card { padding: 0; box-sizing: border-box; }
      .card-body { padding: 14px 14px 0; }

      /* Header */
      .hdr { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; gap: 8px; }
      .hdr-name { font-size: 1em; font-weight: 600; color: var(--primary-text-color); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .hdr-readouts { display: flex; gap: 6px; flex-shrink: 0; }
      .hdr-out { display: inline-flex; align-items: center; gap: 2px; font-size: 0.75em; color: var(--secondary-text-color); background: var(--secondary-background-color, rgba(120,120,120,.1)); padding: 2px 9px; border-radius: 10px; white-space: nowrap; --mdc-icon-size: 14px; }

      /* Ring + Overlay */
      .ring-wrap { position: relative; display: flex; justify-content: center; margin-bottom: 2px; }
      .ring-svg { cursor: default; }
      .ring-svg.dragging, .ring-svg.dragging .arc-interact { cursor: grabbing !important; }
      .arc-interact { cursor: grab; }
      .ring-overlay {
        position: absolute; top: 59%; left: 50%;
        transform: translate(-50%, -50%);
        display: flex; flex-direction: column; align-items: center;
        pointer-events: none; width: 220px;
      }
      .ov-pill-slot { min-height: 28px; display: flex; justify-content: center; align-items: center; margin-bottom: 2px; }
      .ov-icon-badge { display: inline-flex; align-items: center; justify-content: center; }
      .ov-icon--summer   { color: #f57c00; filter: drop-shadow(0 0 6px rgba(255,152,0,.55)); }
      .ov-icon--window   { color: #29b6f6; filter: drop-shadow(0 0 6px rgba(41,182,246,.55)); }
      .ov-icon--obs      { color: #90a4ae; opacity: 0.7; }
      .ov-icon--vacation { color: #7986cb; filter: drop-shadow(0 0 6px rgba(121,134,203,.5)); }
      .ov-icon--preheat  { color: #e65100; filter: drop-shadow(0 0 6px rgba(230,81,0,.6)); }
      .ov-icon--obs.ov-icon--obs-learn { color: #43a047; animation: ts-icon-pulse 2.5s ease-in-out infinite; }
      @keyframes ts-icon-pulse { 0%,100%{filter:drop-shadow(0 0 3px rgba(67,160,71,.2))} 50%{filter:drop-shadow(0 0 9px rgba(67,160,71,.55))} }
      .ov-status { font-size: 0.9em; font-weight: 700; margin-bottom: 1px; white-space: nowrap; letter-spacing: 0.01em; }
      .ov-temp   { line-height: 1; white-space: nowrap; }
      .ov-int    { font-size: 4.8em; font-weight: 400; color: var(--primary-text-color); letter-spacing: -0.03em; }
      .ov-frac   { font-size: 1.6em; font-weight: 400; color: var(--primary-text-color); vertical-align: top; display: inline-block; margin-top: 0.46em; opacity: .85; }
      .ov-unit   { font-size: 0.65em; font-weight: 300; }
      .ov-target  { display: inline-flex; align-items: center; justify-content: center; gap: 4px; line-height: 1; font-size: 1.05em; font-weight: 500; color: var(--secondary-text-color); margin-top: 12px; --mdc-icon-size: 16px; }
      .ov-humidity{ display: inline-flex; align-items: center; justify-content: center; gap: 4px; line-height: 1; font-size: 1.05em; font-weight: 500; color: #6fa3d6; margin-top: 8px; --mdc-icon-size: 16px; }
      .ov-target ha-icon, .ov-humidity ha-icon { display: inline-flex; align-items: center; }

      /* Banners */
      .banner {
        display: flex; align-items: center; gap: 6px;
        border-radius: 8px; font-size: 0.78em; padding: 5px 10px; margin-bottom: 8px;
      }
      .banner.failure  { background: rgba(232,87,63,.1);  border: 1px solid rgba(232,87,63,.35);  color: #e8573f; }
      .banner.window   { background: rgba(41,182,246,.1); border: 1px solid rgba(41,182,246,.3);  color: #29b6f6; }
      .banner.vacation { background: rgba(121,134,203,.1);border: 1px solid rgba(121,134,203,.3); color: #7986cb; }
      .banner.battery  { background: rgba(232,87,63,.08); border: 1px solid rgba(232,87,63,.25);  color: #e8573f; }

      /* Controls */
      .ctrl-row { display: flex; align-items: center; justify-content: center; gap: 8px; margin: 4px 0 12px; }
      .ctrl-btn {
        width: 40px; height: 40px; border-radius: 50%; border: none;
        background: var(--secondary-background-color, rgba(120,120,120,.08));
        color: var(--primary-text-color); cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        --mdc-icon-size: 20px;
        transition: background 0.18s, color 0.18s, transform 0.12s;
        box-shadow: 0 1px 3px rgba(0,0,0,.08);
      }
      .ctrl-btn:hover  { background: var(--secondary-background-color, rgba(120,120,120,.16)); transform: scale(1.06); }
      .ctrl-btn:active { transform: scale(0.95); }
      .ctrl-btn.sm     { width: 34px; height: 34px; --mdc-icon-size: 17px; }
      .ctrl-btn.pwr-off    { color: #e8573f; background: rgba(232,87,63,.1); box-shadow: 0 0 14px rgba(232,87,63,.5), 0 0 5px rgba(232,87,63,.3); }
      .ctrl-btn.ctrl-active{ color: #1e88e5; background: rgba(30,136,229,.12); box-shadow: 0 0 14px rgba(30,136,229,.5), 0 0 5px rgba(30,136,229,.3); }
      .ctrl-btn.learn-active{ color: #43a047; background: rgba(67,160,71,.12); box-shadow: 0 0 14px rgba(67,160,71,.5), 0 0 5px rgba(67,160,71,.3); }
      .ctrl-btn.learn-off  { color: var(--secondary-text-color); opacity: 0.45; }
      .ctrl-val { font-size: 1.6em; font-weight: 500; color: var(--primary-text-color); min-width: 62px; text-align: center; letter-spacing: -0.01em; }

      /* Modes */
      .modes { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; }
      .mode-btn {
        flex: 1; min-width: 44px; height: 48px; border: none; border-radius: 12px; cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        background: transparent;
        color: var(--secondary-text-color);
        transition: color 0.18s, transform 0.14s, filter 0.18s, background 0.18s; --mdc-icon-size: 28px;
        opacity: 0.55;
      }
      .mode-btn.active  { color: var(--mt); opacity: 1; filter: drop-shadow(0 0 11px color-mix(in srgb, var(--mc) 80%, transparent)); }
      .mode-btn:hover:not(.active) { opacity: 0.8; transform: scale(1.1); background: rgba(120,120,120,.06); }
      .mode-btn:active  { transform: scale(0.92); }

      /* Chips */
      .chips { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }
      .chip { padding: 3px 10px; border-radius: 10px; font-size: 0.72em; background: var(--secondary-background-color, rgba(120,120,120,.1)); color: var(--secondary-text-color); }
      .chip.pos  { color: #43a047; }
      .chip.neg  { color: #e53935; }
      .chip.warn { color: #f57c00; }
      .chip.info { color: #1e88e5; }

      /* Confidence */
      .conf-row { display: flex; align-items: center; gap: 6px; font-size: 0.82em; color: var(--secondary-text-color); }
      .conf-lbl { flex-shrink: 0; }
      .conf-bg  { flex: 1; height: 4px; border-radius: 2px; background: var(--secondary-background-color, rgba(120,120,120,.12)); overflow: hidden; }
      .conf-fill { height: 100%; border-radius: 2px; transition: width 0.6s; }
      .conf-pct { min-width: 28px; text-align: right; flex-shrink: 0; font-weight: 500; }

      /* Sparkline */
      .spark-wrap { margin-top: 10px; padding: 8px 0 12px; border-top: 1px solid var(--divider-color, rgba(120,120,120,.2)); }
      .spark-legend { display: flex; justify-content: center; gap: 18px; padding: 2px 14px 0; }
      .spark-legend-item { display: inline-flex; align-items: center; gap: 5px; font-size: 0.82em; font-weight: 500; color: var(--secondary-text-color,#888); }
      .spark-loading { display: flex; align-items: center; justify-content: center; height: 56px; }
      .spark-spinner {
        width: 20px; height: 20px; border-radius: 50%;
        border: 2px solid var(--secondary-background-color, rgba(120,120,120,.15));
        border-top-color: var(--secondary-text-color, #888);
        animation: ts-spin 0.8s linear infinite;
      }
      @keyframes ts-spin { to { transform: rotate(360deg); } }

    </style>`;
  }

  // ── Render ────────────────────────────────────────────────────────────────

  _findThermosmartEntity() {
    return Object.entries(this._hass?.states || {})
      .find(([eid, st]) => {
        if (!eid.startsWith('climate.')) return false;
        const a = st?.attributes || {};
        return eid.startsWith('climate.thermosmart') ||
          ('learning_confidence' in a && 'observation_mode' in a);
      })?.[0] ?? null;
  }

  _render() {
    this._cleanup();
    let d = this._getData();

    if (!d) {
      if (!this._config?.entity) {
        const candidate = this._findThermosmartEntity();
        if (candidate) {
          this._config = { ...this._config, entity: candidate };
          d = this._getData();
        }
      }
      if (!d) {
        const hasEntity = !!this._config?.entity;
        this.shadowRoot.innerHTML = hasEntity
          ? `${this._css()}
            <ha-card style="padding:16px">
              <div style="color:var(--error-color,#e8573f);font-size:.85em">
                ${tr(this._hass, 'not_found')}: <b>${esc(this._config.entity)}</b>
              </div>
            </ha-card>`
          : `${this._css()}
            <ha-card style="padding:24px 16px;text-align:center">
              <div style="font-size:.95em;font-weight:600;color:var(--primary-text-color,#333);margin-bottom:6px">ThermoSmart Card</div>
              <div style="font-size:.82em;color:var(--secondary-text-color,#888);line-height:1.5">${tr(this._hass, 'no_entity')}</div>
            </ha-card>`;
        return;
      }
    }

    this._minTemp = d.entityMinTemp;
    this._maxTemp = d.entityMaxTemp;

    const body = this._renderNormal(d);
    this.shadowRoot.innerHTML = this._css() + body;

    // Mode buttons – optimistic preset for instant visual feedback
    this.shadowRoot.querySelectorAll('.mode-btn').forEach(btn => {
      const h = () => {
        const preset = btn.dataset.preset;
        this._optimisticPreset = preset;
        clearTimeout(this._optimisticPresetTimer);
        this._optimisticPresetTimer = setTimeout(() => { this._optimisticPreset = null; }, 10000);
        this._hass.callService('climate', 'set_preset_mode', { entity_id: d.entityId, preset_mode: preset });
        this._render();
      };
      btn.addEventListener('click', h);
      this._listeners.push(() => btn.removeEventListener('click', h));
    });

    // Control buttons
    const setTempDebounced = debounce((temp) => {
      this._hass.callService('climate', 'set_temperature', { entity_id: d.entityId, temperature: temp });
    }, 300);

    this.shadowRoot.querySelectorAll('.ctrl-btn[data-action]').forEach(btn => {
      const h = () => {
        const act = btn.dataset.action;
        if (act === 'inc' || act === 'dec') {
          const base = this._optimisticTemp ?? d.targetTemp ?? 20;
          const t = Math.max(this._minTemp, Math.min(this._maxTemp,
            base + (act === 'inc' ? 0.5 : -0.5)
          ));
          this._setOptimistic(t);
          this._updateDragUI(t);
          setTempDebounced(t);
        } else if (act === 'learn') {
          if (this._learnSwitch) {
            const on = this._hass.states[this._learnSwitch]?.state === 'on';
            this._hass.callService('switch', on ? 'turn_off' : 'turn_on', { entity_id: this._learnSwitch });
          } else {
            this._hass.callService('climate', 'set_hvac_mode', { entity_id: d.entityId, hvac_mode: d.isObs ? 'heat' : 'off' });
          }
        } else if (act === 'power') {
          if (this._activeSwitch) {
            const on = this._hass.states[this._activeSwitch]?.state === 'on';
            this._hass.callService('switch', on ? 'turn_off' : 'turn_on', { entity_id: this._activeSwitch });
          } else {
            this._hass.callService('climate', 'set_hvac_mode', { entity_id: d.entityId, hvac_mode: d.isObs ? 'heat' : 'off' });
          }
        }
      };
      btn.addEventListener('click', h);
      this._listeners.push(() => btn.removeEventListener('click', h));
    });

    this._setupDrag(d);
    this._startChartTimer();
  }

  // ── Arc drag ──────────────────────────────────────────────────────────────

  _setupDrag(d) {
    const svg = this.shadowRoot.querySelector('.ring-svg');
    if (!svg) return;

    const MIN_T = this._minTemp;
    const MAX_T = this._maxTemp;

    const getAngle = (clientX, clientY) => {
      const rect = svg.getBoundingClientRect();
      const svgX = (clientX - rect.left) * (280 / rect.width);
      const svgY = (clientY - rect.top)  * (196 / rect.height);
      let angle = Math.atan2(svgY - CY, svgX - CX) * 180 / Math.PI + 90;
      if (angle < 0) angle += 360;
      return angle;
    };

    const distFromRing = (clientX, clientY) => {
      const rect = svg.getBoundingClientRect();
      const svgX = (clientX - rect.left) * (280 / rect.width);
      const svgY = (clientY - rect.top)  * (196 / rect.height);
      return Math.abs(Math.hypot(svgX - CX, svgY - CY) - R);
    };

    const angleToTemp = (angle) => {
      let rel = angle - AF;
      if (rel < 0) rel += 360;
      rel = Math.max(0, Math.min(AS, rel));
      return Math.round((MIN_T + (rel / AS) * (MAX_T - MIN_T)) * 2) / 2;
    };

    let dragging = false;
    let lastTemp = null;

    const callSet = debounce((temp) => {
      this._hass.callService('climate', 'set_temperature', {
        entity_id: d.entityId, temperature: temp,
      });
    }, 600);

    const onDown = (e) => {
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      if (distFromRing(clientX, clientY) > 24) return;
      const angle = getAngle(clientX, clientY);
      if (angle > (AF + AS) % 360 && angle < AF) return; // reject open gap (120°–240°)
      dragging = true; this._isDragging = true;
      svg.classList.add('dragging');
      const temp = angleToTemp(angle);
      lastTemp = temp; this._dragTemp = temp;
      this._updateDragUI(temp);
      callSet(temp);
      e.preventDefault();
    };

    const onMove = (e) => {
      if (!dragging) return;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      const temp = angleToTemp(getAngle(clientX, clientY));
      if (temp !== lastTemp) {
        lastTemp = temp; this._dragTemp = temp;
        this._updateDragUI(temp);
        callSet(temp);
      }
      e.preventDefault();
    };

    const onUp = () => {
      if (!dragging) return;
      dragging = false; this._isDragging = false;
      if (lastTemp !== null) this._setOptimistic(lastTemp);
      this._dragTemp = null;
      svg.classList.remove('dragging');
    };

    svg.addEventListener('mousedown',  onDown);
    svg.addEventListener('touchstart', onDown, { passive: false });
    window.addEventListener('mousemove',  onMove);
    window.addEventListener('touchmove',  onMove, { passive: false });
    window.addEventListener('mouseup',    onUp);
    window.addEventListener('touchend',   onUp);

    this._listeners.push(
      () => svg.removeEventListener('mousedown',  onDown),
      () => svg.removeEventListener('touchstart', onDown),
      () => window.removeEventListener('mousemove',  onMove),
      () => window.removeEventListener('touchmove',  onMove),
      () => window.removeEventListener('mouseup',    onUp),
      () => window.removeEventListener('touchend',   onUp),
    );
  }

  _updateDragUI(temp) {
    const MIN_T = this._minTemp;
    const MAX_T = this._maxTemp;
    const frac  = Math.max(0, Math.min(1, (temp - MIN_T) / (MAX_T - MIN_T)));
    const p     = polarToXY(CX, CY, R, AF + frac * AS);
    const sh    = this.shadowRoot;

    const valEl = sh.querySelector('.ctrl-val');
    if (valEl) valEl.textContent = temp.toFixed(1);

    const dot = sh.querySelector('.ring-dot');
    if (dot) {
      dot.setAttribute('cx', p.x.toFixed(1));
      dot.setAttribute('cy', p.y.toFixed(1));
    }

    if (this._config.invert_temps) {
      const intPart = Math.floor(Math.abs(temp)) * (temp < 0 ? -1 : 1);
      const decPart = Math.round(Math.abs(temp - Math.trunc(temp)) * 10);
      const intEl = sh.querySelector('.ov-int');
      const fracEl = sh.querySelector('.ov-frac');
      if (intEl) intEl.textContent = String(intPart);
      if (fracEl) fracEl.innerHTML = `.${decPart}<span class="ov-unit">°C</span>`;
    } else {
      const tgtEl = sh.querySelector('.ov-target');
      if (tgtEl) {
        const rawTrv = this._hass?.states[this._config?.entity]?.attributes?.trv_setpoint;
        const trvVal = (this._config.show_trv_setpoint && rawTrv != null) ? Number(rawTrv) : null;
        tgtEl.style.visibility = '';
        tgtEl.innerHTML = `<ha-icon icon="mdi:thermostat" style="--mdc-icon-size:16px"></ha-icon>${temp.toFixed(1)}°`
          + (trvVal != null && !isNaN(trvVal) ? `<ha-icon icon="mdi:valve" style="--mdc-icon-size:16px;margin-left:10px"></ha-icon>${trvVal.toFixed(1)}°` : '');
      }
    }
  }
}

if (!customElements.get('thermosmart-card'))
  customElements.define('thermosmart-card', ThermosmartCard);

window.customCards = window.customCards || [];
if (!window.customCards.some(c => c.type === 'thermosmart-card')) {
  window.customCards.push({
    type: 'thermosmart-card',
    name: 'ThermoSmart Card',
    description: 'Visual companion for ThermoSmart — displays self-learning heating control, forecasts and thermal behaviour in Home Assistant.',
    preview: true,
    documentationURL: 'https://github.com/Mikasmarthome/thermosmart-card',
  });
}

console.info(
  `%c THERMOSMART-CARD %c v${CARD_VERSION} `,
  'background:#e8573f;color:#fff;padding:2px 6px;border-radius:3px 0 0 3px;font-weight:bold;font-size:11px',
  'background:#2a2a2a;color:#fff;padding:2px 6px;border-radius:0 3px 3px 0;font-size:11px',
);
