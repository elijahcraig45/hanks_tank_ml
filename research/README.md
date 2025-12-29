# Research & Feature Engineering

This directory contains research, brainstorming, and prototyping for the `hanks_tank_ml` project.

## Contents

*   **[competitor_analysis.md](competitor_analysis.md)**: A guide to open-source tools (pybaseball), accuracy benchmarks (aim for 55%), and lessons from famous systems like PECOTA and ZiPS.
*   **[model_deep_dives.md](model_deep_dives.md)**: Detailed breakdown of the methodology, strengths, and weaknesses of top projection systems (PECOTA, ZiPS, Steamer, THE BAT, ATC).
*   **[moneyball_principles.md](moneyball_principles.md)**: A breakdown of core Moneyball concepts, key statistics, and how to translate them into machine learning features for game prediction.
*   **[war_deep_dive.md](war_deep_dive.md)**: A detailed comparison of bWAR vs. fWAR, focusing on their predictive power and how to engineer them into daily model features.
*   **[park_and_weather_factors.md](park_and_weather_factors.md)**: Analysis of how stadium dimensions, altitude, and weather (temp, wind, density) affect game outcomes, including data sources and feature engineering ideas.
*   **[astrology_and_calendar_effects.md](astrology_and_calendar_effects.md)**: Exploration of "fringe" features like Moon Phases, Zodiac signs, Biorhythms, and Circadian Rhythms (Jet Lag) for fun and potential hidden correlations.
*   **[basic_stats_features.md](basic_stats_features.md)**: A guide to traditional "back of the baseball card" statistics (AVG, ERA, HR) and how to transform them into useful ML features (Rolling Averages, Splits).
*   **[advanced_sabermetrics.md](advanced_sabermetrics.md)**: Deep dive into modern metrics (wOBA, wRC+, FIP, SIERA) that isolate skill from luck, providing the strongest predictive signals for the model.
*   **[fangraphs_guts.md](fangraphs_guts.md)**: Explanation of the "Guts" constants (wOBA Scale, cFIP, Run Values) required to correctly calculate advanced stats for different seasons.

## Goals

1.  Identify predictive features for game outcomes.
2.  Develop player performance metrics (daily, weekly, seasonal).
3.  Bridge the gap between traditional Sabermetrics and modern ML techniques.
