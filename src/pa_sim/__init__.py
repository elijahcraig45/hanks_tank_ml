"""Plate-appearance simulation for MLB game prediction.

Components:
  rates   -- empirical-Bayes player rates + log5 odds-ratio matchup combination
  markov  -- 24-state base-out chain -> per-inning run distribution
  usage   -- starter hook curve + bullpen composite
  game    -- convolve two run distributions -> P(home win)
"""
CLASSES = ["K", "BB", "1B", "2B", "3B", "HR", "OUT", "DP"]
