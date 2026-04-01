from typing import Dict

# Correlation matrix: ticker -> {ticker -> float}
CorrelationMatrix = Dict[str, Dict[str, float]]
