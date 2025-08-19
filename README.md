# Thermal Sensation and Comfort Model (TSCM)

This repository contains the code for the Thermal Sensation and Comfort Model (TSCM).
TSCM is a computational framework designed to predict human thermal responses under non-uniform and transient environments.
Unlike traditional indices that assume uniform and steady conditions, TSCM incorporates dynamic and spatially asymmetric effects,
making it suitable for both indoor and outdoor comfort studies. Currently, a five-part series has been developed:
1. **[Part 1: Local Sensation Models](https://doi.org/10.1016/j.buildenv.2009.06.018)** – logistic regression linking skin/core temperatures to local sensations.
2. **[Part 2: Local Comfort Models](https://doi.org/10.1016/j.buildenv.2009.06.015)** – asymmetrical curves mapping local sensations to local comfort.
3. **[Part 3: Overall Models](https://doi.org/10.1016/j.buildenv.2009.06.020)** - pieced methods combining local perceptions into overall sensation/comfort.
4. **[Part 4: Smoothed Overall Sensation Models](https://doi.org/10.1016/j.buildenv.2013.11.004)** - ensuring continuity in time-sequential predictions.
5. **[Part 5: Enhanced Overall Sensation Models](https://doi.org/10.1016/j.buildenv.2025.113562)** - further addressing discontinuity and inaccuracy under “dominated cold” scenarios in warm environments.

## References
- Zhang, H., Arens, E., Huizenga, C., & Han, T. (2010). Thermal sensation and comfort models for non-uniform and transient environments: Part I: Local sensation of individual body parts. Building and Environment, 45(2), 380–388. https://doi.org/10.1016/j.buildenv.2009.06.018
- Zhang, H., Arens, E., Huizenga, C., & Han, T. (2010). Thermal sensation and comfort models for non-uniform and transient environments, part II: Local comfort of individual body parts. Building and Environment, 45(2), 389–398. https://doi.org/10.1016/j.buildenv.2009.06.015
- Zhang, H., Arens, E., Huizenga, C., & Han, T. (2010). Thermal sensation and comfort models for non-uniform and transient environments, part III: Whole-body sensation and comfort. Building and Environment, 45(2), 399–410. https://doi.org/10.1016/j.buildenv.2009.06.020
- Zhao, Y., Zhang, H., Arens, E. A., & Zhao, Q. (2014). Thermal sensation and comfort models for non-uniform and transient environments, part IV: Adaptive neutral setpoints and smoothed whole-body sensation model. Building and Environment, 72, 300–308. https://doi.org/10.1016/j.buildenv.2013.11.004
- Lin, J., Liang, Y., Yang, J., Xie, Y., Zhang, H., & Niu, J. (2025). Thermal sensation and comfort models for non-uniform and transient environments, part V: Enhancements of whole-body sensation model. Building and Environment, 113562. https://doi.org/10.1016/j.buildenv.2025.113562
