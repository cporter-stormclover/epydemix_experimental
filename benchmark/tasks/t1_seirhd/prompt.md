Model a respiratory outbreak in a single, well-mixed population of 1,000,000
people with compartments S, E, I, H (hospitalized), R and D (dead).

- Infectious people (I) transmit at a rate of 0.3 per day. Hospitalized
  patients (H) also infect susceptible people, at a nosocomial rate of 0.05
  per day.
- The latent period lasts 3 days on average.
- The infectious period lasts 5 days; at its end, 10% of cases are
  hospitalized and the others recover.
- A hospital stay lasts 8 days; 20% of hospitalized patients die, the others
  recover.

Start on 2026-01-01 with 100 infectious people and everyone else susceptible,
and simulate 180 days with 100 simulations. Use the compartment names above.
Save the result as outputs/seirhd.epx. Report the peak hospital occupancy
(median and 90% interval), the date of the peak, and the total number of
deaths.
