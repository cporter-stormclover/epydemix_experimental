data/incidence.csv holds the daily number of new infections (column cases)
of an epidemic in a population of 100,000 people, from 2024-01-01 to
2024-02-09. At the start, 10 people were infected. Assume an SIR model with
a recovery rate of 0.1 per day.

1. Estimate the transmission rate by ABC calibration against the incidence.
2. Project the epidemic 60 days beyond the data (to 2024-04-09), sampling
   from the posterior, with contacts halved from 2024-02-10 onwards.

Save the calibration as outputs/calibration.epx and the projection as
outputs/projection.epx. Report the posterior median transmission rate with
its 95% interval, and the projected peak of infections.
