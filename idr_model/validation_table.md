# Validation Table (Article 2025)

## 1) Input Conditions From Article

| field | value | unit | comment |
|---|---:|---|---|
| plasma_medium | 3% (NH4)2SO4 solution | - | Liquid plasma-forming medium |
| frequency | 1.76e6 | Hz | RF excitation |
| pressure_min | 1000 | Pa | Experimental range |
| pressure_max | 10000 | Pa | Experimental range |
| jet_length | 40 | mm | Table 1 |
| jet_diameter | 1.5 | mm | Table 1 |
| jet_velocity | 0.89 | m/s | Table 1 |
| electrolyte_flow | 1.57 | ml/s | Table 1 |

## 2) Validation Targets (By Pressure)

| pressure_pa | U_kV_exp | I_A_exp | P_kW_exp | j_kA_m2_exp | P_ud_MW_m2_exp | ne_cm3_exp_Hb | regime_exp | can_validate_now | note |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1000 | 0.86 | 1.48 | 1.27 | 837 | 0.72 | 2.2e16 | volume cone + boiling | partial | Current model can compare pressure and ne trend only |
| 5000 | 0.86 | 1.48 | 1.27 | 837 | 0.72 | 2.2e16 | continuous filaments | partial | No RF circuit, no jet morphology model |
| 10000 | 0.68 | 1.20 | 0.82 | 679 | 0.46 | 2.2e16 | pulsing microdischarges | partial | No direct U/I morphology prediction |

## 3) Global Diagnostics Targets

| metric | exp_value | unit | can_validate_now | note |
|---|---:|---|---|---|
| external_temperature_max | 47.6 | C | no | No thermal model of setup walls |
| plasma_composition | Cu I, H I, Na I, Mg I, O I, N I, N2+ | - | no | No plasma chemistry/spectroscopy block |

