# TRIPOD+AI Reporting Checklist

**Manuscript title:** Routine urine dipstick analysis for opportunistic hyperuricemia screening: development and external validation of calibrated machine learning models

**Corresponding author:** Kiheon Lee, MD, PhD

**Reference:** Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378.

---

TRIPOD+AI: Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis with Artificial Intelligence.

| Section/Topic | Item | Development/Evaluation | Checklist Item | Reported on page |
|---|---|---|---|---|
| **TITLE** | | | | |
| Title | 1 | D;E | Identify the study as developing or evaluating the performance of a multivariable prediction model, the target population, and the outcome to be predicted | 1 |
| **ABSTRACT** | | | | |
| Abstract | 2 | D;E | See TRIPOD+AI for Abstracts checklist | 2 |
| **INTRODUCTION** | | | | |
| Background | 3a | D;E | Explain the healthcare context (including whether diagnostic or prognostic) and rationale for developing or evaluating the prediction model, including references to existing models | 3 |
| | 3b | D;E | Describe the target population and the intended purpose of the prediction model in the context of the care pathway, including its intended users (e.g., healthcare professionals, patients, public) | 3 |
| | 3c | D;E | Describe any known health inequalities between sociodemographic groups | N/A |
| Objectives | 4 | D;E | Specify the study objectives, including whether the study describes the development or validation of a prediction model (or both) | 4 |
| **METHODS** | | | | |
| Data | 5a | D;E | Describe the sources of data separately for the development and evaluation datasets (e.g., randomised trial, cohort, routine care or registry data), the rationale for using these data, and representativeness of the data | 4 |
| | 5b | D;E | Specify the dates of the collected participant data, including start and end of participant accrual; and, if applicable, end of follow-up | 4 |
| Participants | 6a | D;E | Specify key elements of the study setting (e.g., primary care, secondary care, general population) including the number and location of centres | 4 |
| | 6b | D;E | Describe the eligibility criteria for study participants | 4 |
| | 6c | D;E | Give details of any treatments received, and how they were handled during model development or evaluation, if relevant | N/A |
| Data preparation | 7 | D;E | Describe any data pre-processing and quality checking, including whether this was similar across relevant sociodemographic groups | 5 |
| Outcome | 8a | D;E | Clearly define the outcome that is being predicted and the time horizon, including how and when assessed, the rationale for choosing this outcome, and whether the method of outcome assessment is consistent across sociodemographic groups | 5 |
| | 8b | D;E | If outcome assessment requires subjective interpretation, describe the qualifications and demographic characteristics of the outcome assessors | N/A |
| | 8c | D;E | Report any actions to blind assessment of the outcome to be predicted | N/A |
| Predictors | 9a | D | Describe the choice of initial predictors (e.g., literature, previous models, all available predictors) and any pre-selection of predictors before model building | 5 |
| | 9b | D;E | Clearly define all predictors, including how and when they were measured (and any actions to blind assessment of predictors for the outcome and other predictors) | 5 |
| | 9c | D;E | If predictor measurement requires subjective interpretation, describe the qualifications and demographic characteristics of the predictor assessors | N/A |
| Sample size | 10 | D;E | Explain how the study size was arrived at (separately for development and evaluation), and justify that the study size was sufficient to answer the research question. Include details of any sample size calculation | 4 |
| Missing data | 11 | D;E | Describe how missing data were handled. Provide reasons for omitting any data | 5 |
| Analytical methods | 12a | D | Describe how the data were used (e.g., for development and evaluation of model performance) in the analysis, including whether the data were partitioned, considering any sample size requirements | 6 |
| | 12b | D | Depending on the type of model, describe how predictors were handled in the analyses (functional form, rescaling, transformation, or any standardisation) | 5 |
| | 12c | D | Specify the type of model, rationale, all model-building steps, including any hyperparameter tuning, and method for internal validation | 6 |
| | 12d | D;E | Describe if and how any heterogeneity in estimates of model parameter values and model performance was handled and quantified across clusters | N/A |
| | 12e | D;E | Specify all measures and plots used (and their rationale) to evaluate model performance (e.g., discrimination, calibration, clinical utility) and, if relevant, to compare multiple models | 6 |
| | 12f | E | Describe any model updating (e.g., recalibration) arising from the model evaluation, either overall or for particular sociodemographic groups or settings | 6 |
| | 12g | E | For model evaluation, describe how the model predictions were calculated (e.g., formula, code, object, application programming interface) | 6 |
| Class imbalance | 13 | D;E | If class imbalance methods were used, state why and how this was done, and any subsequent methods to recalibrate the model or the model predictions | N/A |
| Fairness | 14 | D;E | Describe any approaches that were used to address model fairness and their rationale | 7, 9 |
| Model output | 15 | D | Specify the output of the prediction model (e.g., probabilities, classification). Provide details and rationale for any classification and how the thresholds were identified | 5, Table 2 |
| Training versus evaluation | 16 | D;E | Identify any differences between the development and evaluation data in healthcare setting, eligibility criteria, outcome, and predictors | 7 |
| Ethical approval | 17 | D;E | Name the institutional research board or ethics committee that approved the study and describe the participant-informed consent or the ethics committee waiver of informed consent | 4 |
| **OPEN SCIENCE** | | | | |
| Funding | 18a | D;E | Give the source of funding and the role of the funders for the present study | 10 |
| Conflicts of interest | 18b | D;E | Declare any conflicts of interest and financial disclosures for all authors | 10 |
| Protocol | 18c | D;E | Indicate where the study protocol can be accessed or state that a protocol was not prepared | N/A |
| Registration | 18d | D;E | Provide registration information for the study, including register name and registration number, or state that the study was not registered | N/A |
| Data sharing | 18e | D;E | Provide details of the availability of the study data | 6 |
| Code sharing | 18f | D;E | Provide details of the availability of the analytical code | 6 |
| **PATIENT & PUBLIC INVOLVEMENT** | | | | |
| Patient & Public Involvement | 19 | D;E | Provide details of any patient and public involvement during the design, conduct, reporting, interpretation, or dissemination of the study or state no involvement | N/A |
| **RESULTS** | | | | |
| Participants | 20a | D;E | Describe the flow of participants through the study, including the number of participants with and without the outcome and, if applicable, a summary of the follow-up time. A diagram may be helpful | 7, eFigure 1 |
| | 20b | D;E | Report the characteristics overall and, where applicable, for each data source or setting, including the key dates, key predictors (including demographics), treatments received, sample size, number of outcome events, follow-up time, and amount of missing data. A table may be helpful. Report any differences across key demographic groups | 7, Table 1 |
| | 20c | E | For model evaluation, show a comparison with the development data of the distribution of important predictors (demographics, predictors, and outcome) | 7, Table 1 |
| Model development | 21 | D;E | Specify the number of participants and outcome events in each analysis (e.g., for model development, hyperparameter tuning, model evaluation) | 7, Table 1, Table 2 |
| Model specification | 22 | D | Provide details of the full prediction model (e.g., formula, code, object, application programming interface) to allow predictions in new individuals and to enable third-party evaluation and implementation, including any restrictions to access or re-use (e.g., freely available, proprietary) | 6 |
| Model performance | 23a | D;E | Report model performance estimates with confidence intervals, including for any key subgroups (e.g., sociodemographic). Consider plots to aid presentation | 7–8, Table 2, Figure 1, Figure 2 |
| | 23b | D;E | If examined, report results of any heterogeneity in model performance across clusters | N/A |
| Model updating | 24 | E | Report the results from any model updating, including the updated model and subsequent performance | 6, Table 2 (Brier scores) |
| **DISCUSSION** | | | | |
| Interpretation | 25 | D;E | Give an overall interpretation of the main results, including issues of fairness in the context of the objectives and previous studies | 8–9 |
| Limitations | 26 | D;E | Discuss any limitations of the study (such as a non-representative sample, sample size, overfitting, missing data) and their effects on any biases, statistical uncertainty, and generalizability | 9 |
| Usability of the model in the context of current care | 27a | D | Describe how poor quality or unavailable input data (e.g., predictor values) should be assessed and handled when implementing the prediction model | 9 |
| | 27b | D | Specify whether users will be required to interact in the handling of the input data or use of the model, and what level of expertise is required of users | 9 |
| | 27c | D;E | Discuss any next steps for future research, with a specific view to applicability and generalizability of the model | 9 |

Abbreviations: TRIPOD+AI, Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis with Artificial Intelligence; D, development; E, evaluation; N/A, not applicable.

Notes:
- Item 3c: Health inequalities between sociodemographic groups not explicitly addressed; female-specific performance reported separately on p.7 (Subgroup Analysis).
- Item 14 (Fairness): Subgroup analyses by age and sex reported on p.7; sex-specific performance for severe-end thresholds discussed on p.9.
- Item 24 (Model updating): Isotonic recalibration applied during development (p.6); post-recalibration Brier scores reported in Table 2.
