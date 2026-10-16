# Multimedia Appendix 2

**TRIPOD+AI and TRIPOD+AI for Abstracts checklists for: Machine Learning Models Combining Routine Urine Dipstick Results With Age, Sex, and Body Mass Index to Triage Serum Urate Testing: Development and External Validation Study.**

Checklist items from Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. BMJ 2024;385:e078378. Locations refer to section headings of the manuscript, tables (Table n) and figures (Figure n) of the main text, and supplementary tables and figures (Table Sn, Figure Sn) of Multimedia Appendix 1; a panel letter (for example, Table S3 panel A) refers to one panel of a supplementary table. D: development; E: evaluation.

## TRIPOD+AI

| Section | Item | D/E | Checklist item | Where reported |
|---|---|---|---|---|
| **Title** |  |  |  |  |
| Title | 1 | D;E | Identify the study as developing or evaluating the performance of a multivariable prediction model, the target population, and the outcome to be predicted | Title |
| **Abstract** |  |  |  |  |
| Abstract | 2 | D;E | See TRIPOD+AI for Abstracts checklist | Abstract (see TRIPOD+AI for Abstracts below) |
| **Introduction** |  |  |  |  |
| Background | 3a | D;E | Explain the healthcare context (including whether diagnostic or prognostic) and rationale for developing or evaluating the prediction model, including references to existing models | Introduction, paragraphs 1-2 and 4 |
|  | 3b | D;E | Describe the target population and the intended purpose of the prediction model in the context of the care pathway, including its intended users (e.g., healthcare professionals, patients, public) | Introduction, paragraphs 2 and 4; Methods, Development Cohort (intended population: adults); Discussion, Principal Results and Implications for Implementation |
|  | 3c | D;E | Describe any known health inequalities between sociodemographic groups | Results, Clinical Utility (operating points by sex); Table 3; Discussion, Principal Results |
| Objectives | 4 | D;E | Specify the study objectives, including whether the study describes the development or validation of a prediction model (or both) | Introduction, paragraph 4 |
| **Methods** |  |  |  |  |
| Data | 5a | D;E | Describe the sources of data separately for the development and evaluation datasets (e.g., randomised trial, cohort, routine care or registry data), the rationale for using these data, and representativeness of the data | Methods, Data Sources and Participants; Results, Study Population and Setting; Table S1 panels B-D |
|  | 5b | D;E | Specify the dates of the collected participant data, including start and end of participant accrual; and, if applicable, end of follow-up | Methods, Development Cohort and External Validation Cohort |
| Participants | 6a | D;E | Specify key elements of the study setting (e.g., primary care, secondary care, general population) including the number and location of centres | Methods, Development Cohort (single tertiary hospital; setting marker) and External Validation Cohort; Table 1; Table S1 panels B and C |
|  | 6b | D;E | Describe the eligibility criteria for study participants | Methods, Development Cohort (adults with serum urate and BMI recorded on the day of the urine test), Training and Internal Test Sets, and External Validation Cohort; Figure 1 |
|  | 6c | D;E | Give details of any treatments received, and how they were handled during model development or evaluation, if relevant | Not applicable (treatments not recorded); Discussion, Limitations (first) |
| Data preparation | 7 | D;E | Describe any data pre-processing and quality checking, including whether this was similar across relevant sociodemographic groups | Methods, Input Features (dipstick coding; KNHANES codes mapped to the development grade with the same label, a mapping fixed before the models were fitted or evaluated); Tables S2 and S3 |
| Outcome | 8a | D;E | Clearly define the outcome that is being predicted and the time horizon, including how and when assessed, the rationale for choosing this outcome, and whether the method of outcome assessment is consistent across sociodemographic groups | Methods, Outcomes (serum urate measured on the day of the urine test); Discussion, Limitations (first) |
|  | 8b | D;E | If outcome assessment requires subjective interpretation, describe the qualifications and demographic characteristics of the outcome assessors | Not applicable (laboratory measurement) |
|  | 8c | D;E | Report any actions to blind assessment of the outcome to be predicted | Not applicable (laboratory measurement) |
| Predictors | 9a | D | Describe the choice of initial predictors (e.g., literature, previous models, all available predictors) and any pre-selection of predictors before model building | Methods, Input Features (12 inputs: the 9 dipstick fields measured in both cohorts, age, sex, and BMI; leukocyte esterase is not an input because KNHANES does not measure it) |
|  | 9b | D;E | Clearly define all predictors, including how and when they were measured (and any actions to blind assessment of predictors for the outcome and other predictors) | Methods, Development Cohort (serum urate and BMI recorded on the day of the urine test; BMI from a same-day health checkup or the nursing record) and Input Features; Table S2 and Table S3 panel A |
|  | 9c | D;E | If predictor measurement requires subjective interpretation, describe the qualifications and demographic characteristics of the predictor assessors | Not applicable (laboratory and anthropometric measurements) |
| Sample size | 10 | D;E | Explain how the study size was arrived at (separately for development and evaluation), and justify that the study size was sufficient to answer the research question. Include details of any sample size calculation | Methods, External Validation Cohort (no sample size calculation; events against published minimums) |
| Missing data | 11 | D;E | Describe how missing data were handled. Provide reasons for omitting any data | Methods, Missing Data; Results, Missing Data; Table S5 |
| Analytical methods | 12a | D | Describe how the data were used (e.g., for development and evaluation of model performance) in the analysis, including whether the data were partitioned, considering any sample size requirements | Methods, Training and Internal Test Sets (one person-level split, stratified on serum urate and shared by the thresholds); Added Value of the Dipstick |
|  | 12b | D | Depending on the type of model, describe how predictors were handled in the analyses (functional form, rescaling, transformation, or any standardisation) | Methods, Input Features |
|  | 12c | D | Specify the type of model, rationale, all model-building steps, including any hyperparameter tuning, and method for internal validation | Methods, Model Development (hyperparameters tuned on the training persons only; data, inputs, split, KNHANES coding, and modeling procedure fixed before the models were fitted or evaluated); Table S8 |
|  | 12d | D;E | Describe if and how any heterogeneity in estimates of model parameter values and model performance was handled and quantified across clusters | Not applicable (single development site); survey periods and years in Methods, Sensitivity Analyses for Coding, Linkage, and Time |
|  | 12e | D;E | Specify all measures and plots used (and their rationale) to evaluate model performance (e.g., discrimination, calibration, clinical utility) and, if relevant, to compare multiple models | Methods, Statistical Analysis (Discrimination and Calibration; Confidence Intervals and Comparisons; Clinical Utility; Added Value) |
|  | 12f | E | Describe any model updating (e.g., recalibration) arising from the model evaluation, either overall or for particular sociodemographic groups or settings | Methods, Discrimination and Calibration (external intercept update); Clinical Utility (cross-fitted update) |
|  | 12g | E | For model evaluation, describe how the model predictions were calculated (e.g., formula, code, object, application programming interface) | Methods, Model Development and Software (predictions computed with the saved model files) |
| Class imbalance | 13 | D;E | If class imbalance methods were used, state why and how this was done, and any subsequent methods to recalibrate the model or the model predictions | Not applicable (no class-imbalance methods) |
| Fairness | 14 | D;E | Describe any approaches that were used to address model fairness and their rationale | Methods, Clinical Utility (operating points by sex, sex-specific thresholds) and Subgroups; Results, Clinical Utility and Subgroups |
| Model output | 15 | D | Specify the output of the prediction model (e.g., probabilities, classification). Provide details and rationale for any classification and how the thresholds were identified | Methods, Clinical Utility (probability output; thresholds and how they were chosen) |
| Training versus evaluation | 16 | D;E | Identify any differences between the development and evaluation data in healthcare setting, eligibility criteria, outcome, and predictors | Methods, Input Features (leukocyte esterase is not measured in KNHANES and is not an input; KNHANES dipstick coding fixed before the models were fitted or evaluated, alternative codings as sensitivity analyses) and External Validation Cohort; Results, Study Population and Setting; Table 1; Table S1 panels B-D; Tables S2 and S3 |
| Ethical approval | 17 | D;E | Name the institutional research board or ethics committee that approved the study and describe the participant-informed consent or the ethics committee waiver of informed consent | Methods, Ethical Considerations |
| **Open Science** |  |  |  |  |
| Funding | 18a | D;E | Give the source of funding and the role of the funders for the present study | Funding Statement |
| Conflicts of interest | 18b | D;E | Declare any conflicts of interest and financial disclosures for all authors | Conflicts of Interest |
| Protocol | 18c | D;E | Indicate where the study protocol can be accessed or state that a protocol was not prepared | Methods, Study Design (no protocol prepared) |
| Registration | 18d | D;E | Provide registration information for the study, including register name and registration number, or state that the study was not registered | Methods, Study Design (not registered) |
| Data sharing | 18e | D;E | Provide details of the availability of the study data | Data Availability |
| Code sharing | 18f | D;E | Provide details of the availability of the analytical code | Data Availability |
| **Patient & Public Involvement** |  |  |  |  |
| Patient & Public Involvement | 19 | D;E | Provide details of any patient and public involvement during the design, conduct, reporting, interpretation, or dissemination of the study or state no involvement | Methods, Study Design (no patient or public involvement) |
| **Results** |  |  |  |  |
| Participants | 20a | D;E | Describe the flow of participants through the study, including the number of participants with and without the outcome and, if applicable, a summary of the follow-up time. A diagram may be helpful | Results, Study Population and Setting; Figure 1; Table S1 panel A |
|  | 20b | D;E | Report the characteristics overall and, where applicable, for each data source or setting, including the key dates, key predictors (including demographics), treatments received, sample size, number of outcome events, follow-up time, and amount of missing data. A table may be helpful. Report any differences across key demographic groups | Results, Study Population and Setting; Table 1; Table S1 panels B-D; Table S5 panel A |
|  | 20c | E | For model evaluation, show a comparison with the development data of the distribution of important predictors (demographics, predictors, and outcome) | Table 1; Table S3 panel B |
| Model development | 21 | D;E | Specify the number of participants and outcome events in each analysis (e.g., for model development, hyperparameter tuning, model evaluation) | Methods, Training and Internal Test Sets (one split shared by the thresholds) and Added Value of the Dipstick; Tables 1-2 and 4 |
| Model specification | 22 | D | Provide details of the full prediction model (e.g., formula, code, object, application programming interface) to allow predictions in new individuals and to enable third-party evaluation and implementation, including any restrictions to access or re-use (e.g., freely available, proprietary) | Data Availability (model files and their access terms); Table S8 (hyperparameters) |
| Model performance | 23a | D;E | Report model performance estimates with confidence intervals, including for any key subgroups (e.g., sociodemographic). Consider plots to aid presentation | Results; Tables 2-5; Figures 2-5 |
|  | 23b | D;E | If examined, report results of any heterogeneity in model performance across clusters | Not applicable (single development site); survey years in Table 5 and Table S14 panel D |
| Model updating | 24 | E | Report the results from any model updating, including the updated model and subsequent performance | Results, Discrimination and Calibration (intercept update); Table 2; Clinical Utility (updated decision curves); Figure 4; Table S11 panel F, Table S12 panel E, and Table S13 panel B |
| **Discussion** |  |  |  |  |
| Interpretation | 25 | D;E | Give an overall interpretation of the main results, including issues of fairness in the context of the objectives and previous studies | Discussion, Principal Results and Comparison With Prior Work |
| Limitations | 26 | D;E | Discuss any limitations of the study (such as a non-representative sample, sample size, overfitting, missing data) and their effects on any biases, statistical uncertainty, and generalizability | Discussion, Limitations |
| Usability of the model in the context of current care | 27a | D | Describe how poor quality or unavailable input data (e.g., predictor values) should be assessed and handled when implementing the prediction model | Discussion, Implications for Implementation (BMI availability, dipstick coding, input monitoring) |
|  | 27b | D | Specify whether users will be required to interact in the handling of the input data or use of the model, and what level of expertise is required of users | Discussion, Implications for Implementation |
|  | 27c | D;E | Discuss any next steps for future research, with a specific view to applicability and generalizability of the model | Discussion, Implications for Implementation, Limitations and Conclusions |

## TRIPOD+AI for Abstracts

| Section | Item | Checklist item (abbreviated) | Where reported |
|---|---|---|---|
| Title | 1 | Identify the study as developing or evaluating the performance of a multivariable prediction model, the target population, and the outcome to be predicted | Title |
| Background | 2 | Brief explanation of the healthcare context and rationale | Abstract, Background |
| Objectives | 3 | Specify the study objectives, including whether the study describes the development or validation of a prediction model (or both) | Abstract, Objective |
| Methods: data | 4 | Describe the sources of data | Abstract, Methods |
| Methods: participants | 5 | Describe the eligibility criteria and the setting | Abstract, Methods |
| Methods: outcome | 6 | Specify the outcome predicted by the model | Abstract, Methods |
| Methods: model | 7 | Specify the type of model and a summary of the model-building steps | Abstract, Methods |
| Methods: model evaluation | 8 | Specify the measures used to assess model performance | Abstract, Methods |
| Results: participants | 9 | Report the number of participants and outcome events | Abstract, Methods |
| Results: predictors | 10 | Summarise the predictors in the final model | Abstract, Objective and Methods |
| Results: model performance | 11 | Report model performance estimates with confidence intervals | Abstract, Results |
| Discussion: interpretation | 12 | Give an overall interpretation of the main results | Abstract, Conclusions |
| Registration | 13 | Registration number and name of registry, or state not registered | Not registered (Methods, Study Design); not stated in the abstract |

TRIPOD+AI: Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis with Artificial Intelligence.
