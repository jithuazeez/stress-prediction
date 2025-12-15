## Exploring the Optimal Time Window for Predicting

## Cognitive Load Using Physiological Sensor Data

### Minghao Cai, Carrie Demmans Epp

```
EdTeKLA Research Group, Department of Computing Science, University of Alberta, Edmonton, Canada
```
```
Abstract
Learning analytics has begun to use physiological signals because these have been linked with learners’
cognitive and affective states. These signals, when interpreted through machine learning techniques,
offer a nuanced understanding of the temporal dynamics of student learning experiences and processes.
However, there is a lack of clear guidance on the optimal time window to use for analyzing physiological
signals within predictive models. We conducted an empirical investigation of different time windows
(ranging from 60 to 210 seconds) when analysing multichannel physiological sensor data for predicting
cognitive load. Our results demonstrate a preference for longer time windows, with optimal window
length typically exceeding 90 seconds. These findings challenge the conventional focus on immediate
physiological responses, suggesting that a broader temporal scope could provide a more comprehensive
understanding of cognitive processes. In addition, the variation in which time windows best supported
prediction across classifiers underscores the complexity of integrating physiological measures. Our
findings provide new insights for developing educational technologies that more accurately reflect and
respond to the dynamic nature of learner cognitive load in complex learning environments.
Keywords
Cognitive Load, Physiological Signals, Learning Analytics, User Modelling
```
## 1. Introduction

```
Physiological signals have long been used within learning analytics research because these
signals are associated with learners’ underlying cognitive and affective states. Recent advances
in machine learning and sensors have enabled researchers to adopt a more nuanced approach
that accounts for temporality. This more nuanced approach supports improved understanding of
underlying, latent constructs that are important to learning. These constructs include cognitive
load and its impact on performance [ 1 , 2 ]. However, there is a lack of established evidence
for the choice of time window when using physiological signals in predictive modeling. In
previous research, there has been a focus on immediate physiological responses to discrete
tasks or events, often employing a narrow time window (e.g., 2s) to isolate specific responses
directly attributable to such activities [ 3 ]. While this provides insight into a learner’s immediate
cognitive reactions, this approach may not fully reflect the complexities of learning processes
that unfold over extended periods, especially in more interactive and dynamic settings [4].
We investigated how varying the amount of sensor signal (time window length) affects
model performance when using multichannel physiological sensors to predict cognitive load
```
```
PhysioCHI: Towards Best Practices for Integrating Physiological Signals in HCI, May 11, 2024, Honolulu, HI, USA
$minghaocai@ualberta.ca (M. Cai); cdemmansepp@ualberta.ca (C. Demmans Epp)
0000-0002-3331-5979 (M. Cai); 0000-0001-9079-4921 (C. Demmans Epp)
©2024 Copyright for this paper by its authors. Use permitted under Creative Commons License Attribution 4.0 International (CC BY 4.0).
```
# arXiv:2406.13793v1 [cs.HC] 19 Jun 2024


in complex learning environments. We conducted a empirical study assessing the impact of
different window lengths when performing prediction with different classifiers. By extending
the observation period, we aim to capture immediate physiological reactions as well as the
gradual changes and patterns that emerge over time because these changes characterize learning
processes [5].

## 2. Related Work

Cognitive Load Theory (CLT) is grounded in research about working memory and mental effort
[ 6 ]. CLT provides a framework for understanding the link between cognitive demands and
learning performance [ 7 , 8 , 9 ]. It consists of three types of load: intrinsic, extraneous, and
germane. Intrinsic load relates to the difficulty of the subject matter and varies with the learner’s
expertise. Extraneous load hinders learning and is a consequence of system or activity designs
that cause unnecessary processing. Germane load supports learning via schema development
and is under scrutiny because it is difficult to distinguish from the other subtypes of cognitive
load.
Cognitive load measurement has primarily used self-reports and task-based assessments.
However, self-reports miss unconscious cognitive processes and are prone to recall bias, affecting
the accuracy of reported cognitive effort [ 10 , 11 ]. Task-based methods are also limited. They
evaluate cognitive load through reaction times or learner accuracy on secondary tasks that
learners are asked to perform while completing the primary activity. These task-based methods
may impact performance on the primary task [ 12 ] even though they are intended to use few
cognitive resources.
Recently, the use of physiological methods, particularly pupil diameter, has been gaining
attention. Pupil dilation is associated with increased cognitive challenges [ 13 ] and can be
measured via eye-tracking technologies. This method involves comparing pupil size to a
baseline, which can be affected by factors like lighting and camera angle [ 14 ]. To overcome
these issues, researchers developed the Index of Cognitive Activity (ICA) [ 15 ] and the Index of
Pupillary Activity (IPA) [ 16 ], focusing on real-time pupil fluctuations to provide more accurate
assessments of cognitive load. These innovations offer improved reliability over traditional
baseline comparisons by accounting for changes that may occur.
Despite advances in measuring cognitive load through physiology, there remains a gap in the
literature regarding the time window that should be employed when using physiological signals
in predictive modeling. Prior studies have predominantly concentrated on capturing short-term
physiological responses to distinct tasks or events, often employing narrow time windows.
This approach helps link the measured responses to the cognitive tasks being examined. For
example, Chen and Epps used a 12-second window for measuring the average size of the pupil to
estimate cognitive load [ 17 ]. In contrast, some have adopted a coarse, time-aggregated approach
that consolidates the average pupil size from prolonged tasks into a single metric, typically
overlooking nuances such as the timing, magnitude, or form of specific responses [ 18 ]. This
approach introduces questions regarding the precision and efficacy of these measurements, as
extended observation periods might blend specific cognitive load indicators with unrelated data.
Not surprisingly, we have yet to form a consensus on how to best select or adjust the time


window to ensure models accurately reflect cognitive load. The selection of the time window
is critical, as it can influence the reliability and validity of model inferences. A time window
that is too brief may not fully reflect the extent of the cognitive load experienced during
learning, whereas one that is overly extended could include irrelevant information, thereby
complicating the task of pinpointing the exact cognitive demands placed on individuals. The
absence of established guidelines for which window size to use necessitates further research.
So, we explored the tuning of the time window for physiological signals and tested their
differential predictive performance. In this paper, three channels of sensor data (pupil diameter,
electrodermal activity [EDA], and heart rate) were used to predict cognitive load.

## 3. Methods

### 3.1. Dataset

The dataset used in this study was collected from 35 English learners who interacted with an
online literacy game for one hour [ 19 ]. Due to significant sensor noise, data from one participant
was excluded, resulting in a final sample of 34 individuals aged 17 to 33 years (𝑀= 24. 1 years).
None of the participants had prior experience with the game.
The dataset had two parts: physiological sensor data and self-reported cognitive load.
Physiological data were collected using a two-sensor system: an open-source eye tracker
(Pupil Core, Pupil Labs) and a wireless wristband (E4, Empatica Inc.). The eye tracker, which
has the form factor of standard glasses, captured eye movements and pupil dynamics at 200 Hz,
alongside a scene camera recording the user’s viewpoint. The wristband, worn like a watch,
collected EDA data at 4 Hz. The wristband also collected heart rate from blood volume pulse
(BVP) readings. Both BVP and EDA were measured in 10-second intervals. This setup allowed
for the synchronous recording of learners’ physiological responses during gameplay.
Self-reported cognitive load was measured approximately every 5 minutes. Scores ranged
from 1 to 10 (𝑀= 5. 0 ,𝑆𝐷= 1. 83 ) and were categorized into three levels for classification:
low (1 to 3.33), moderate (3.34 to 6.66), and high (6.67 to 10). These levels were used as labels
for the predictive model.

### 3.2. Pre-processing of Physiological Data

Pupil diameter signals for both eyes along with blink activity records were exported. This data
included time-stamped pupil diameters and an associated confidence level, ranging from 0.0 (no
detection) to 1.0 (high certainty). To prepare the data, we removed segments corresponding to
blinks and instances of low confidence (below 0.65). For data continuity, missing data due to
blink and low-confidence removal were linearly interpolated. To further reduce noise, we applied
a third-order Butterworth filter with a 4 Hz cutoff to minimize high-frequency disturbances
while retaining signal fidelity [3].
The EDA and heart rate signals were smoothed using a Simple Moving Average (SMA) filter.
This technique helps to minimize high-frequency noise while preserving the characteristics of
the signals that are relevant for assessing arousal [20].


Post-cleaning, we segmented the physiological data (pupil diameter, EDA, heart rate), aligning
it with the timestamps from participants’ self-reported cognitive load data. We segmented the
sensor data around these timestamps to reflect distinct periods of system interaction.
Data recorded during self-report completion was excluded to avoid confounding effects from
cognitive processes related to data collection. This process resulted in 312 segments of data.

### 3.3. Experimental Procedures

We experimented with different time windows for extracting physiological features. Features
were extracted for each data segment. We tested the predictive performance of physiological
data from 6 windows: 60s, 90s, 120s, 150s, 180s, and 210s.
For each time window, we extracted 7 features from the pupil diameter data: the frequency of
changes in pupil diameter (PCF); the minimum pupil size in the dataset (MinPD); the average size
of the pupil diameter (AvgPD); the maximum pupil size in the dataset (MaxPD); the average speed
at which the pupil diameter changes (AvgPV); the maximum speed at which the pupil diameter
changes (MaxPV); and the largest continuous change in pupil diameter without direction change
(MaxPC).
For each time window, we extracted 9 features from the EDA data: the frequency of changes
in EDA (ECF); the maximum continuous change in EDA without direction change (MaxEC); the
minimum value of the EDA (MinE); the maximum value of the EDA (MaxE); the average value
of the EDA (AvgE); the standard deviation of the EDA (SDGE); the average speed of change in
EDA levels over time (AvgEV); the maximum speed at which the EDA changes (MaxEV); the
difference between the maximum and minimum EDA (RngE).
For each time window, we extracted 9 features from the heart rate data: the frequency of
changes in heart rate (HCF); the minimum value of the heart rate (MinH); the maximum value
of the heart rate (MaxH); the average value of the heart rate (AvgH); the standard deviation of
the heart rate (SDH); the average speed of change in heart rate (AvgHV); the maximum speed
at which the heart rate changes (MaxHV); the difference between the maximum and minimum
heart rate (RngH); the maximum continuous change in heart rate without direction change
(MaxHC).
We tested the predictive capabilities of each window with six machine learning classifiers:
Naive Bayes (NB), Decision Tree (DT), Linear Support Vector Machine (Linear SVM), Radial
Basis Function Support Vector Machine (RBF SVM), Logistic Regression (LR), and Random
Forest (RF).
The classification was subject-independent. We allocated 80% of data for training and 20% for
testing. During training, we performed hyperparameter tuning with 4-fold cross-validation.
Grid search was used to find the optimal settings, with tuning focusing on maximizing Cohen’s
𝜅[21].

## 4. Results and Discussion

Table 1 reports the time window where the model performed best. Most models performed best
when the window was longer than 90s and the best performing model (Random Forest) used a
window that exceeded 3 minutes.


**Table 1**
Model Performance and window size
**Classifier** 𝜅 **Accuracy Window (s)**
Naive Bayes .107 .417 90
Decision Tree .211 .600 150
Linear SVM .218 .533 120
RBF SVM .279 .617 150
Logistic Regression .276 .534 210
**Random Forest .434 .700 210**
Bold font indicates the best result

**Figure 1:** Kappa values for each model by window size.

When we inspected model performance across windows, it became clear that window size
impacted classifier performance (Figure 1). Our results indicated that peak performance for
all classifiers was achieved when 90 seconds or more of sensor data were used as input to the
model. Most classifiers, apart from Naive Bayes, demonstrated improved performance with
wider windows.
Our findings suggest that when it comes to predicting cognitive load in interactive game-based
learning settings, classifiers tested with a wider window (150 or 210) generally performed better.
This outcome challenges previous research that prioritized immediate physiological reactions
using shorter time-frames. Learning sessions, especially in complex environments like games,
can last longer. Our findings suggest that a larger time window may allow comprehensive
tracking of physiological changes, potentially reflecting the gradual increase and decrease of
cognitive load as learners navigate through different interactive events. Such settings might


include the impact of delayed physiological responses or the accumulation of mental fatigue
that shorter windows may not capture. This shift in methodology not only enriches our
understanding of cognitive processes in complex learning environments but also enhances
the potential for developing more effective educational technologies that are sensitive to the
fluctuating dynamics of learner engagement and cognitive load.
Furthermore, the variation in optimal time windows among different classifiers highlights the
complexity of integrating physiological measures for cognitive load assessment. This variability
suggests that the best analysis window differs based on several factors, including the educational
setting, task complexity, types of physiological data, and the specific algorithms used. Therefore,
a universal approach to time window selection for all scenarios seems impractical. Instead,
customizing the time window based on the algorithm, context, and learning environment could
yield more accurate and meaningful insights into cognitive load dynamics during educational
activities.

## 5. Conclusion

Learner physiology is an important resource for understanding changes in internal states, such
as cognitive load. Yet, there exists a lack of evidence that can inform time window selection
when using physiological signals in predictive modeling, particularly within complex educa-
tional environments. In this study, we conducted an experiment to explore model performance
when using different window sizes to predict cognitive load from physiological data. Our study
provides insight into how to employ physiological signals in educational technology research
by demonstrating that a wider time window is beneficial for modelling some constructs, i.e.,
cognitive load. We posit this benefit is due to the extended sensor data’s ability to capture cog-
nitive adaptation. Most importantly, the presented analyses highlight the potential interactions
between algorithms and the amount of physiological sensor data used. By understanding how
to select an appropriate time window for physiological data, researchers and practitioners can
optimize the data collection process thereby enhancing both the precision and reliability of the
systems used to monitor and respond to student cognitive states. It opens up possibilities for
creating more personalized learning experiences that adjust content difficulty and presentation
using real-time assessments of cognitive load.

## Acknowledgments

This work was supported in part by funding from the Social Sciences and Humanities Research
Council of Canada and the Natural Sciences and Engineering Research Council of Canada
(NSERC), [RGPIN-2018-03834].

## References

```
[1] S. D’Mello, E. Dieterle, A. Duckworth, Advanced, Analytic, Automated (AAA) Mea-
surement of Engagement During Learning, Educational psychologist 52 (2017) 104–123.
doi:10.1080/00461520.2017.1281747.
```

[2] C. Baker, The Impact of Instructor Immediacy and Presence for Online Student Affective
Learning, Cognition, and Motivation, Journal of Educators Online 7 (2010).
[3] K. Krejtz, A. T. Duchowski, A. Niedzielska, C. Biele, I. Krejtz, Eye tracking cognitive load
using pupil diameter and microsaccades with fixed gaze, PLOS ONE 13 (2018) e0203629.
doi:10.1371/journal.pone.0203629, publisher: Public Library of Science.
[4] J. Sweller, The Development of Cognitive Load Theory: Replication Crises and Incorpora-
tion of Other Theories Can Lead to Theory Expansion, Educational Psychology Review 35
(2023) 95. doi:10.1007/s10648-023-09817-2.
[5] B. Xie, G. Salvendy, Review and reappraisal of modelling and predicting mental workload
in single- and multi-task environments, Work & Stress 14 (2000) 74–99.
[6] G. Salomon, Television is "easy" and print is "tough": The differential investment of mental
effort in learning as a function of perceptions and attributions, Journal of Educational
Psychology 76 (1984) 647–658. doi:10.1037/0022-0663.76.4.647.
[7] J. Sweller, Cognitive Load Theory, in: J. P. Mestre, B. H. Ross (Eds.), Psychology of Learning
and Motivation, volume 55, Academic Press, 2011, pp. 37–76.
[8] F. Paas, J. E. Tuovinen, H. Tabbers, P. W. M. Van Gerven, Cognitive Load Measurement
as a Means to Advance Cognitive Load Theory, Educational Psychologist 38 (2003) 63–
71.doi:10.1207/S15326985EP3801_8.
[9] F. Paas, P. Ayres, Cognitive Load Theory: A Broader View on the Role of Memory in
Learning and Education, Educational Psychology Review 26 (2014) 191–195. doi:10.1007/
s10648-014-9263-5.
[10] M. Krell, K. M. Xu, G. D. Rey, F. Paas, Editorial: Recent Approaches for Assessing Cognitive
Load From a Validity Perspective, Frontiers in Education 6 (2022).
[11] J. Leppink, F. Paas, C. P. M. Van der Vleuten, T. Van Gog, J. J. G. Van Merriënboer, Develop-
ment of an instrument for measuring different types of cognitive load, Behavior Research
Methods 45 (2013) 1058–1072. doi:10.3758/s13428-013-0334-1.
[12] R. Brunken, J. L. Plass, D. Leutner, Direct Measurement of Cognitive Load in Multimedia
Learning, Educational Psychologist 38 (2003) 53–61. doi:10.1207/S15326985EP3801_7,
publisher: Routledge _eprint: https://doi.org/10.1207/S15326985EP3801_7.
[13] J. Beatty, Task-evoked pupillary responses, processing load, and the structure of processing
resources, Psychological Bulletin 91 (1982) 276–292. doi:10.1037/0033-2909.91.2.
276 , place: US Publisher: American Psychological Association.
[14] J. Beatty, B. Lucero-Wagoner, The pupillary system, in: Handbook of psychophysiology,
2nd ed, Cambridge University Press, New York, NY, US, 2000, pp. 142–162.
[15] S. Marshall, The Index of Cognitive Activity: measuring cognitive workload, in: Pro-
ceedings of the IEEE 7th Conference on Human Factors and Power Plants, 2002, pp. 7–7.
doi:10.1109/HFPP.2002.1042860.
[16] A. T. Duchowski, K. Krejtz, I. Krejtz, C. Biele, A. Niedzielska, P. Kiefer, M. Raubal, I. Gi-
annopoulos, The Index of Pupillary Activity: Measuring Cognitive Load vis-à-vis Task
Difficulty with Pupil Oscillation, in: Proceedings of the 2018 CHI Conference on Human
Factors in Computing Systems, CHI ’18, Association for Computing Machinery, New York,
NY, USA, 2018, pp. 1–13.
[17] S. Chen, J. Epps, Using Task-Induced Pupil Diameter and Blink Rate to
Infer Cognitive Load, Human–Computer Interaction 29 (2014) 390–413.


doi:10.1080/07370024.2014.892428, publisher: Taylor & Francis _eprint:
https://doi.org/10.1080/07370024.2014.892428.
[18] J. Klingner, R. Kumar, P. Hanrahan, Measuring the task-evoked pupillary response with a
remote eye tracker, in: Proceedings of the 2008 symposium on Eye tracking research &
applications, ETRA ’08, Association for Computing Machinery, New York, NY, USA, 2008,
pp. 69–72. doi:10.1145/1344471.1344489.
[19] M. Cai, G. Rebolledo Mendez, G. Arevalo, S. S. Tang, Y. A Abdullah, and C. Demmans Epp.
Toward Supporting Adaptation: Exploring Affect’s Role in Cognitive Load when Using
a Literacy Game. In _Proceedings of the CHI Conference on Human Factors in Computing
Systems (CHI ’24)_ , Honolulu, HI, USA, May 2024. doi:10.1145/3613904.3642150.
Type = Article
[20] H. F. Posada-Quintero, K. H. Chon, Innovations in Electrodermal Activity Data Collec-
tion and Signal Processing: A Systematic Review, Sensors 20 (2020) 479.doi:10.3390/
s20020479, number: 2 Publisher: Multidisciplinary Digital Publishing Institute.
[21] J. Cohen, A Coefficient of Agreement for Nominal Scales, Educational and Psychological
Measurement 20 (1960) 37–46.doi:10.1177/001316446002000104, publisher: SAGE
Publications Inc.


