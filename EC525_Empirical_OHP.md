EC525_EmpiricalProj
================
Connor Burton
May, 2022

# OHP Project 1

## Preliminaries:

``` r
#get packages
library(pacman)
p_load(data.table, tidyverse, broom, sandwich, haven)
# load in data
df = read_dta('ohp.dta') %>% as.data.table()
tibble(df)
```

    ## # A tibble: 12,229 x 24
    ##    person_id household_id weight_total_inp  treatment age_inp bp_sar_inp chl_inp
    ##        <dbl>        <dbl>            <dbl>  <dbl+lbl>   <dbl>      <dbl>   <dbl>
    ##  1         5       100005            1.15  1 [Select~      60        144    241.
    ##  2         8       102094            0.897 0 [Not se~      41        134    230.
    ##  3        16       140688            1     0 [Not se~      39        126    230.
    ##  4        17       100017            1.21  0 [Not se~      52        168    235.
    ##  5        18       100018            1     0 [Not se~      51        119    178.
    ##  6        23       115253            1.00  1 [Select~      32         98    174.
    ##  7        24       169996            1.20  1 [Select~      34        108    153.
    ##  8        29       100029            1     0 [Not se~      23        125    253.
    ##  9        47       100047            1     0 [Not se~      43        100    199.
    ## 10        57       100057            1.00  1 [Select~      46        104    199.
    ## # ... with 12,219 more rows, and 17 more variables:
    ## #   dep_dx_post_lottery <dbl+lbl>, dep_dx_pre_lottery <dbl+lbl>,
    ## #   dia_dx_post_lottery <dbl+lbl>, dia_dx_pre_lottery <dbl+lbl>,
    ## #   doc_num_mod_inp <dbl>, edu_inp <dbl+lbl>, gender_inp <dbl+lbl>,
    ## #   hbp_dx_post_lottery <dbl+lbl>, hbp_dx_pre_lottery <dbl+lbl>,
    ## #   hispanic_inp <dbl+lbl>, itvw_english_inp <dbl>, numhh_list <dbl+lbl>,
    ## #   ohp_all_ever_survey <dbl+lbl>, race_black_inp <dbl+lbl>, ...

## 1.

The variable ‘ohp_all_ever_survey’ is a binary indicator for if the
observation was ever enrolled in Medicaid regardless of treatment
assignment in the OHP experiment. The ‘treatment’ variable is a binary
indicator for if the case was assigned treatment in the random lottery.
‘Treatment’ is the treatment variable because the experiment is not
testing necessarily the impact of Medicaid itself, rather the impact of
the random assignment to Medicaid through the lottery. The former
variable is a form of noncompliance and suffers from selection into
Medicaid rather than random assignment to it.

## 2.

``` r
df_cont = df[treatment==0, .(age_inp, dia_dx_pre_lottery, hbp_dx_pre_lottery, gender_inp)]

#iterating mean function over subsetted df with target variables and pivoting
cmean = lapply(df_cont, mean, na.rm = TRUE) %>% as.data.table()
cmean = cmean %>% pivot_longer(
age_inp:gender_inp,
names_to = 'variable',
values_to = 'control mean') %>%
as.data.table()
tibble(cmean)
```

    ## # A tibble: 4 x 2
    ##   variable           `control mean`
    ##   <chr>                       <dbl>
    ## 1 age_inp                   40.6   
    ## 2 dia_dx_pre_lottery         0.0717
    ## 3 hbp_dx_pre_lottery         0.183 
    ## 4 gender_inp                 0.569

## 3.

``` r
#regressing target variables on treatment to find difference and SE, grabbing relevant cells
reg_age = lm(age_inp~treatment,data=df) %>% tidy() %>% as.data.table()
reg_age = reg_age[2,2:3]
reg_dia = lm(dia_dx_pre_lottery ~ treatment, data = df) %>% tidy() %>% as.data.table()
reg_dia = reg_dia[2,2:3]
reg_hbp = lm(hbp_dx_pre_lottery ~ treatment, data = df) %>% tidy() %>% as.data.table()
reg_hbp = reg_hbp[2,2:3]
reg_gen = lm(gender_inp ~ treatment, data = df) %>% tidy() %>% as.data.table()
reg_gen = reg_gen[2,2:3]

#combining into single df, renaming, and binding to control means df
treated = rbind(reg_age, reg_dia, reg_hbp, reg_gen)
names(treated)[1:2] = c('difference', 'std error')
means_comp = cbind(cmean, treated)
means_comp
```

    ##              variable control mean   difference   std error
    ## 1:            age_inp  40.60606061  0.380317975 0.211772551
    ## 2: dia_dx_pre_lottery   0.07172201 -0.000796696 0.004659082
    ## 3: hbp_dx_pre_lottery   0.18264293 -0.001337153 0.006984929
    ## 4:         gender_inp   0.56881205 -0.006106555 0.008977078

## 4.

The balance table above is consistent with random assignment to
treatment, as shown by the standard error nullifying the statistical
significance in all variables at the 95% level. These standard errors
were determined through separate regressions of health statuses prior to
treatment on treatment assignment.

## 5.

``` r
#estimating compliance rate by regressing random assignment on selection into OHP
df_cont2 = df[treatment==0, .(age_inp, dia_dx_pre_lottery, hbp_dx_pre_lottery, gender_inp)]
df_comp = df[treatment==1]
reg_comp = lm(ohp_all_ever_survey ~ treatment, data = df) %>% tidy() %>% as.data.table()
reg_comp[,1:3]
```

    ##           term  estimate   std.error
    ## 1: (Intercept) 0.1583362 0.005706118
    ## 2:   treatment 0.2535943 0.007895648

The compliance rate is the impact that random assignment had on
treatment assignment. In this regression, the baseline enrollment in
Medicaid among the control group is about 15.8%, while the impact that
random assignment had on Medicaid enrollment is 25.4%, meaning the
compliance rate = 0.254, while the total level of Medicaid enrollment
among the treated is 0.254 + 0.158 = 0.412.

## 6.

``` r
# outcomes:
## diabetes binary diagnosis post study
itt_dia = lm(dia_dx_post_lottery ~ treatment, data = df) %>% tidy() %>% as.data.table()
itt_dia = itt_dia[2, 2:3]
## hypertension binary diagnosis post study
itt_hbp = lm(hbp_dx_post_lottery ~ treatment, data = df) %>% tidy() %>% as.data.table()
itt_hbp = itt_hbp[2, 2:3]
## blood pressure (lower better) (inperson measurement)
itt_bp = lm(bp_sar_inp ~ treatment, data = df) %>% tidy() %>% as.data.table()
itt_bp = itt_bp[2, 2:3]
## cholesterol (lower better) (inperson measurement)
itt_chl = lm(chl_inp ~ treatment, data = df) %>% tidy() %>% as.data.table()
itt_chl = itt_chl[2, 2:3]
itt = rbind(itt_dia, itt_hbp, itt_bp, itt_chl)
itt_names = c('diabetes', 'hypertension', 'blood pressure', 'cholesterol')
itt = cbind(itt_names, itt)
tibble(itt)
```

    ## # A tibble: 4 x 3
    ##   itt_names      estimate std.error
    ##   <chr>             <dbl>     <dbl>
    ## 1 diabetes        0.00861   0.00231
    ## 2 hypertension    0.00241   0.00429
    ## 3 blood pressure -0.0583    0.300  
    ## 4 cholesterol    -0.642     0.613

The ITT is the raw difference between treatment and control groups. I
ran four regressions of diagnoses of diabetes and hypertension
post-treatment and of blood pressure and cholesterol on treatment
assignment and list them in a table with standard errors above. The
latter two health outcomes are not statistically significant though they
are the expected sign. The hypertension regression shows an increase in
diagnoses post-treatment but is not statistically significant, while
diabetes diagnoses show a statistically significant increase of 0.86%.
However, the statistical significance of these estimates should not be
shown as causal because this is merely the intent to treat effect, not
the average treatment effect on the treated because it has not been
transformed by the compliance rate within the experiment. Similarly,
there is the question of if the positive coefficient on the two
diagnoses variables are merely the impact of having better health
insurance that allowed patients to be diagnosed with issues that are
similarly prevalent between treatment and control groups. Restated:
perhaps increased diagnoses are the impact of better care and this
result should be carefully interpreted.

## 7.

``` r
#ATET is the intent to treat effect divided by the compliance rate in the experiment
itt$atet = itt$estimate/0.2535943
tibble(itt)
```

    ## # A tibble: 4 x 4
    ##   itt_names      estimate std.error     atet
    ##   <chr>             <dbl>     <dbl>    <dbl>
    ## 1 diabetes        0.00861   0.00231  0.0339 
    ## 2 hypertension    0.00241   0.00429  0.00948
    ## 3 blood pressure -0.0583    0.300   -0.230  
    ## 4 cholesterol    -0.642     0.613   -2.53

## 8.

There should be concern about attrition bias in this study. People who
qualified for the lottery and accepted the Medicaid may have passed
away, moved to a pay bracket that disqualified them from Medicaid, or
moved to a different state, which would create systematic treatment
attrition. This attrition would bias results through numerous
mechanisms. For instance, if a patient died or moved away they may not
have been available for the in-person surveys which would likely appear
as NA in the data and bias estimates in the ITT and ATET through their
exclusion in the regressions. Similarly, those who get bracketed out of
Medicaid are no longer under the impact of treatment assignment (perhaps
they have better or worse insurance through a new employer), thus
showing that their health outcomes are not the impact of treatment. The
sign of this bias cannot truly be estimated because of the myriad ways
attrition might have occurred.

## 9.

Experimental conditions were created in Oregon in 2008 that allowed
researchers to study the impact of random assignment into Medicaid
coverage through a random lottery. I have provided details above on the
data collected in this experiment that support its viability as a
randomized control trial. Using data collected at the onset and 25
months after random assignment to Medicaid, I show that balance tests
affirming that treatment and control groups are statistically comparable
and that the compliance rate in the experiment is equal to \~25.4%,
providing a large enough sample to perform regression analysis. Intent
to Treat effects are weak, however Average Treatment effects on the
Treated indicate statistically significant impacts of Medicaid
assignment on health outcomes. There is concern in this study about
attrition bias through patient mortality and pay-bracket Medicaid
exclusion, however these concerns are assuaged by balance tests and can
be rectified through further research into this question.

Citations: “Medicaid Increases Emergency-Department Use: Evidence from
Oregon’s Health Insurance Experiment.” (2014). Retrieved 28 April 2022,
from <https://www.science.org/doi/full/10.1126/science.1246183>
