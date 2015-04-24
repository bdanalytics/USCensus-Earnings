# UCI ML Repo: US Census Earnings: 1994: over50k classification
bdanalytics  

**  **    
**Date: (Fri) Apr 24, 2015**    

# Introduction:  

Data: 
Source: 
    Training:   https://courses.edx.org/c4x/MITx/15.071x_2/asset/census.csv  
    New:        <newdt_url>  
Time period: 



# Synopsis:

Based on analysis utilizing <> techniques, <conclusion heading>:  

### ![](<filename>.png)

## Potential next steps include:
- Organization:
    - Categorize by chunk
    - Priority criteria:
        0. Ease of change
        1. Impacts report
        2. Cleans innards
        3. Bug report
        
- manage.missing.data chunk:
    - cleaner way to manage re-splitting of training vs. new entity

- fit.models chunk:
    - Prediction accuracy scatter graph:
    -   Add tiles (raw vs. PCA)
    -   Use shiny for drop-down of "important" features
    -   Use plot.ly for interactive plots ?
    
    - Change .fit suffix of model metrics to .mdl if it's data independent (e.g. AIC, Adj.R.Squared - is it truly data independent ?, etc.)
    - move model_type parameter to myfit_mdl before indep_vars_vctr (keep all model_* together)
    - create a custom model for rpart that has minbucket as a tuning parameter
    - varImp for randomForest crashes in caret version:6.0.41 -> submit bug report

- Probability handling for multinomials vs. desired binomial outcome
-   ROCR currently supports only evaluation of binary classification tasks (version 1.0.7)
-   extensions toward multiclass classification are scheduled for the next release

- Skip trControl.method="cv" for dummy classifier ?
- Add custom model to caret for a dummy (baseline) classifier (binomial & multinomial) that generates proba/outcomes which mimics the freq distribution of glb_rsp_var values; Right now glb_dmy_glm_mdl always generates most frequent outcome in training data
- glm_dmy_mdl should use the same method as glm_sel_mdl until custom dummy classifer is implemented

- Compare glb_sel_mdl vs. glb_fin_mdl:
    - varImp
    - Prediction differences (shd be minimal ?)

- Move glb_analytics_diag_plots to mydsutils.R: (+) Easier to debug (-) Too many glb vars used
- Add print(ggplot.petrinet(glb_analytics_pn) + coord_flip()) at the end of every major chunk
- Parameterize glb_analytics_pn
- Move glb_impute_missing_data to mydsutils.R: (-) Too many glb vars used; glb_<>_df reassigned
- Replicate myfit_mdl_classification features in myfit_mdl_regression
- Do non-glm methods handle interaction terms ?
- f-score computation for classifiers should be summation across outcomes (not just the desired one ?)
- Add accuracy computation to glb_dmy_mdl in predict.data.new chunk
- Why does splitting fit.data.training.all chunk into separate chunks add an overhead of ~30 secs ? It's not rbind b/c other chunks have lower elapsed time. Is it the number of plots ?
- Incorporate code chunks in print_sessionInfo
- Test against 
    - projects in github.com/bdanalytics
    - lectures in jhu-datascience track

# Analysis: 

```r
rm(list=ls())
set.seed(12345)
options(stringsAsFactors=FALSE)
source("~/Dropbox/datascience/R/mydsutils.R")
source("~/Dropbox/datascience/R/myplot.R")
source("~/Dropbox/datascience/R/mypetrinet.R")
# Gather all package requirements here
#suppressPackageStartupMessages(require())
#packageVersion("snow")

#require(sos); findFn("pinv", maxPages=2, sortby="MaxScore")

# Analysis control global variables
glb_trnng_url <- "https://courses.edx.org/c4x/MITx/15.071x_2/asset/census.csv"
glb_newdt_url <- "<newdt_url>"
glb_is_separate_newent_dataset <- FALSE    # or TRUE
glb_split_entity_newent_datasets <- TRUE   # or FALSE
glb_split_newdata_method <- "sample"          # "condition" or "sample"
glb_split_newdata_condition <- "<col_name> <condition_operator> <value>"    # or NULL
glb_split_newdata_size_ratio <- 0.4               # > 0 & < 1
glb_split_sample.seed <- 2000               # or any integer
glb_max_obs <- 1000 # or any integer

glb_is_regression <- FALSE; glb_is_classification <- TRUE; glb_is_binomial <- TRUE

glb_rsp_var_raw <- "over50k"

# for classification, the response variable has to be a factor
glb_rsp_var <- "over50k.fctr"

# if the response factor is based on numbers e.g (0/1 vs. "A"/"B"), 
#   caret predict(..., type="prob") crashes
glb_map_rsp_raw_to_var <- function(raw) {
    relevel(factor(ifelse(raw == " >50K", "Y", "N")), as.factor(c("Y", "N")), ref="N")
    #as.factor(paste0("B", raw))
    #as.factor(raw)    
}
glb_map_rsp_raw_to_var(c(" >50K", " <=50K", " >50K", " <=50K", " <=50K"))
```

```
## [1] Y N Y N N
## Levels: N Y
```

```r
glb_map_rsp_var_to_raw <- function(var) {
    #as.numeric(var) - 1
    #as.numeric(var)
    c(" <=50K", " >50K")[as.numeric(var)]
}
glb_map_rsp_var_to_raw(glb_map_rsp_raw_to_var(c(" >50K", " <=50K", " >50K", " <=50K", " <=50K")))
```

```
## [1] " >50K"  " <=50K" " >50K"  " <=50K" " <=50K"
```

```r
if ((glb_rsp_var != glb_rsp_var_raw) & is.null(glb_map_rsp_raw_to_var))
    stop("glb_map_rsp_raw_to_var function expected")

glb_rsp_var_out <- paste0(glb_rsp_var, ".predict.") # model_id is appended later
glb_id_vars <- NULL # or c("<id_var>")

# List transformed vars  
glb_exclude_vars_as_features <- c(NULL) # or c("<var_name>")    
# List feats that shd be excluded due to known causation by prediction variable
if (glb_rsp_var_raw != glb_rsp_var)
    glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
                                            glb_rsp_var_raw)
glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
                                      c(NULL)) # or c("<col_name>")
# List output vars (useful during testing in console)          
# glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
#                         grep(glb_rsp_var_out, names(glb_trnent_df), value=TRUE)) 

glb_impute_na_data <- FALSE            # or TRUE
glb_mice_complete.seed <- 144               # or any integer

# rpart:  .rnorm messes with the models badly
#         caret creates dummy vars for factor feats which messes up the tuning
#             - better to feed as.numeric(<feat>.fctr) to caret 
# Regression
if (glb_is_regression)
    glb_models_method_vctr <- c("lm", "glm", "rpart", "rf") else
# Classification
    if (glb_is_binomial)
        glb_models_method_vctr <- c("glm", "rpart", "rf") else  
        glb_models_method_vctr <- c("rpart", "rf")

glb_models_lst <- list(); glb_models_df <- data.frame()
# Baseline prediction model feature(s)
glb_Baseline_mdl_var <- NULL # or c("<col_name>")

glb_model_metric_terms <- NULL # or matrix(c(
#                               0,1,2,3,4,
#                               2,0,1,2,3,
#                               4,2,0,1,2,
#                               6,4,2,0,1,
#                               8,6,4,2,0
#                           ), byrow=TRUE, nrow=5)
glb_model_metric <- NULL # or "<metric_name>"
glb_model_metric_maximize <- NULL # or FALSE (TRUE is not the default for both classification & regression) 
glb_model_metric_smmry <- NULL # or function(data, lev=NULL, model=NULL) {
#     confusion_mtrx <- t(as.matrix(confusionMatrix(data$pred, data$obs)))
#     #print(confusion_mtrx)
#     #print(confusion_mtrx * glb_model_metric_terms)
#     metric <- sum(confusion_mtrx * glb_model_metric_terms) / nrow(data)
#     names(metric) <- glb_model_metric
#     return(metric)
# }

glb_tune_models_df <- 
   rbind(
    #data.frame(parameter="cp", min=0.00005, max=0.00005, by=0.000005),
                            #seq(from=0.01,  to=0.01, by=0.01)
    #data.frame(parameter="mtry", min=2, max=4, by=1),
    data.frame(parameter="dummy", min=2, max=4, by=1)
        ) 
# or NULL
glb_n_cv_folds <- 3 # or NULL

glb_clf_proba_threshold <- NULL # 0.5

# Model selection criteria
if (glb_is_regression)
    glb_model_evl_criteria <- c("min.RMSE.OOB", "max.R.sq.OOB", "max.Adj.R.sq.fit")
if (glb_is_classification) {
    if (glb_is_binomial)
        glb_model_evl_criteria <- c("max.Accuracy.OOB", "max.Kappa.OOB", "min.aic.fit") else
        glb_model_evl_criteria <- c("max.Accuracy.OOB", "max.Kappa.OOB")
}

glb_sel_mdl_id <- NULL # or "<model_id_prefix>.<model_method>"
glb_fin_mdl_id <- glb_sel_mdl_id # or "Final"

# Depict process
glb_analytics_pn <- petrinet(name="glb_analytics_pn",
                        trans_df=data.frame(id=1:6,
    name=c("data.training.all","data.new",
           "model.selected","model.final",
           "data.training.all.prediction","data.new.prediction"),
    x=c(   -5,-5,-15,-25,-25,-35),
    y=c(   -5, 5,  0,  0, -5,  5)
                        ),
                        places_df=data.frame(id=1:4,
    name=c("bgn","fit.data.training.all","predict.data.new","end"),
    x=c(   -0,   -20,                    -30,               -40),
    y=c(    0,     0,                      0,                 0),
    M0=c(   3,     0,                      0,                 0)
                        ),
                        arcs_df=data.frame(
    begin=c("bgn","bgn","bgn",        
            "data.training.all","model.selected","fit.data.training.all",
            "fit.data.training.all","model.final",    
            "data.new","predict.data.new",
            "data.training.all.prediction","data.new.prediction"),
    end  =c("data.training.all","data.new","model.selected",
            "fit.data.training.all","fit.data.training.all","model.final",
            "data.training.all.prediction","predict.data.new",
            "predict.data.new","data.new.prediction",
            "end","end")
                        ))
#print(ggplot.petrinet(glb_analytics_pn))
print(ggplot.petrinet(glb_analytics_pn) + coord_flip())
```

```
## Loading required package: grid
```

![](USCensus_Earnings_files/figure-html/set_global_options-1.png) 

```r
glb_analytics_avl_objs <- NULL

glb_script_tm <- proc.time()
glb_script_df <- data.frame(chunk_label="import_data", 
                            chunk_step_major=1, chunk_step_minor=0,
                            elapsed=(proc.time() - glb_script_tm)["elapsed"])
print(tail(glb_script_df, 2))
```

```
##         chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed import_data                1                0   0.002
```

## Step `1`: import data

```r
glb_entity_df <- myimport_data(url=glb_trnng_url, 
    comment=ifelse(!glb_is_separate_newent_dataset, "glb_entity_df", "glb_trnent_df"), 
                                force_header=TRUE)
```

```
## [1] "Reading file ./data/census.csv..."
## [1] "dimensions of data in ./data/census.csv: 31,978 rows x 13 cols"
##   age         workclass  education       maritalstatus         occupation
## 1  39         State-gov  Bachelors       Never-married       Adm-clerical
## 2  50  Self-emp-not-inc  Bachelors  Married-civ-spouse    Exec-managerial
## 3  38           Private    HS-grad            Divorced  Handlers-cleaners
## 4  53           Private       11th  Married-civ-spouse  Handlers-cleaners
## 5  28           Private  Bachelors  Married-civ-spouse     Prof-specialty
## 6  37           Private    Masters  Married-civ-spouse    Exec-managerial
##     relationship   race     sex capitalgain capitalloss hoursperweek
## 1  Not-in-family  White    Male        2174           0           40
## 2        Husband  White    Male           0           0           13
## 3  Not-in-family  White    Male           0           0           40
## 4        Husband  Black    Male           0           0           40
## 5           Wife  Black  Female           0           0           40
## 6           Wife  White  Female           0           0           40
##    nativecountry over50k
## 1  United-States   <=50K
## 2  United-States   <=50K
## 3  United-States   <=50K
## 4  United-States   <=50K
## 5           Cuba   <=50K
## 6  United-States   <=50K
##       age         workclass     education       maritalstatus
## 1105   20           Private     Assoc-voc       Never-married
## 4872   34           Private       HS-grad       Never-married
## 10396  33           Private  Some-college            Divorced
## 16284  70  Self-emp-not-inc     Bachelors  Married-civ-spouse
## 23270  22           Private       HS-grad       Never-married
## 31647  19           Private          11th       Never-married
##             occupation    relationship   race     sex capitalgain
## 1105      Adm-clerical       Own-child  White  Female           0
## 4872   Exec-managerial  Other-relative  White    Male           0
## 10396     Adm-clerical   Not-in-family  White  Female           0
## 16284   Prof-specialty         Husband  White    Male       20051
## 23270  Exec-managerial       Own-child  White  Female           0
## 31647            Sales       Own-child  Black  Female           0
##       capitalloss hoursperweek  nativecountry over50k
## 1105            0           40  United-States   <=50K
## 4872            0           50  United-States   <=50K
## 10396           0           40  United-States   <=50K
## 16284           0           35  United-States    >50K
## 23270           0           40  United-States   <=50K
## 31647           0           35  United-States   <=50K
##       age     workclass     education       maritalstatus
## 31973  22       Private  Some-college       Never-married
## 31974  27       Private    Assoc-acdm  Married-civ-spouse
## 31975  40       Private       HS-grad  Married-civ-spouse
## 31976  58       Private       HS-grad             Widowed
## 31977  22       Private       HS-grad       Never-married
## 31978  52  Self-emp-inc       HS-grad  Married-civ-spouse
##               occupation   relationship   race     sex capitalgain
## 31973    Protective-serv  Not-in-family  White    Male           0
## 31974       Tech-support           Wife  White  Female           0
## 31975  Machine-op-inspct        Husband  White    Male           0
## 31976       Adm-clerical      Unmarried  White  Female           0
## 31977       Adm-clerical      Own-child  White    Male           0
## 31978    Exec-managerial           Wife  White  Female       15024
##       capitalloss hoursperweek  nativecountry over50k
## 31973           0           40  United-States   <=50K
## 31974           0           38  United-States   <=50K
## 31975           0           40  United-States    >50K
## 31976           0           40  United-States   <=50K
## 31977           0           20  United-States   <=50K
## 31978           0           40  United-States    >50K
## 'data.frame':	31978 obs. of  13 variables:
##  $ age          : int  39 50 38 53 28 37 49 52 31 42 ...
##  $ workclass    : chr  " State-gov" " Self-emp-not-inc" " Private" " Private" ...
##  $ education    : chr  " Bachelors" " Bachelors" " HS-grad" " 11th" ...
##  $ maritalstatus: chr  " Never-married" " Married-civ-spouse" " Divorced" " Married-civ-spouse" ...
##  $ occupation   : chr  " Adm-clerical" " Exec-managerial" " Handlers-cleaners" " Handlers-cleaners" ...
##  $ relationship : chr  " Not-in-family" " Husband" " Not-in-family" " Husband" ...
##  $ race         : chr  " White" " White" " White" " Black" ...
##  $ sex          : chr  " Male" " Male" " Male" " Male" ...
##  $ capitalgain  : int  2174 0 0 0 0 0 0 0 14084 5178 ...
##  $ capitalloss  : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ hoursperweek : int  40 13 40 40 40 40 16 45 50 40 ...
##  $ nativecountry: chr  " United-States" " United-States" " United-States" " United-States" ...
##  $ over50k      : chr  " <=50K" " <=50K" " <=50K" " <=50K" ...
##  - attr(*, "comment")= chr "glb_entity_df"
## NULL
```

```r
if (!glb_is_separate_newent_dataset) {
    glb_trnent_df <- glb_entity_df; comment(glb_trnent_df) <- "glb_trnent_df"
} # else glb_entity_df is maintained as is for chunk:inspectORexplore.data
    
if (glb_is_separate_newent_dataset) {
    glb_newent_df <- myimport_data(
        url=glb_newdt_url, 
        comment="glb_newent_df", force_header=TRUE)
    
    # To make plots / stats / checks easier in chunk:inspectORexplore.data
    glb_entity_df <- rbind(glb_trnent_df, glb_newent_df); comment(glb_entity_df) <- "glb_entity_df"
} else {
    if (!glb_split_entity_newent_datasets) {
        stop("Not implemented yet") 
        glb_newent_df <- glb_trnent_df[sample(1:nrow(glb_trnent_df),
                                          max(2, nrow(glb_trnent_df) / 1000)),]                    
    } else      if (glb_split_newdata_method == "condition") {
            glb_newent_df <- do.call("subset", 
                list(glb_trnent_df, parse(text=glb_split_newdata_condition)))
            glb_trnent_df <- do.call("subset", 
                list(glb_trnent_df, parse(text=paste0("!(", 
                                                      glb_split_newdata_condition,
                                                      ")"))))
        } else if (glb_split_newdata_method == "sample") {
                require(caTools)
                
                set.seed(glb_split_sample.seed)
                split <- sample.split(glb_trnent_df[, glb_rsp_var_raw], 
                                      SplitRatio=(1-glb_split_newdata_size_ratio))
                glb_newent_df <- glb_trnent_df[!split, ] 
                glb_trnent_df <- glb_trnent_df[split ,]
        } else stop("glb_split_newdata_method should be %in% c('condition', 'sample')")   

    comment(glb_newent_df) <- "glb_newent_df"
    myprint_df(glb_newent_df)
    str(glb_newent_df)

    if (glb_split_entity_newent_datasets) {
        myprint_df(glb_trnent_df)
        str(glb_trnent_df)        
    }
}         
```

```
## Loading required package: caTools
```

```
##    age         workclass     education          maritalstatus
## 2   50  Self-emp-not-inc     Bachelors     Married-civ-spouse
## 5   28           Private     Bachelors     Married-civ-spouse
## 7   49           Private           9th  Married-spouse-absent
## 8   52  Self-emp-not-inc       HS-grad     Married-civ-spouse
## 11  37           Private  Some-college     Married-civ-spouse
## 12  30         State-gov     Bachelors     Married-civ-spouse
##          occupation   relationship                race     sex capitalgain
## 2   Exec-managerial        Husband               White    Male           0
## 5    Prof-specialty           Wife               Black  Female           0
## 7     Other-service  Not-in-family               Black  Female           0
## 8   Exec-managerial        Husband               White    Male           0
## 11  Exec-managerial        Husband               Black    Male           0
## 12   Prof-specialty        Husband  Asian-Pac-Islander    Male           0
##    capitalloss hoursperweek  nativecountry over50k
## 2            0           13  United-States   <=50K
## 5            0           40           Cuba   <=50K
## 7            0           16        Jamaica   <=50K
## 8            0           45  United-States    >50K
## 11           0           80  United-States    >50K
## 12           0           40          India    >50K
##       age         workclass   education       maritalstatus
## 19     43  Self-emp-not-inc     Masters            Divorced
## 13546  44           Private  Assoc-acdm            Divorced
## 16559  49           Private   Assoc-voc  Married-civ-spouse
## 23733  22           Private     HS-grad       Never-married
## 29170  34           Private     HS-grad            Divorced
## 30522  26           Private   Bachelors       Never-married
##               occupation   relationship                race     sex
## 19       Exec-managerial      Unmarried               White  Female
## 13546  Machine-op-inspct      Own-child               White  Female
## 16559       Adm-clerical        Husband               White    Male
## 23733       Adm-clerical      Own-child               White  Female
## 29170   Transport-moving  Not-in-family               White    Male
## 30522    Exec-managerial      Unmarried  Asian-Pac-Islander    Male
##       capitalgain capitalloss hoursperweek  nativecountry over50k
## 19              0           0           45  United-States    >50K
## 13546           0           0           45  United-States   <=50K
## 16559           0           0           40  United-States   <=50K
## 23733           0           0           40  United-States   <=50K
## 29170           0           0           50  United-States   <=50K
## 30522           0           0           60        Vietnam   <=50K
##       age         workclass     education       maritalstatus
## 31966  65  Self-emp-not-inc   Prof-school       Never-married
## 31968  43  Self-emp-not-inc  Some-college  Married-civ-spouse
## 31972  53           Private       Masters  Married-civ-spouse
## 31974  27           Private    Assoc-acdm  Married-civ-spouse
## 31975  40           Private       HS-grad  Married-civ-spouse
## 31976  58           Private       HS-grad             Widowed
##               occupation   relationship   race     sex capitalgain
## 31966     Prof-specialty  Not-in-family  White    Male        1086
## 31968       Craft-repair        Husband  White    Male           0
## 31972    Exec-managerial        Husband  White    Male           0
## 31974       Tech-support           Wife  White  Female           0
## 31975  Machine-op-inspct        Husband  White    Male           0
## 31976       Adm-clerical      Unmarried  White  Female           0
##       capitalloss hoursperweek  nativecountry over50k
## 31966           0           60  United-States   <=50K
## 31968           0           50  United-States   <=50K
## 31972           0           40  United-States    >50K
## 31974           0           38  United-States   <=50K
## 31975           0           40  United-States    >50K
## 31976           0           40  United-States   <=50K
## 'data.frame':	12791 obs. of  13 variables:
##  $ age          : int  50 28 49 52 37 30 23 32 34 43 ...
##  $ workclass    : chr  " Self-emp-not-inc" " Private" " Private" " Self-emp-not-inc" ...
##  $ education    : chr  " Bachelors" " Bachelors" " 9th" " HS-grad" ...
##  $ maritalstatus: chr  " Married-civ-spouse" " Married-civ-spouse" " Married-spouse-absent" " Married-civ-spouse" ...
##  $ occupation   : chr  " Exec-managerial" " Prof-specialty" " Other-service" " Exec-managerial" ...
##  $ relationship : chr  " Husband" " Wife" " Not-in-family" " Husband" ...
##  $ race         : chr  " White" " Black" " Black" " White" ...
##  $ sex          : chr  " Male" " Female" " Female" " Male" ...
##  $ capitalgain  : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ capitalloss  : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ hoursperweek : int  13 40 16 45 80 40 30 50 45 45 ...
##  $ nativecountry: chr  " United-States" " Cuba" " Jamaica" " United-States" ...
##  $ over50k      : chr  " <=50K" " <=50K" " <=50K" " >50K" ...
##  - attr(*, "comment")= chr "glb_newent_df"
##    age  workclass  education       maritalstatus         occupation
## 1   39  State-gov  Bachelors       Never-married       Adm-clerical
## 3   38    Private    HS-grad            Divorced  Handlers-cleaners
## 4   53    Private       11th  Married-civ-spouse  Handlers-cleaners
## 6   37    Private    Masters  Married-civ-spouse    Exec-managerial
## 9   31    Private    Masters       Never-married     Prof-specialty
## 10  42    Private  Bachelors  Married-civ-spouse    Exec-managerial
##      relationship   race     sex capitalgain capitalloss hoursperweek
## 1   Not-in-family  White    Male        2174           0           40
## 3   Not-in-family  White    Male           0           0           40
## 4         Husband  Black    Male           0           0           40
## 6            Wife  White  Female           0           0           40
## 9   Not-in-family  White  Female       14084           0           50
## 10        Husband  White    Male        5178           0           40
##     nativecountry over50k
## 1   United-States   <=50K
## 3   United-States   <=50K
## 4   United-States   <=50K
## 6   United-States   <=50K
## 9   United-States    >50K
## 10  United-States    >50K
##       age         workclass     education       maritalstatus
## 3944   19                 ?  Some-college       Never-married
## 4463   38  Self-emp-not-inc  Some-college  Married-civ-spouse
## 8872   30           Private       HS-grad  Married-civ-spouse
## 15414  28           Private       HS-grad       Never-married
## 17475  67           Private       1st-4th             Widowed
## 30989  42         State-gov  Some-college            Divorced
##               occupation   relationship   race     sex capitalgain
## 3944                   ?      Own-child  White  Female           0
## 4463   Machine-op-inspct        Husband  White    Male        7688
## 8872       Other-service        Husband  Black    Male           0
## 15414  Machine-op-inspct  Not-in-family  White    Male           0
## 17475  Machine-op-inspct  Not-in-family  White  Female        2062
## 30989       Adm-clerical      Unmarried  Black  Female           0
##       capitalloss hoursperweek  nativecountry over50k
## 3944            0           45  United-States   <=50K
## 4463            0           40  United-States    >50K
## 8872            0           25  United-States   <=50K
## 15414           0           50  United-States   <=50K
## 17475           0           34        Ecuador   <=50K
## 30989           0           40  United-States   <=50K
##       age     workclass     education       maritalstatus
## 31969  32       Private          10th  Married-civ-spouse
## 31970  43       Private     Assoc-voc  Married-civ-spouse
## 31971  32       Private       Masters       Never-married
## 31973  22       Private  Some-college       Never-married
## 31977  22       Private       HS-grad       Never-married
## 31978  52  Self-emp-inc       HS-grad  Married-civ-spouse
##               occupation   relationship                race     sex
## 31969  Handlers-cleaners        Husband  Amer-Indian-Eskimo    Male
## 31970              Sales        Husband               White    Male
## 31971       Tech-support  Not-in-family  Asian-Pac-Islander    Male
## 31973    Protective-serv  Not-in-family               White    Male
## 31977       Adm-clerical      Own-child               White    Male
## 31978    Exec-managerial           Wife               White  Female
##       capitalgain capitalloss hoursperweek  nativecountry over50k
## 31969           0           0           40  United-States   <=50K
## 31970           0           0           45  United-States   <=50K
## 31971           0           0           11         Taiwan   <=50K
## 31973           0           0           40  United-States   <=50K
## 31977           0           0           20  United-States   <=50K
## 31978       15024           0           40  United-States    >50K
## 'data.frame':	19187 obs. of  13 variables:
##  $ age          : int  39 38 53 37 31 42 25 32 38 54 ...
##  $ workclass    : chr  " State-gov" " Private" " Private" " Private" ...
##  $ education    : chr  " Bachelors" " HS-grad" " 11th" " Masters" ...
##  $ maritalstatus: chr  " Never-married" " Divorced" " Married-civ-spouse" " Married-civ-spouse" ...
##  $ occupation   : chr  " Adm-clerical" " Handlers-cleaners" " Handlers-cleaners" " Exec-managerial" ...
##  $ relationship : chr  " Not-in-family" " Not-in-family" " Husband" " Wife" ...
##  $ race         : chr  " White" " White" " Black" " White" ...
##  $ sex          : chr  " Male" " Male" " Male" " Female" ...
##  $ capitalgain  : int  2174 0 0 0 14084 5178 0 0 0 0 ...
##  $ capitalloss  : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ hoursperweek : int  40 40 40 40 50 40 35 40 50 20 ...
##  $ nativecountry: chr  " United-States" " United-States" " United-States" " United-States" ...
##  $ over50k      : chr  " <=50K" " <=50K" " <=50K" " <=50K" ...
##  - attr(*, "comment")= chr "glb_trnent_df"
```

```r
if (!is.null(glb_max_obs)) {
    if (nrow(glb_trnent_df) > glb_max_obs) {
        warning("glb_trnent_df restricted to glb_max_obs: ", format(glb_max_obs, big.mark=","))
        org_entity_df <- glb_trnent_df
        glb_trnent_df <- org_entity_df[split <- 
            sample.split(org_entity_df[, glb_rsp_var_raw], SplitRatio=glb_max_obs), ]
        org_entity_df <- NULL
    }
    if (nrow(glb_newent_df) > glb_max_obs) {
        warning("glb_newent_df restricted to glb_max_obs: ", format(glb_max_obs, big.mark=","))        
        org_newent_df <- glb_newent_df
        glb_newent_df <- org_newent_df[split <- 
            sample.split(org_newent_df[, glb_rsp_var_raw], SplitRatio=glb_max_obs), ]
        org_newent_df <- NULL
    }    
}
```

```
## Warning: glb_trnent_df restricted to glb_max_obs: 1,000
```

```
## Warning: glb_newent_df restricted to glb_max_obs: 1,000
```

```r
glb_script_df <- rbind(glb_script_df,
                   data.frame(chunk_label="cleanse_data", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##           chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed   import_data                1                0   0.002
## elapsed1 cleanse_data                2                0   1.654
```

## Step `2`: cleanse data

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="inspectORexplore.data", 
                              chunk_step_major=max(glb_script_df$chunk_step_major), 
                              chunk_step_minor=1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed1          cleanse_data                2                0   1.654
## elapsed2 inspectORexplore.data                2                1   1.692
```

### Step `2`.`1`: inspect/explore data

```r
#print(str(glb_trnent_df))
#View(glb_trnent_df)

# List info gathered for various columns
# <col_name>:   <description>; <notes>
# age = the age of the individual in years
# workclass = the classification of the individual's working status (does the person work for the federal government, work for the local government, work without pay, and so on)
# education = the level of education of the individual (e.g., 5th-6th grade, high school graduate, PhD, so on)
# maritalstatus = the marital status of the individual
# occupation = the type of work the individual does (e.g., administrative/clerical work, farming/fishing, sales and so on)
# relationship = relationship of individual to his/her household
# race = the individual's race
# sex = the individual's sex
# capitalgain = the capital gains of the individual in 1994 (from selling an asset such as a stock or bond for more than the original purchase price)
# capitalloss = the capital losses of the individual in 1994 (from selling an asset such as a stock or bond for less than the original purchase price)
# hoursperweek = the number of hours the individual works per week
# nativecountry = the native country of the individual
# over50k = whether or not the individual earned more than $50,000 in 1994

# Create new features that help diagnostics
#   Create factors of string variables
str_vars <- sapply(1:length(names(glb_trnent_df)), 
    function(col) ifelse(class(glb_trnent_df[, names(glb_trnent_df)[col]]) == "character",
                         names(glb_trnent_df)[col], ""))
if (length(str_vars <- setdiff(str_vars[str_vars != ""], 
                               glb_exclude_vars_as_features)) > 0) {
    warning("Creating factors of string variables:", paste0(str_vars, collapse=", "))
    glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, str_vars)
    for (var in str_vars) {
        glb_entity_df[, paste0(var, ".fctr")] <- factor(glb_entity_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
        glb_trnent_df[, paste0(var, ".fctr")] <- factor(glb_trnent_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
        glb_newent_df[, paste0(var, ".fctr")] <- factor(glb_newent_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
    }
}
```

```
## Warning: Creating factors of string variables:workclass, education,
## maritalstatus, occupation, relationship, race, sex, nativecountry
```

```r
#   Convert factors to dummy variables
#   Build splines   require(splines); bsBasis <- bs(training$age, df=3)

add_new_diag_feats <- function(obs_df, ref_df=glb_entity_df) {
    require(plyr)
    
    obs_df <- mutate(obs_df,
#         <col_name>.NA=is.na(<col_name>),

#         <col_name>.fctr=factor(<col_name>, 
#                     as.factor(union(obs_df$<col_name>, obs_twin_df$<col_name>))), 
#         <col_name>.fctr=relevel(factor(<col_name>, 
#                     as.factor(union(obs_df$<col_name>, obs_twin_df$<col_name>))),
#                                   "<ref_val>"), 
#         <col2_name>.fctr=relevel(factor(ifelse(<col1_name> == <val>, "<oth_val>", "<ref_val>")), 
#                               as.factor(c("R", "<ref_val>")),
#                               ref="<ref_val>"),

          # This doesn't work - use sapply instead
#         <col_name>.fctr_num=grep(<col_name>, levels(<col_name>.fctr)), 
#         
#         Date.my=as.Date(strptime(Date, "%m/%d/%y %H:%M")),
#         Year=year(Date.my),
#         Month=months(Date.my),
#         Weekday=weekdays(Date.my)

#         <col_name>.log=log(<col.name>),        
#         <col_name>=<table>[as.character(<col2_name>)],
#         <col_name>=as.numeric(<col2_name>),

        .rnorm=rnorm(n=nrow(obs_df))
                        )

    # If levels of a factor are different across obs_df & glb_newent_df; predict.glm fails  
    # Transformations not handled by mutate
#     obs_df$<col_name>.fctr.num <- sapply(1:nrow(obs_df), 
#         function(row_ix) grep(obs_df[row_ix, "<col_name>"],
#                               levels(obs_df[row_ix, "<col_name>.fctr"])))
    
    print(summary(obs_df))
    print(sapply(names(obs_df), function(col) sum(is.na(obs_df[, col]))))
    return(obs_df)
}

glb_entity_df <- add_new_diag_feats(glb_entity_df)
```

```
## Loading required package: plyr
```

```
##       age         workclass          education         maritalstatus     
##  Min.   :17.00   Length:31978       Length:31978       Length:31978      
##  1st Qu.:28.00   Class :character   Class :character   Class :character  
##  Median :37.00   Mode  :character   Mode  :character   Mode  :character  
##  Mean   :38.58                                                           
##  3rd Qu.:48.00                                                           
##  Max.   :90.00                                                           
##                                                                          
##   occupation        relationship           race          
##  Length:31978       Length:31978       Length:31978      
##  Class :character   Class :character   Class :character  
##  Mode  :character   Mode  :character   Mode  :character  
##                                                          
##                                                          
##                                                          
##                                                          
##      sex             capitalgain     capitalloss       hoursperweek  
##  Length:31978       Min.   :    0   Min.   :   0.00   Min.   : 1.00  
##  Class :character   1st Qu.:    0   1st Qu.:   0.00   1st Qu.:40.00  
##  Mode  :character   Median :    0   Median :   0.00   Median :40.00  
##                     Mean   : 1064   Mean   :  86.74   Mean   :40.42  
##                     3rd Qu.:    0   3rd Qu.:   0.00   3rd Qu.:45.00  
##                     Max.   :99999   Max.   :4356.00   Max.   :99.00  
##                                                                      
##  nativecountry        over50k                    workclass.fctr 
##  Length:31978       Length:31978        Private         :22286  
##  Class :character   Class :character    Self-emp-not-inc: 2499  
##  Mode  :character   Mode  :character    Local-gov       : 2067  
##                                         ?               : 1809  
##                                         State-gov       : 1279  
##                                         Self-emp-inc    : 1074  
##                                        (Other)          :  964  
##        education.fctr               maritalstatus.fctr
##   HS-grad     :10368    Never-married        :10488   
##   Some-college: 7187    Married-civ-spouse   :14692   
##   Bachelors   : 5210    Divorced             : 4394   
##   Masters     : 1674    Married-spouse-absent:  397   
##   Assoc-voc   : 1366    Separated            : 1005   
##   11th        : 1167    Married-AF-spouse    :   23   
##  (Other)      : 5006    Widowed              :  979   
##          occupation.fctr       relationship.fctr
##   Prof-specialty :4038    Not-in-family : 8156  
##   Craft-repair   :4030    Husband       :12947  
##   Exec-managerial:3992    Wife          : 1534  
##   Adm-clerical   :3721    Own-child     : 5005  
##   Sales          :3584    Unmarried     : 3384  
##   Other-service  :3212    Other-relative:  952  
##  (Other)         :9401                          
##                race.fctr        sex.fctr          nativecountry.fctr
##   White             :27430    Male  :21370    United-States:29170   
##   Black             : 3028    Female:10608    Mexico       :  643   
##   Asian-Pac-Islander:  956                    Philippines  :  198   
##   Amer-Indian-Eskimo:  311                    Germany      :  137   
##   Other             :  253                    Canada       :  121   
##                                               Puerto-Rico  :  114   
##                                              (Other)       : 1595   
##      .rnorm         
##  Min.   :-4.450396  
##  1st Qu.:-0.685706  
##  Median :-0.000782  
##  Mean   :-0.006829  
##  3rd Qu.: 0.669782  
##  Max.   : 4.108640  
##                     
##                age          workclass          education 
##                  0                  0                  0 
##      maritalstatus         occupation       relationship 
##                  0                  0                  0 
##               race                sex        capitalgain 
##                  0                  0                  0 
##        capitalloss       hoursperweek      nativecountry 
##                  0                  0                  0 
##            over50k     workclass.fctr     education.fctr 
##                  0                  0                  0 
## maritalstatus.fctr    occupation.fctr  relationship.fctr 
##                  0                  0                  0 
##          race.fctr           sex.fctr nativecountry.fctr 
##                  0                  0                  0 
##             .rnorm 
##                  0
```

```r
glb_trnent_df <- add_new_diag_feats(glb_trnent_df)
```

```
##       age         workclass          education         maritalstatus     
##  Min.   :17.00   Length:1000        Length:1000        Length:1000       
##  1st Qu.:28.00   Class :character   Class :character   Class :character  
##  Median :37.00   Mode  :character   Mode  :character   Mode  :character  
##  Mean   :38.88                                                           
##  3rd Qu.:47.00                                                           
##  Max.   :90.00                                                           
##                                                                          
##   occupation        relationship           race          
##  Length:1000        Length:1000        Length:1000       
##  Class :character   Class :character   Class :character  
##  Mode  :character   Mode  :character   Mode  :character  
##                                                          
##                                                          
##                                                          
##                                                          
##      sex             capitalgain     capitalloss       hoursperweek 
##  Length:1000        Min.   :    0   Min.   :   0.00   Min.   : 1.0  
##  Class :character   1st Qu.:    0   1st Qu.:   0.00   1st Qu.:40.0  
##  Mode  :character   Median :    0   Median :   0.00   Median :40.0  
##                     Mean   : 1107   Mean   :  91.74   Mean   :40.7  
##                     3rd Qu.:    0   3rd Qu.:   0.00   3rd Qu.:45.0  
##                     Max.   :99999   Max.   :2457.00   Max.   :99.0  
##                                                                     
##  nativecountry        over50k                    workclass.fctr
##  Length:1000        Length:1000         Private         :685   
##  Class :character   Class :character    Self-emp-not-inc: 77   
##  Mode  :character   Mode  :character    Local-gov       : 77   
##                                         ?               : 52   
##                                         Self-emp-inc    : 42   
##                                         State-gov       : 36   
##                                        (Other)          : 31   
##        education.fctr              maritalstatus.fctr
##   HS-grad     :326     Never-married        :314     
##   Some-college:251     Married-civ-spouse   :461     
##   Bachelors   :155     Divorced             :148     
##   Masters     : 60     Married-spouse-absent: 10     
##   Assoc-voc   : 46     Separated            : 36     
##   10th        : 37     Married-AF-spouse    :  1     
##  (Other)      :125     Widowed              : 30     
##          occupation.fctr       relationship.fctr               race.fctr  
##   Exec-managerial:129     Not-in-family :245      White             :862  
##   Prof-specialty :129     Husband       :408      Black             : 88  
##   Craft-repair   :125     Wife          : 52      Asian-Pac-Islander: 30  
##   Adm-clerical   :122     Own-child     :159      Amer-Indian-Eskimo:  9  
##   Sales          :112     Unmarried     :114      Other             : 11  
##   Other-service  :103     Other-relative: 22                              
##  (Other)         :280                                                     
##     sex.fctr        nativecountry.fctr     .rnorm        
##   Male  :664    United-States:910      Min.   :-3.04264  
##   Female:336    Mexico       : 21      1st Qu.:-0.59901  
##                 India        :  6      Median : 0.02307  
##                 El-Salvador  :  6      Mean   : 0.02115  
##                 England      :  5      3rd Qu.: 0.68508  
##                 Germany      :  5      Max.   : 3.63423  
##                (Other)       : 47                        
##                age          workclass          education 
##                  0                  0                  0 
##      maritalstatus         occupation       relationship 
##                  0                  0                  0 
##               race                sex        capitalgain 
##                  0                  0                  0 
##        capitalloss       hoursperweek      nativecountry 
##                  0                  0                  0 
##            over50k     workclass.fctr     education.fctr 
##                  0                  0                  0 
## maritalstatus.fctr    occupation.fctr  relationship.fctr 
##                  0                  0                  0 
##          race.fctr           sex.fctr nativecountry.fctr 
##                  0                  0                  0 
##             .rnorm 
##                  0
```

```r
glb_newent_df <- add_new_diag_feats(glb_newent_df)
```

```
##       age         workclass          education         maritalstatus     
##  Min.   :17.00   Length:1000        Length:1000        Length:1000       
##  1st Qu.:27.00   Class :character   Class :character   Class :character  
##  Median :37.00   Mode  :character   Mode  :character   Mode  :character  
##  Mean   :38.56                                                           
##  3rd Qu.:47.25                                                           
##  Max.   :82.00                                                           
##                                                                          
##   occupation        relationship           race          
##  Length:1000        Length:1000        Length:1000       
##  Class :character   Class :character   Class :character  
##  Mode  :character   Mode  :character   Mode  :character  
##                                                          
##                                                          
##                                                          
##                                                          
##      sex             capitalgain     capitalloss       hoursperweek  
##  Length:1000        Min.   :    0   Min.   :   0.00   Min.   : 3.00  
##  Class :character   1st Qu.:    0   1st Qu.:   0.00   1st Qu.:40.00  
##  Mode  :character   Median :    0   Median :   0.00   Median :40.00  
##                     Mean   : 1183   Mean   :  97.81   Mean   :39.57  
##                     3rd Qu.:    0   3rd Qu.:   0.00   3rd Qu.:45.00  
##                     Max.   :99999   Max.   :2392.00   Max.   :99.00  
##                                                                      
##  nativecountry        over50k                    workclass.fctr
##  Length:1000        Length:1000         Private         :698   
##  Class :character   Class :character    Local-gov       : 73   
##  Mode  :character   Mode  :character    ?               : 69   
##                                         Self-emp-not-inc: 67   
##                                         Federal-gov     : 34   
##                                         State-gov       : 33   
##                                        (Other)          : 26   
##        education.fctr              maritalstatus.fctr
##   HS-grad     :323     Never-married        :318     
##   Some-college:222     Married-civ-spouse   :467     
##   Bachelors   :159     Divorced             :141     
##   Masters     : 68     Married-spouse-absent: 13     
##   11th        : 44     Separated            : 27     
##   Assoc-voc   : 39     Married-AF-spouse    :  1     
##  (Other)      :145     Widowed              : 33     
##          occupation.fctr       relationship.fctr               race.fctr  
##   Exec-managerial:137     Not-in-family :254      White             :865  
##   Prof-specialty :130     Husband       :407      Black             : 95  
##   Sales          :121     Wife          : 51      Asian-Pac-Islander: 29  
##   Craft-repair   :120     Own-child     :148      Amer-Indian-Eskimo:  5  
##   Adm-clerical   :116     Unmarried     :112      Other             :  6  
##   Other-service  : 93     Other-relative: 28                              
##  (Other)         :283                                                     
##     sex.fctr        nativecountry.fctr     .rnorm        
##   Male  :647    United-States:905      Min.   :-3.11383  
##   Female:353    Mexico       : 19      1st Qu.:-0.73504  
##                 Germany      :  8      Median :-0.07260  
##                 Philippines  :  7      Mean   :-0.05372  
##                 Iran         :  5      3rd Qu.: 0.61448  
##                 Puerto-Rico  :  4      Max.   : 2.85309  
##                (Other)       : 52                        
##                age          workclass          education 
##                  0                  0                  0 
##      maritalstatus         occupation       relationship 
##                  0                  0                  0 
##               race                sex        capitalgain 
##                  0                  0                  0 
##        capitalloss       hoursperweek      nativecountry 
##                  0                  0                  0 
##            over50k     workclass.fctr     education.fctr 
##                  0                  0                  0 
## maritalstatus.fctr    occupation.fctr  relationship.fctr 
##                  0                  0                  0 
##          race.fctr           sex.fctr nativecountry.fctr 
##                  0                  0                  0 
##             .rnorm 
##                  0
```

```r
# Histogram of predictor in glb_trnent_df & glb_newent_df
plot_df <- rbind(cbind(glb_trnent_df[, glb_rsp_var_raw, FALSE], data.frame(.data="Training")),
                 cbind(glb_trnent_df[, glb_rsp_var_raw, FALSE], data.frame(.data="New")))
print(myplot_histogram(plot_df, glb_rsp_var_raw) + facet_wrap(~ .data))
```

```
## Warning in mean.default(sort(x, partial = half + 0L:1L)[half + 0L:1L]):
## argument is not numeric or logical: returning NA
```

```
## Warning: Removed 1 rows containing missing values (geom_segment).
```

```
## Warning: Removed 1 rows containing missing values (geom_segment).
```

![](USCensus_Earnings_files/figure-html/inspectORexplore_data-1.png) 

```r
if (glb_is_classification) {
    xtab_df <- mycreate_xtab(plot_df, c(".data", glb_rsp_var_raw))
    rownames(xtab_df) <- xtab_df$.data
    xtab_df <- subset(xtab_df, select=-.data)
    print(xtab_df / rowSums(xtab_df))    
}    
```

```
## Loading required package: reshape2
```

```
##          over50k. <=50K over50k. >50K
## New               0.759         0.241
## Training          0.759         0.241
```

```r
# Check for duplicates in glb_id_vars
if (length(glb_id_vars) > 0) {
    id_vars_dups_df <- subset(id_vars_df <- 
            mycreate_tbl_df(glb_entity_df[, glb_id_vars, FALSE], glb_id_vars),
                                .freq > 1)
    if (nrow(id_vars_dups_df) > 0) {
        warning("Duplicates found in glb_id_vars data:", nrow(id_vars_dups_df))
        myprint_df(id_vars_dups_df)
    } else {
        # glb_id_vars are unique across obs in both glb_<>_df
        glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, glb_id_vars)
    }
}

#pairs(subset(glb_trnent_df, select=-c(col_symbol)))
# Check for glb_newent_df & glb_trnent_df features range mismatches

# Other diagnostics:
# print(subset(glb_trnent_df, <col1_name> == max(glb_trnent_df$<col1_name>, na.rm=TRUE) & 
#                         <col2_name> <= mean(glb_trnent_df$<col1_name>, na.rm=TRUE)))

# print(glb_trnent_df[which.max(glb_trnent_df$<col_name>),])

# print(<col_name>_freq_glb_trnent_df <- mycreate_tbl_df(glb_trnent_df, "<col_name>"))
# print(which.min(table(glb_trnent_df$<col_name>)))
# print(which.max(table(glb_trnent_df$<col_name>)))
# print(which.max(table(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>)[, 2]))
# print(table(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>))
# print(table(is.na(glb_trnent_df$<col1_name>), glb_trnent_df$<col2_name>))
# print(table(sign(glb_trnent_df$<col1_name>), glb_trnent_df$<col2_name>))
# print(mycreate_xtab(glb_trnent_df, <col1_name>))
# print(mycreate_xtab(glb_trnent_df, c(<col1_name>, <col2_name>)))
# print(<col1_name>_<col2_name>_xtab_glb_trnent_df <- 
#   mycreate_xtab(glb_trnent_df, c("<col1_name>", "<col2_name>")))
# <col1_name>_<col2_name>_xtab_glb_trnent_df[is.na(<col1_name>_<col2_name>_xtab_glb_trnent_df)] <- 0
# print(<col1_name>_<col2_name>_xtab_glb_trnent_df <- 
#   mutate(<col1_name>_<col2_name>_xtab_glb_trnent_df, 
#             <col3_name>=(<col1_name> * 1.0) / (<col1_name> + <col2_name>))) 

# print(<col2_name>_min_entity_arr <- 
#    sort(tapply(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>, min, na.rm=TRUE)))
# print(<col1_name>_na_by_<col2_name>_arr <- 
#    sort(tapply(glb_trnent_df$<col1_name>.NA, glb_trnent_df$<col2_name>, mean, na.rm=TRUE)))

# Other plots:
# print(myplot_box(df=glb_trnent_df, ycol_names="<col1_name>"))
# print(myplot_box(df=glb_trnent_df, ycol_names="<col1_name>", xcol_name="<col2_name>"))
# print(myplot_line(subset(glb_trnent_df, Symbol %in% c("KO", "PG")), 
#                   "Date.my", "StockPrice", facet_row_colnames="Symbol") + 
#     geom_vline(xintercept=as.numeric(as.Date("2003-03-01"))) +
#     geom_vline(xintercept=as.numeric(as.Date("1983-01-01")))        
#         )
# print(myplot_scatter(glb_trnent_df, "<col1_name>", "<col2_name>", smooth=TRUE))
# print(myplot_scatter(glb_entity_df, "<col1_name>", "<col2_name>", colorcol_name="<Pred.fctr>") + 
#         geom_point(data=subset(glb_entity_df, <condition>), 
#                     mapping=aes(x=<x_var>, y=<y_var>), color="red", shape=4, size=5))

glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="manage_missing_data", 
        chunk_step_major=max(glb_script_df$chunk_step_major), 
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed2 inspectORexplore.data                2                1   1.692
## elapsed3   manage_missing_data                2                2   2.860
```

### Step `2`.`2`: manage missing data

```r
# print(sapply(names(glb_trnent_df), function(col) sum(is.na(glb_trnent_df[, col]))))
# print(sapply(names(glb_newent_df), function(col) sum(is.na(glb_newent_df[, col]))))
# glb_trnent_df <- na.omit(glb_trnent_df)
# glb_newent_df <- na.omit(glb_newent_df)
# df[is.na(df)] <- 0

# Not refactored into mydsutils.R since glb_*_df might be reassigned
glb_impute_missing_data <- function(entity_df, newent_df) {
    if (!glb_is_separate_newent_dataset) {
        # Combine entity & newent
        union_df <- rbind(mutate(entity_df, .src = "entity"),
                          mutate(newent_df, .src = "newent"))
        union_imputed_df <- union_df[, setdiff(setdiff(names(entity_df), 
                                                       glb_rsp_var), 
                                               glb_exclude_vars_as_features)]
        print(summary(union_imputed_df))
    
        require(mice)
        set.seed(glb_mice_complete.seed)
        union_imputed_df <- complete(mice(union_imputed_df))
        print(summary(union_imputed_df))
    
        union_df[, names(union_imputed_df)] <- union_imputed_df[, names(union_imputed_df)]
        print(summary(union_df))
#         union_df$.rownames <- rownames(union_df)
#         union_df <- orderBy(~.rownames, union_df)
#         
#         imp_entity_df <- myimport_data(
#             url="<imputed_trnng_url>", 
#             comment="imp_entity_df", force_header=TRUE, print_diagn=TRUE)
#         print(all.equal(subset(union_df, select=-c(.src, .rownames, .rnorm)), 
#                         imp_entity_df))
        
        # Partition again
        glb_trnent_df <<- subset(union_df, .src == "entity", select=-c(.src, .rownames))
        comment(glb_trnent_df) <- "entity_df"
        glb_newent_df <<- subset(union_df, .src == "newent", select=-c(.src, .rownames))
        comment(glb_newent_df) <- "newent_df"
        
        # Generate summaries
        print(summary(entity_df))
        print(sapply(names(entity_df), function(col) sum(is.na(entity_df[, col]))))
        print(summary(newent_df))
        print(sapply(names(newent_df), function(col) sum(is.na(newent_df[, col]))))
    
    } else stop("Not implemented yet")
}

if (glb_impute_na_data) {
    if ((sum(sapply(names(glb_trnent_df), 
                    function(col) sum(is.na(glb_trnent_df[, col])))) > 0) | 
        (sum(sapply(names(glb_newent_df), 
                    function(col) sum(is.na(glb_newent_df[, col])))) > 0))
        glb_impute_missing_data(glb_trnent_df, glb_newent_df)
}    

glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="encode_retype_data", 
        chunk_step_major=max(glb_script_df$chunk_step_major), 
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                  chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed3 manage_missing_data                2                2   2.860
## elapsed4  encode_retype_data                2                3   3.141
```

### Step `2`.`3`: encode/retype data

```r
# map_<col_name>_df <- myimport_data(
#     url="<map_url>", 
#     comment="map_<col_name>_df", print_diagn=TRUE)
# map_<col_name>_df <- read.csv(paste0(getwd(), "/data/<file_name>.csv"), strip.white=TRUE)

# glb_trnent_df <- mymap_codes(glb_trnent_df, "<from_col_name>", "<to_col_name>", 
#     map_<to_col_name>_df, map_join_col_name="<map_join_col_name>", 
#                           map_tgt_col_name="<to_col_name>")
# glb_newent_df <- mymap_codes(glb_newent_df, "<from_col_name>", "<to_col_name>", 
#     map_<to_col_name>_df, map_join_col_name="<map_join_col_name>", 
#                           map_tgt_col_name="<to_col_name>")
    					
# glb_trnent_df$<col_name>.fctr <- factor(glb_trnent_df$<col_name>, 
#                     as.factor(union(glb_trnent_df$<col_name>, glb_newent_df$<col_name>)))
# glb_newent_df$<col_name>.fctr <- factor(glb_newent_df$<col_name>, 
#                     as.factor(union(glb_trnent_df$<col_name>, glb_newent_df$<col_name>)))

if (!is.null(glb_map_rsp_raw_to_var)) {
    glb_entity_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_entity_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_entity_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)
        
    glb_trnent_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_trnent_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_trnent_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)
        
    glb_newent_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_newent_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_newent_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)    
}
```

```
## Loading required package: sqldf
## Loading required package: gsubfn
## Loading required package: proto
## Loading required package: RSQLite
## Loading required package: DBI
## Loading required package: tcltk
```

```
##   over50k over50k.fctr    .n
## 1   <=50K            N 24283
## 2    >50K            Y  7695
```

![](USCensus_Earnings_files/figure-html/encode_retype_data_1-1.png) 

```
##   over50k over50k.fctr  .n
## 1   <=50K            N 759
## 2    >50K            Y 241
```

![](USCensus_Earnings_files/figure-html/encode_retype_data_1-2.png) 

```
##   over50k over50k.fctr  .n
## 1   <=50K            N 759
## 2    >50K            Y 241
```

![](USCensus_Earnings_files/figure-html/encode_retype_data_1-3.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="extract_features", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                 chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed4 encode_retype_data                2                3   3.141
## elapsed5   extract_features                3                0   6.361
```

## Step `3`: extract features

```r
# Create new features that help prediction
# <col_name>.lag.2 <- lag(zoo(glb_trnent_df$<col_name>), -2, na.pad=TRUE)
# glb_trnent_df[, "<col_name>.lag.2"] <- coredata(<col_name>.lag.2)
# <col_name>.lag.2 <- lag(zoo(glb_newent_df$<col_name>), -2, na.pad=TRUE)
# glb_newent_df[, "<col_name>.lag.2"] <- coredata(<col_name>.lag.2)
# 
# glb_newent_df[1, "<col_name>.lag.2"] <- glb_trnent_df[nrow(glb_trnent_df) - 1, 
#                                                    "<col_name>"]
# glb_newent_df[2, "<col_name>.lag.2"] <- glb_trnent_df[nrow(glb_trnent_df), 
#                                                    "<col_name>"]
                                                   
# glb_trnent_df <- mutate(glb_trnent_df,
#     <new_col_name>=
#                     )

# glb_newent_df <- mutate(glb_newent_df,
#     <new_col_name>=
#                     )

# print(summary(glb_trnent_df))
# print(summary(glb_newent_df))

# print(sapply(names(glb_trnent_df), function(col) sum(is.na(glb_trnent_df[, col]))))
# print(sapply(names(glb_newent_df), function(col) sum(is.na(glb_newent_df[, col]))))

# print(myplot_scatter(glb_trnent_df, "<col1_name>", "<col2_name>", smooth=TRUE))

replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all","data.new")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0
```

![](USCensus_Earnings_files/figure-html/extract_features-1.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="select_features", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##               chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed5 extract_features                3                0   6.361
## elapsed6  select_features                4                0   7.735
```

## Step `4`: select features

```r
print(glb_feats_df <- myselect_features(entity_df=glb_trnent_df, 
                       exclude_vars_as_features=glb_exclude_vars_as_features, 
                       rsp_var=glb_rsp_var))
```

```
##                                    id       cor.y exclude.as.feat
## age                               age  0.24780278               0
## capitalgain               capitalgain  0.23222327               0
## hoursperweek             hoursperweek  0.22402106               0
## sex.fctr                     sex.fctr -0.20283667               0
## relationship.fctr   relationship.fctr -0.12845800               0
## occupation.fctr       occupation.fctr -0.12686813               0
## capitalloss               capitalloss  0.10182189               0
## workclass.fctr         workclass.fctr  0.07835390               0
## nativecountry.fctr nativecountry.fctr -0.06238053               0
## education.fctr         education.fctr -0.06153968               0
## .rnorm                         .rnorm -0.01983209               0
## race.fctr                   race.fctr -0.01730292               0
## maritalstatus.fctr maritalstatus.fctr  0.01310876               0
##                     cor.y.abs
## age                0.24780278
## capitalgain        0.23222327
## hoursperweek       0.22402106
## sex.fctr           0.20283667
## relationship.fctr  0.12845800
## occupation.fctr    0.12686813
## capitalloss        0.10182189
## workclass.fctr     0.07835390
## nativecountry.fctr 0.06238053
## education.fctr     0.06153968
## .rnorm             0.01983209
## race.fctr          0.01730292
## maritalstatus.fctr 0.01310876
```

```r
glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="remove_correlated_features", 
        chunk_step_major=max(glb_script_df$chunk_step_major),
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))        
print(tail(glb_script_df, 2))
```

```
##                         chunk_label chunk_step_major chunk_step_minor
## elapsed6            select_features                4                0
## elapsed7 remove_correlated_features                4                1
##          elapsed
## elapsed6   7.735
## elapsed7   7.936
```

### Step `4`.`1`: remove correlated features

```r
print(glb_feats_df <- orderBy(~-cor.y, 
          myfind_cor_features(feats_df=glb_feats_df, entity_df=glb_trnent_df, 
                                rsp_var=glb_rsp_var)))
```

```
##                             age  capitalgain  hoursperweek     sex.fctr
## age                 1.000000000  0.072443308  0.0110659302 -0.080920555
## capitalgain         0.072443308  1.000000000  0.1217883888 -0.075673733
## hoursperweek        0.011065930  0.121788389  1.0000000000 -0.209041786
## sex.fctr           -0.080920555 -0.075673733 -0.2090417861  1.000000000
## relationship.fctr  -0.190618059 -0.029404101 -0.1659531807  0.285002802
## occupation.fctr     0.040943039 -0.070899545 -0.0869868797 -0.176193489
## capitalloss         0.110511930 -0.033355499  0.0761720818 -0.045816770
## workclass.fctr      0.092168662 -0.011794652  0.0179896735  0.017949975
## nativecountry.fctr -0.012988665 -0.009189994 -0.0009124827  0.002655962
## education.fctr     -0.005613266  0.037463070 -0.0403007629  0.017736991
## .rnorm              0.035211871 -0.014286016  0.0176255140 -0.028654619
## race.fctr          -0.054878458  0.012941440 -0.0240451637  0.066931205
## maritalstatus.fctr  0.430017861  0.003097908 -0.0338687249  0.210674206
##                    relationship.fctr occupation.fctr  capitalloss
## age                     -0.190618059     0.040943039  0.110511930
## capitalgain             -0.029404101    -0.070899545 -0.033355499
## hoursperweek            -0.165953181    -0.086986880  0.076172082
## sex.fctr                 0.285002802    -0.176193489 -0.045816770
## relationship.fctr        1.000000000    -0.042382363 -0.051318073
## occupation.fctr         -0.042382363     1.000000000 -0.070478487
## capitalloss             -0.051318073    -0.070478487  1.000000000
## workclass.fctr           0.026873152     0.112771328  0.035862175
## nativecountry.fctr       0.049642761     0.142303542 -0.021438489
## education.fctr           0.047961487     0.102991556  0.019690262
## .rnorm                  -0.068910555     0.019105621 -0.047871152
## race.fctr                0.086935095     0.081786816 -0.041638604
## maritalstatus.fctr      -0.003588018     0.007456857 -0.008739605
##                    workclass.fctr nativecountry.fctr education.fctr
## age                  0.0921686621      -0.0129886652   -0.005613266
## capitalgain         -0.0117946522      -0.0091899941    0.037463070
## hoursperweek         0.0179896735      -0.0009124827   -0.040300763
## sex.fctr             0.0179499750       0.0026559619    0.017736991
## relationship.fctr    0.0268731522       0.0496427605    0.047961487
## occupation.fctr      0.1127713282       0.1423035422    0.102991556
## capitalloss          0.0358621748      -0.0214384887    0.019690262
## workclass.fctr       1.0000000000       0.0217407910   -0.077275365
## nativecountry.fctr   0.0217407910       1.0000000000    0.061613276
## education.fctr      -0.0772753645       0.0616132761    1.000000000
## .rnorm               0.0003025034       0.0204038404    0.028315437
## race.fctr            0.1083651907       0.1666146385    0.024751420
## maritalstatus.fctr   0.0269878295      -0.0184864395   -0.012940592
##                           .rnorm    race.fctr maritalstatus.fctr
## age                 0.0352118707 -0.054878458        0.430017861
## capitalgain        -0.0142860157  0.012941440        0.003097908
## hoursperweek        0.0176255140 -0.024045164       -0.033868725
## sex.fctr           -0.0286546193  0.066931205        0.210674206
## relationship.fctr  -0.0689105546  0.086935095       -0.003588018
## occupation.fctr     0.0191056214  0.081786816        0.007456857
## capitalloss        -0.0478711523 -0.041638604       -0.008739605
## workclass.fctr      0.0003025034  0.108365191        0.026987830
## nativecountry.fctr  0.0204038404  0.166614638       -0.018486439
## education.fctr      0.0283154370  0.024751420       -0.012940592
## .rnorm              1.0000000000  0.008562811        0.003017561
## race.fctr           0.0085628113  1.000000000       -0.014055313
## maritalstatus.fctr  0.0030175606 -0.014055313        1.000000000
##                            age capitalgain hoursperweek    sex.fctr
## age                0.000000000 0.072443308 0.0110659302 0.080920555
## capitalgain        0.072443308 0.000000000 0.1217883888 0.075673733
## hoursperweek       0.011065930 0.121788389 0.0000000000 0.209041786
## sex.fctr           0.080920555 0.075673733 0.2090417861 0.000000000
## relationship.fctr  0.190618059 0.029404101 0.1659531807 0.285002802
## occupation.fctr    0.040943039 0.070899545 0.0869868797 0.176193489
## capitalloss        0.110511930 0.033355499 0.0761720818 0.045816770
## workclass.fctr     0.092168662 0.011794652 0.0179896735 0.017949975
## nativecountry.fctr 0.012988665 0.009189994 0.0009124827 0.002655962
## education.fctr     0.005613266 0.037463070 0.0403007629 0.017736991
## .rnorm             0.035211871 0.014286016 0.0176255140 0.028654619
## race.fctr          0.054878458 0.012941440 0.0240451637 0.066931205
## maritalstatus.fctr 0.430017861 0.003097908 0.0338687249 0.210674206
##                    relationship.fctr occupation.fctr capitalloss
## age                      0.190618059     0.040943039 0.110511930
## capitalgain              0.029404101     0.070899545 0.033355499
## hoursperweek             0.165953181     0.086986880 0.076172082
## sex.fctr                 0.285002802     0.176193489 0.045816770
## relationship.fctr        0.000000000     0.042382363 0.051318073
## occupation.fctr          0.042382363     0.000000000 0.070478487
## capitalloss              0.051318073     0.070478487 0.000000000
## workclass.fctr           0.026873152     0.112771328 0.035862175
## nativecountry.fctr       0.049642761     0.142303542 0.021438489
## education.fctr           0.047961487     0.102991556 0.019690262
## .rnorm                   0.068910555     0.019105621 0.047871152
## race.fctr                0.086935095     0.081786816 0.041638604
## maritalstatus.fctr       0.003588018     0.007456857 0.008739605
##                    workclass.fctr nativecountry.fctr education.fctr
## age                  0.0921686621       0.0129886652    0.005613266
## capitalgain          0.0117946522       0.0091899941    0.037463070
## hoursperweek         0.0179896735       0.0009124827    0.040300763
## sex.fctr             0.0179499750       0.0026559619    0.017736991
## relationship.fctr    0.0268731522       0.0496427605    0.047961487
## occupation.fctr      0.1127713282       0.1423035422    0.102991556
## capitalloss          0.0358621748       0.0214384887    0.019690262
## workclass.fctr       0.0000000000       0.0217407910    0.077275365
## nativecountry.fctr   0.0217407910       0.0000000000    0.061613276
## education.fctr       0.0772753645       0.0616132761    0.000000000
## .rnorm               0.0003025034       0.0204038404    0.028315437
## race.fctr            0.1083651907       0.1666146385    0.024751420
## maritalstatus.fctr   0.0269878295       0.0184864395    0.012940592
##                          .rnorm   race.fctr maritalstatus.fctr
## age                0.0352118707 0.054878458        0.430017861
## capitalgain        0.0142860157 0.012941440        0.003097908
## hoursperweek       0.0176255140 0.024045164        0.033868725
## sex.fctr           0.0286546193 0.066931205        0.210674206
## relationship.fctr  0.0689105546 0.086935095        0.003588018
## occupation.fctr    0.0191056214 0.081786816        0.007456857
## capitalloss        0.0478711523 0.041638604        0.008739605
## workclass.fctr     0.0003025034 0.108365191        0.026987830
## nativecountry.fctr 0.0204038404 0.166614638        0.018486439
## education.fctr     0.0283154370 0.024751420        0.012940592
## .rnorm             0.0000000000 0.008562811        0.003017561
## race.fctr          0.0085628113 0.000000000        0.014055313
## maritalstatus.fctr 0.0030175606 0.014055313        0.000000000
##                                    id       cor.y exclude.as.feat
## age                               age  0.24780278               0
## capitalgain               capitalgain  0.23222327               0
## hoursperweek             hoursperweek  0.22402106               0
## capitalloss               capitalloss  0.10182189               0
## workclass.fctr         workclass.fctr  0.07835390               0
## maritalstatus.fctr maritalstatus.fctr  0.01310876               0
## race.fctr                   race.fctr -0.01730292               0
## .rnorm                         .rnorm -0.01983209               0
## education.fctr         education.fctr -0.06153968               0
## nativecountry.fctr nativecountry.fctr -0.06238053               0
## occupation.fctr       occupation.fctr -0.12686813               0
## relationship.fctr   relationship.fctr -0.12845800               0
## sex.fctr                     sex.fctr -0.20283667               0
##                     cor.y.abs cor.low
## age                0.24780278       1
## capitalgain        0.23222327       1
## hoursperweek       0.22402106       1
## capitalloss        0.10182189       1
## workclass.fctr     0.07835390       1
## maritalstatus.fctr 0.01310876       1
## race.fctr          0.01730292       1
## .rnorm             0.01983209       1
## education.fctr     0.06153968       1
## nativecountry.fctr 0.06238053       1
## occupation.fctr    0.12686813       1
## relationship.fctr  0.12845800       1
## sex.fctr           0.20283667       1
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.models", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                         chunk_label chunk_step_major chunk_step_minor
## elapsed7 remove_correlated_features                4                1
## elapsed8                 fit.models                5                0
##          elapsed
## elapsed7   7.936
## elapsed8   7.989
```

## Step `5`: fit models

```r
if (glb_is_classification && glb_is_binomial && (length(unique(glb_trnent_df[, glb_rsp_var])) < 2))
    stop("glb_trnent_df$", glb_rsp_var, ": contains less than 2 unique values: ", paste0(unique(glb_trnent_df[, glb_rsp_var]), collapse=", "))

max_cor_y_x_var <- orderBy(~ -cor.y.abs, 
        subset(glb_feats_df, (exclude.as.feat == 0) & (cor.low == 1)))[1, "id"]
if (!is.null(glb_Baseline_mdl_var)) {
    if ((max_cor_y_x_var != glb_Baseline_mdl_var) & 
        (glb_feats_df[max_cor_y_x_var, "cor.y.abs"] > 
         glb_feats_df[glb_Baseline_mdl_var, "cor.y.abs"]))
        stop(max_cor_y_x_var, " has a lower correlation with ", glb_rsp_var, 
             " than the Baseline var: ", glb_Baseline_mdl_var)
}

glb_model_type <- ifelse(glb_is_regression, "regression", "classification")
    
# Any models that have tuning parameters has "better" results with cross-validation (except rf)
#   & "different" results for different outcome metrics

# Baseline
if (!is.null(glb_Baseline_mdl_var)) {
#     lm_mdl <- lm(reformulate(glb_Baseline_mdl_var, 
#                             response="bucket2009"), data=glb_trnent_df)
#     print(summary(lm_mdl))
#     plot(lm_mdl, ask=FALSE)
#     ret_lst <- myfit_mdl_fn(model_id="Baseline", 
#                             model_method=ifelse(glb_is_regression, "lm", 
#                                         ifelse(glb_is_binomial, "glm", "rpart")),
#                             indep_vars_vctr=glb_Baseline_mdl_var,
#                             rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
#                             fit_df=glb_trnent_df, OOB_df=glb_newent_df,
#                             n_cv_folds=0, tune_models_df=NULL,
#                             model_loss_mtrx=glb_model_metric_terms,
#                             model_summaryFunction=glb_model_metric_smmry,
#                             model_metric=glb_model_metric,
#                             model_metric_maximize=glb_model_metric_maximize)
    ret_lst <- myfit_mdl_fn(model_id="Baseline", model_method="mybaseln_classfr",
                            indep_vars_vctr=glb_Baseline_mdl_var,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df)
}

# Most Frequent Outcome "MFO" model: mean(y) for regression
#   Not using caret's nullModel since model stats not avl
#   Cannot use rpart for multinomial classification since it predicts non-MFO
ret_lst <- myfit_mdl(model_id="MFO", 
                     model_method=ifelse(glb_is_regression, "lm", "myMFO_classfr"), 
                     model_type=glb_model_type,
                        indep_vars_vctr=".rnorm",
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## Loading required package: caret
## Loading required package: lattice
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:survival':
## 
##     cluster
```

```
## [1] "fitting model: MFO.myMFO_classfr"
## [1] "    indep_vars: .rnorm"
## Fitting parameter = none on full training set
## [1] "in MFO.Classifier$fit"
## [1] "unique.vals:"
## [1] N Y
## Levels: N Y
## [1] "unique.prob:"
## y
##     N     Y 
## 0.759 0.241 
## [1] "MFO.val:"
## [1] "N"
##             Length Class      Mode     
## unique.vals 2      factor     numeric  
## unique.prob 2      -none-     numeric  
## MFO.val     1      -none-     character
## x.names     1      -none-     character
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   2      -none-     character
```

```
## Loading required package: ROCR
## Loading required package: gplots
## 
## Attaching package: 'gplots'
## 
## The following object is masked from 'package:stats':
## 
##     lowess
```

```
## [1] "in MFO.Classifier$predict"
## [1] "in MFO.Classifier$prob"
##       N     Y
## 1 0.759 0.241
## 2 0.759 0.241
## 3 0.759 0.241
## 4 0.759 0.241
## 5 0.759 0.241
## 6 0.759 0.241
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.MFO.myMFO_classfr.N
## 1            N                                      759
## 2            Y                                      241
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.MFO.myMFO_classfr.N
## 1            N                                      759
## 2            Y                                      241
##   over50k.fctr.predict.MFO.myMFO_classfr.Y
## 1                                        0
## 2                                        0
##          Prediction
## Reference   N   Y
##         N 759   0
##         Y 241   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.590000e-01   0.000000e+00   7.312521e-01   7.852137e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   5.172861e-01   6.484063e-54 
## [1] "in MFO.Classifier$predict"
## [1] "in MFO.Classifier$prob"
##       N     Y
## 1 0.759 0.241
## 2 0.759 0.241
## 3 0.759 0.241
## 4 0.759 0.241
## 5 0.759 0.241
## 6 0.759 0.241
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.MFO.myMFO_classfr.N
## 1            N                                      759
## 2            Y                                      241
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.MFO.myMFO_classfr.N
## 1            N                                      759
## 2            Y                                      241
##   over50k.fctr.predict.MFO.myMFO_classfr.Y
## 1                                        0
## 2                                        0
##          Prediction
## Reference   N   Y
##         N 759   0
##         Y 241   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.590000e-01   0.000000e+00   7.312521e-01   7.852137e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   5.172861e-01   6.484063e-54 
##            model_id  model_method  feats max.nTuningRuns
## 1 MFO.myMFO_classfr myMFO_classfr .rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.372                 0.002         0.5
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.5               0            0.759
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.7312521             0.7852137             0         0.5
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5               0            0.759
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.7312521             0.7852137             0
```

```r
if (glb_is_classification)
    # "random" model - only for classification; none needed for regression since it is same as MFO
    ret_lst <- myfit_mdl(model_id="Random", model_method="myrandom_classfr",
                            model_type=glb_model_type,                         
                            indep_vars_vctr=".rnorm",
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## [1] "fitting model: Random.myrandom_classfr"
## [1] "    indep_vars: .rnorm"
## Fitting parameter = none on full training set
##             Length Class      Mode     
## unique.vals 2      factor     numeric  
## unique.prob 2      table      numeric  
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   2      -none-     character
## [1] "in Random.Classifier$prob"
```

![](USCensus_Earnings_files/figure-html/fit.models_0-1.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                              0
## 2            Y                                              0
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            759
## 2                                            241
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                              0
## 2            Y                                              0
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            759
## 2                                            241
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                              0
## 2            Y                                              0
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            759
## 2                                            241
##           Reference
## Prediction   N   Y
##          N 578 183
##          Y 181  58
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            578
## 2            Y                                            183
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            181
## 2                                             58
##           Reference
## Prediction   N   Y
##          N 578 183
##          Y 181  58
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            578
## 2            Y                                            183
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            181
## 2                                             58
##           Reference
## Prediction   N   Y
##          N 578 183
##          Y 181  58
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            578
## 2            Y                                            183
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            181
## 2                                             58
##           Reference
## Prediction   N   Y
##          N 578 183
##          Y 181  58
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            578
## 2            Y                                            183
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            181
## 2                                             58
##           Reference
## Prediction   N   Y
##          N 578 183
##          Y 181  58
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            578
## 2            Y                                            183
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            181
## 2                                             58
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            759
## 2            Y                                            241
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                              0
## 2                                              0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            759
## 2            Y                                            241
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                              0
## 2                                              0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            759
## 2            Y                                            241
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                              0
## 2                                              0
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.3883965
## 3        0.2 0.3883965
## 4        0.3 0.2416667
## 5        0.4 0.2416667
## 6        0.5 0.2416667
## 7        0.6 0.2416667
## 8        0.7 0.2416667
## 9        0.8 0.0000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](USCensus_Earnings_files/figure-html/fit.models_0-2.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.Y
## 1            N                                            759
## 2            Y                                            241
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                              0
## 2            Y                                              0
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            759
## 2                                            241
##          Prediction
## Reference   N   Y
##         N   0 759
##         Y   0 241
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   2.410000e-01   0.000000e+00   2.147863e-01   2.687479e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   1.000000e+00  1.205237e-166 
## [1] "in Random.Classifier$prob"
```

![](USCensus_Earnings_files/figure-html/fit.models_0-3.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                              0
## 2            Y                                              0
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            759
## 2                                            241
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                              0
## 2            Y                                              0
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            759
## 2                                            241
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                              0
## 2            Y                                              0
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            759
## 2                                            241
##           Reference
## Prediction   N   Y
##          N 558 183
##          Y 201  58
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            558
## 2            Y                                            183
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            201
## 2                                             58
##           Reference
## Prediction   N   Y
##          N 558 183
##          Y 201  58
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            558
## 2            Y                                            183
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            201
## 2                                             58
##           Reference
## Prediction   N   Y
##          N 558 183
##          Y 201  58
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            558
## 2            Y                                            183
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            201
## 2                                             58
##           Reference
## Prediction   N   Y
##          N 558 183
##          Y 201  58
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            558
## 2            Y                                            183
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            201
## 2                                             58
##           Reference
## Prediction   N   Y
##          N 558 183
##          Y 201  58
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            558
## 2            Y                                            183
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            201
## 2                                             58
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            759
## 2            Y                                            241
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                              0
## 2                                              0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            759
## 2            Y                                            241
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                              0
## 2                                              0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                            759
## 2            Y                                            241
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                              0
## 2                                              0
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.3883965
## 3        0.2 0.3883965
## 4        0.3 0.2320000
## 5        0.4 0.2320000
## 6        0.5 0.2320000
## 7        0.6 0.2320000
## 8        0.7 0.2320000
## 9        0.8 0.0000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](USCensus_Earnings_files/figure-html/fit.models_0-4.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.Y
## 1            N                                            759
## 2            Y                                            241
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Random.myrandom_classfr.N
## 1            N                                              0
## 2            Y                                              0
##   over50k.fctr.predict.Random.myrandom_classfr.Y
## 1                                            759
## 2                                            241
##          Prediction
## Reference   N   Y
##         N   0 759
##         Y   0 241
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   2.410000e-01   0.000000e+00   2.147863e-01   2.687479e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   1.000000e+00  1.205237e-166 
##                  model_id     model_method  feats max.nTuningRuns
## 1 Random.myrandom_classfr myrandom_classfr .rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.223                 0.001   0.5010961
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.2       0.3883965            0.241
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.2147863             0.2687479             0   0.4879209
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.2       0.3883965            0.241
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.2147863             0.2687479             0
```

```r
# Max.cor.Y
#   Check impact of cv
#       rpart is not a good candidate since caret does not optimize cp (only tuning parameter of rpart) well
ret_lst <- myfit_mdl(model_id="Max.cor.Y.cv.0", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## [1] "fitting model: Max.cor.Y.cv.0.rpart"
## [1] "    indep_vars: age"
```

```
## Loading required package: rpart
```

```
## Fitting cp = 0.00083 on full training set
```

```
## Loading required package: rpart.plot
```

![](USCensus_Earnings_files/figure-html/fit.models_0-5.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1000 
## 
##             CP nsplit rel error
## 1 0.0008298755      0         1
## 
## Node number 1: 1000 observations
##   predicted class=N  expected loss=0.241  P(node) =1
##     class counts:   759   241
##    probabilities: 0.759 0.241 
## 
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1000 241 N (0.7590000 0.2410000) *
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1            N                                         759
## 2            Y                                         241
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1            N                                         759
## 2            Y                                         241
##   over50k.fctr.predict.Max.cor.Y.cv.0.rpart.Y
## 1                                           0
## 2                                           0
##          Prediction
## Reference   N   Y
##         N 759   0
##         Y 241   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.590000e-01   0.000000e+00   7.312521e-01   7.852137e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   5.172861e-01   6.484063e-54 
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1            N                                         759
## 2            Y                                         241
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1            N                                         759
## 2            Y                                         241
##   over50k.fctr.predict.Max.cor.Y.cv.0.rpart.Y
## 1                                           0
## 2                                           0
##          Prediction
## Reference   N   Y
##         N 759   0
##         Y 241   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.590000e-01   0.000000e+00   7.312521e-01   7.852137e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   5.172861e-01   6.484063e-54 
##               model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.cv.0.rpart        rpart   age               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.779                  0.02         0.5
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.5               0            0.759
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.7312521             0.7852137             0         0.5
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5               0            0.759
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.7312521             0.7852137             0
```

```r
ret_lst <- myfit_mdl(model_id="Max.cor.Y.cv.0.cp.0", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=0, tune_models_df=data.frame(parameter="cp", min=0.0, max=0.0, by=0.1))
```

```
## [1] "fitting model: Max.cor.Y.cv.0.cp.0.rpart"
## [1] "    indep_vars: age"
## Fitting cp = 0 on full training set
```

![](USCensus_Earnings_files/figure-html/fit.models_0-6.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1000 
## 
##             CP nsplit rel error
## 1 0.0008298755      0 1.0000000
## 2 0.0000000000      5 0.9958506
## 
## Variable importance
## age 
## 100 
## 
## Node number 1: 1000 observations,    complexity param=0.0008298755
##   predicted class=N  expected loss=0.241  P(node) =1
##     class counts:   759   241
##    probabilities: 0.759 0.241 
##   left son=2 (256 obs) right son=3 (744 obs)
##   Primary splits:
##       age < 28.5 to the left,  improve=33.75374, (0 missing)
## 
## Node number 2: 256 observations
##   predicted class=N  expected loss=0.01953125  P(node) =0.256
##     class counts:   251     5
##    probabilities: 0.980 0.020 
## 
## Node number 3: 744 observations,    complexity param=0.0008298755
##   predicted class=N  expected loss=0.3172043  P(node) =0.744
##     class counts:   508   236
##    probabilities: 0.683 0.317 
##   left son=6 (265 obs) right son=7 (479 obs)
##   Primary splits:
##       age < 38.5 to the left,  improve=5.198796, (0 missing)
## 
## Node number 6: 265 observations
##   predicted class=N  expected loss=0.2377358  P(node) =0.265
##     class counts:   202    63
##    probabilities: 0.762 0.238 
## 
## Node number 7: 479 observations,    complexity param=0.0008298755
##   predicted class=N  expected loss=0.3611691  P(node) =0.479
##     class counts:   306   173
##    probabilities: 0.639 0.361 
##   left son=14 (129 obs) right son=15 (350 obs)
##   Primary splits:
##       age < 55.5 to the right, improve=2.379942, (0 missing)
## 
## Node number 14: 129 observations
##   predicted class=N  expected loss=0.2790698  P(node) =0.129
##     class counts:    93    36
##    probabilities: 0.721 0.279 
## 
## Node number 15: 350 observations,    complexity param=0.0008298755
##   predicted class=N  expected loss=0.3914286  P(node) =0.35
##     class counts:   213   137
##    probabilities: 0.609 0.391 
##   left son=30 (230 obs) right son=31 (120 obs)
##   Primary splits:
##       age < 47.5 to the left,  improve=0.9217598, (0 missing)
## 
## Node number 30: 230 observations
##   predicted class=N  expected loss=0.3652174  P(node) =0.23
##     class counts:   146    84
##    probabilities: 0.635 0.365 
## 
## Node number 31: 120 observations,    complexity param=0.0008298755
##   predicted class=N  expected loss=0.4416667  P(node) =0.12
##     class counts:    67    53
##    probabilities: 0.558 0.442 
##   left son=62 (83 obs) right son=63 (37 obs)
##   Primary splits:
##       age < 49.5 to the right, improve=0.5522685, (0 missing)
## 
## Node number 62: 83 observations
##   predicted class=N  expected loss=0.4096386  P(node) =0.083
##     class counts:    49    34
##    probabilities: 0.590 0.410 
## 
## Node number 63: 37 observations
##   predicted class=Y  expected loss=0.4864865  P(node) =0.037
##     class counts:    18    19
##    probabilities: 0.486 0.514 
## 
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1000 241 N (0.75900000 0.24100000)  
##    2) age< 28.5 256   5 N (0.98046875 0.01953125) *
##    3) age>=28.5 744 236 N (0.68279570 0.31720430)  
##      6) age< 38.5 265  63 N (0.76226415 0.23773585) *
##      7) age>=38.5 479 173 N (0.63883090 0.36116910)  
##       14) age>=55.5 129  36 N (0.72093023 0.27906977) *
##       15) age< 55.5 350 137 N (0.60857143 0.39142857)  
##         30) age< 47.5 230  84 N (0.63478261 0.36521739) *
##         31) age>=47.5 120  53 N (0.55833333 0.44166667)  
##           62) age>=49.5 83  34 N (0.59036145 0.40963855) *
##           63) age< 49.5 37  18 Y (0.48648649 0.51351351) *
```

![](USCensus_Earnings_files/figure-html/fit.models_0-7.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                                0
## 2            Y                                                0
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              759
## 2                                              241
##           Reference
## Prediction   N   Y
##          N 251   5
##          Y 508 236
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              251
## 2            Y                                                5
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              508
## 2                                              236
##           Reference
## Prediction   N   Y
##          N 251   5
##          Y 508 236
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              251
## 2            Y                                                5
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              508
## 2                                              236
##           Reference
## Prediction   N   Y
##          N 546 104
##          Y 213 137
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              546
## 2            Y                                              104
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              213
## 2                                              137
##           Reference
## Prediction   N   Y
##          N 692 188
##          Y  67  53
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              692
## 2            Y                                              188
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               67
## 2                                               53
##           Reference
## Prediction   N   Y
##          N 741 222
##          Y  18  19
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              741
## 2            Y                                              222
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               18
## 2                                               19
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              759
## 2            Y                                              241
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              759
## 2            Y                                              241
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              759
## 2            Y                                              241
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              759
## 2            Y                                              241
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              759
## 2            Y                                              241
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.4791878
## 3        0.2 0.4791878
## 4        0.3 0.4636210
## 5        0.4 0.2936288
## 6        0.5 0.1366906
## 7        0.6 0.0000000
## 8        0.7 0.0000000
## 9        0.8 0.0000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](USCensus_Earnings_files/figure-html/fit.models_0-8.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              251
## 2            Y                                                5
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              508
## 2                                              236
##           Reference
## Prediction   N   Y
##          N 251   5
##          Y 508 236
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              251
## 2            Y                                                5
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              508
## 2                                              236
##          Prediction
## Reference   N   Y
##         N 251 508
##         Y   5 236
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   4.870000e-01   1.810240e-01   4.555980e-01   5.184787e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   1.000000e+00  7.671886e-109
```

![](USCensus_Earnings_files/figure-html/fit.models_0-9.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                                0
## 2            Y                                                0
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              759
## 2                                              241
##           Reference
## Prediction   N   Y
##          N 274  14
##          Y 485 227
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              274
## 2            Y                                               14
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              485
## 2                                              227
##           Reference
## Prediction   N   Y
##          N 274  14
##          Y 485 227
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              274
## 2            Y                                               14
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              485
## 2                                              227
##           Reference
## Prediction   N   Y
##          N 549 124
##          Y 210 117
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              549
## 2            Y                                              124
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              210
## 2                                              117
##           Reference
## Prediction   N   Y
##          N 687 198
##          Y  72  43
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              687
## 2            Y                                              198
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               72
## 2                                               43
##           Reference
## Prediction   N   Y
##          N 736 232
##          Y  23   9
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              736
## 2            Y                                              232
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               23
## 2                                                9
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              759
## 2            Y                                              241
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              759
## 2            Y                                              241
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              759
## 2            Y                                              241
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              759
## 2            Y                                              241
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              759
## 2            Y                                              241
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##    threshold    f.score
## 1        0.0 0.38839645
## 2        0.1 0.47639035
## 3        0.2 0.47639035
## 4        0.3 0.41197183
## 5        0.4 0.24157303
## 6        0.5 0.06593407
## 7        0.6 0.00000000
## 8        0.7 0.00000000
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

![](USCensus_Earnings_files/figure-html/fit.models_0-10.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              274
## 2            Y                                               14
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              485
## 2                                              227
##           Reference
## Prediction   N   Y
##          N 274  14
##          Y 485 227
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1            N                                              274
## 2            Y                                               14
##   over50k.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              485
## 2                                              227
##          Prediction
## Reference   N   Y
##         N 274 485
##         Y  14 227
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   5.010000e-01   1.817204e-01   4.695463e-01   5.324478e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   1.000000e+00   2.818448e-98 
##                    model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.cv.0.cp.0.rpart        rpart   age               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.668                 0.018   0.7183097
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.2       0.4791878            0.487
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.455598             0.5184787      0.181024   0.6837507
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.2       0.4763903            0.501
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.4695463             0.5324478     0.1817204
```

```r
if (glb_is_regression || glb_is_binomial) # For multinomials this model will be run next by default
ret_lst <- myfit_mdl(model_id="Max.cor.Y", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Max.cor.Y.rpart"
## [1] "    indep_vars: age"
## + Fold1: cp=0 
## - Fold1: cp=0 
## + Fold2: cp=0 
## - Fold2: cp=0 
## + Fold3: cp=0 
## - Fold3: cp=0 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.00083 on full training set
```

```
## Warning in myfit_mdl(model_id = "Max.cor.Y", model_method = "rpart",
## model_type = glb_model_type, : model's bestTune found at an extreme of
## tuneGrid for parameter: cp
```

![](USCensus_Earnings_files/figure-html/fit.models_0-11.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-12.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1000 
## 
##             CP nsplit rel error
## 1 0.0008298755      0         1
## 
## Node number 1: 1000 observations
##   predicted class=N  expected loss=0.241  P(node) =1
##     class counts:   759   241
##    probabilities: 0.759 0.241 
## 
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 1000 241 N (0.7590000 0.2410000) *
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.rpart.N
## 1            N                                    759
## 2            Y                                    241
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.rpart.N
## 1            N                                    759
## 2            Y                                    241
##   over50k.fctr.predict.Max.cor.Y.rpart.Y
## 1                                      0
## 2                                      0
##          Prediction
## Reference   N   Y
##         N 759   0
##         Y 241   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.590000e-01   0.000000e+00   7.312521e-01   7.852137e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   5.172861e-01   6.484063e-54 
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.rpart.N
## 1            N                                    759
## 2            Y                                    241
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.rpart.N
## 1            N                                    759
## 2            Y                                    241
##   over50k.fctr.predict.Max.cor.Y.rpart.Y
## 1                                      0
## 2                                      0
##          Prediction
## Reference   N   Y
##         N 759   0
##         Y 241   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.590000e-01   0.000000e+00   7.312521e-01   7.852137e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   5.172861e-01   6.484063e-54 
##          model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.rpart        rpart   age               3
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.978                 0.021         0.5
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.5               0        0.7459975
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.7312521             0.7852137      0.052497         0.5
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5               0            0.759
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.7312521             0.7852137             0
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.01220642      0.02765327
```

```r
# Used to compare vs. Interactions.High.cor.Y 
ret_lst <- myfit_mdl(model_id="Max.cor.Y", 
                        model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Max.cor.Y.glm"
## [1] "    indep_vars: age"
## + Fold1: parameter=none 
## - Fold1: parameter=none 
## + Fold2: parameter=none 
## - Fold2: parameter=none 
## + Fold3: parameter=none 
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](USCensus_Earnings_files/figure-html/fit.models_0-13.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-14.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-15.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-16.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.5972  -0.7599  -0.5935  -0.4677   1.9655  
## 
## Coefficients:
##              Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -2.881280   0.250694  -11.49  < 2e-16 ***
## age          0.042547   0.005628    7.56 4.02e-14 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1104.5  on 999  degrees of freedom
## Residual deviance: 1044.1  on 998  degrees of freedom
## AIC: 1048.1
## 
## Number of Fisher Scoring iterations: 4
```

![](USCensus_Earnings_files/figure-html/fit.models_0-17.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                    0
## 2            Y                                    0
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  759
## 2                                  241
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                    0
## 2            Y                                    0
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  759
## 2                                  241
##           Reference
## Prediction   N   Y
##          N 396  54
##          Y 363 187
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  396
## 2            Y                                   54
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  363
## 2                                  187
##           Reference
## Prediction   N   Y
##          N 599 152
##          Y 160  89
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  599
## 2            Y                                  152
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  160
## 2                                   89
##           Reference
## Prediction   N   Y
##          N 690 213
##          Y  69  28
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  690
## 2            Y                                  213
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                   69
## 2                                   28
##           Reference
## Prediction   N   Y
##          N 745 234
##          Y  14   7
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  745
## 2            Y                                  234
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                   14
## 2                                    7
##           Reference
## Prediction   N   Y
##          N 757 239
##          Y   2   2
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  757
## 2            Y                                  239
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                    2
## 2                                    2
##           Reference
## Prediction   N   Y
##          N 758 241
##          Y   1   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  758
## 2            Y                                  241
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                    1
## 2                                    0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  759
## 2            Y                                  241
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                    0
## 2                                    0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  759
## 2            Y                                  241
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                    0
## 2                                    0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  759
## 2            Y                                  241
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                    0
## 2                                    0
##    threshold    f.score
## 1        0.0 0.38839645
## 2        0.1 0.38839645
## 3        0.2 0.47281922
## 4        0.3 0.36326531
## 5        0.4 0.16568047
## 6        0.5 0.05343511
## 7        0.6 0.01632653
## 8        0.7 0.00000000
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

![](USCensus_Earnings_files/figure-html/fit.models_0-18.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  396
## 2            Y                                   54
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  363
## 2                                  187
##           Reference
## Prediction   N   Y
##          N 396  54
##          Y 363 187
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  396
## 2            Y                                   54
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  363
## 2                                  187
##          Prediction
## Reference   N   Y
##         N 396 363
##         Y  54 187
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   5.830000e-01   2.070736e-01   5.517330e-01   6.137769e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   1.000000e+00   2.100846e-51
```

![](USCensus_Earnings_files/figure-html/fit.models_0-19.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                    0
## 2            Y                                    0
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  759
## 2                                  241
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                    0
## 2            Y                                    0
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  759
## 2                                  241
##           Reference
## Prediction   N   Y
##          N 411  49
##          Y 348 192
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  411
## 2            Y                                   49
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  348
## 2                                  192
##           Reference
## Prediction   N   Y
##          N 597 153
##          Y 162  88
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  597
## 2            Y                                  153
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  162
## 2                                   88
##           Reference
## Prediction   N   Y
##          N 694 207
##          Y  65  34
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  694
## 2            Y                                  207
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                   65
## 2                                   34
##           Reference
## Prediction   N   Y
##          N 737 236
##          Y  22   5
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  737
## 2            Y                                  236
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                   22
## 2                                    5
##           Reference
## Prediction   N   Y
##          N 753 241
##          Y   6   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  753
## 2            Y                                  241
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                    6
## 2                                    0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  759
## 2            Y                                  241
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                    0
## 2                                    0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  759
## 2            Y                                  241
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                    0
## 2                                    0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  759
## 2            Y                                  241
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                    0
## 2                                    0
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  759
## 2            Y                                  241
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                    0
## 2                                    0
##    threshold    f.score
## 1        0.0 0.38839645
## 2        0.1 0.38839645
## 3        0.2 0.49167734
## 4        0.3 0.35845214
## 5        0.4 0.20000000
## 6        0.5 0.03731343
## 7        0.6 0.00000000
## 8        0.7 0.00000000
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

![](USCensus_Earnings_files/figure-html/fit.models_0-20.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  411
## 2            Y                                   49
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  348
## 2                                  192
##           Reference
## Prediction   N   Y
##          N 411  49
##          Y 348 192
##   over50k.fctr over50k.fctr.predict.Max.cor.Y.glm.N
## 1            N                                  411
## 2            Y                                   49
##   over50k.fctr.predict.Max.cor.Y.glm.Y
## 1                                  348
## 2                                  192
##          Prediction
## Reference   N   Y
##         N 411 348
##         Y  49 192
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   6.030000e-01   2.375941e-01   5.719080e-01   6.334837e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   1.000000e+00   1.419192e-50 
##        model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.glm          glm   age               1
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      1.074                 0.019   0.6880368
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.2       0.4728192        0.7550095
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.551733             0.6137769    0.02934095   0.6937196
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.2       0.4916773            0.603
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1              0.571908             0.6334837     0.2375941    1048.081
##   max.AccuracySD.fit max.KappaSD.fit
## 1        0.008758591      0.03529993
```

```r
# Interactions.High.cor.Y
if (nrow(int_feats_df <- subset(glb_feats_df, (cor.low == 0) & 
                                              (exclude.as.feat == 0))) > 0) {
    # lm & glm handle interaction terms; rpart & rf do not
    #   This does not work - why ???
#     indep_vars_vctr <- ifelse(glb_is_binomial, 
#         c(max_cor_y_x_var, paste(max_cor_y_x_var, 
#                         subset(glb_feats_df, is.na(cor.low))[, "id"], sep=":")),
#         union(max_cor_y_x_var, subset(glb_feats_df, is.na(cor.low))[, "id"]))
    if (glb_is_regression || glb_is_binomial) {
        indep_vars_vctr <- 
            c(max_cor_y_x_var, paste(max_cor_y_x_var, int_feats_df[, "id"], sep=":"))       
    } else { indep_vars_vctr <- union(max_cor_y_x_var, int_feats_df[, "id"]) }
    
    ret_lst <- myfit_mdl(model_id="Interact.High.cor.y", 
                            model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                         model_type=glb_model_type,
                            indep_vars_vctr,
                            glb_rsp_var, glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                            n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)                        
}    

# Low.cor.X
ret_lst <- myfit_mdl(model_id="Low.cor.X", 
                        model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                        indep_vars_vctr=subset(glb_feats_df, cor.low == 1)[, "id"],
                         model_type=glb_model_type,                     
                        glb_rsp_var, glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Low.cor.X.glm"
## [1] "    indep_vars: age, capitalgain, hoursperweek, capitalloss, workclass.fctr, maritalstatus.fctr, race.fctr, .rnorm, education.fctr, nativecountry.fctr, occupation.fctr, relationship.fctr, sex.fctr"
## + Fold1: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold1: parameter=none 
## + Fold2: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold2: parameter=none 
## + Fold3: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning: not plotting observations with leverage one:
##   118, 160, 182, 194, 371, 422, 477, 518, 535, 638, 739, 798, 805, 808, 811, 921
```

![](USCensus_Earnings_files/figure-html/fit.models_0-21.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-22.png) 

```
## Warning: not plotting observations with leverage one:
##   118, 160, 182, 194, 371, 422, 477, 518, 535, 638, 739, 798, 805, 808, 811, 921
```

![](USCensus_Earnings_files/figure-html/fit.models_0-23.png) 

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##      Min        1Q    Median        3Q       Max  
## -2.24059  -0.43840  -0.08362   0.00000   2.62797  
## 
## Coefficients: (12 not defined because of singularities)
##                                                   Estimate Std. Error
## (Intercept)                                     -6.821e+00  1.159e+00
## age                                              3.818e-02  1.089e-02
## capitalgain                                      4.161e-04  7.105e-05
## hoursperweek                                     2.970e-02  1.051e-02
## capitalloss                                      2.194e-04  2.357e-04
## `workclass.fctr Self-emp-not-inc`                1.610e-01  7.478e-01
## `workclass.fctr Private`                         1.125e+00  6.789e-01
## `workclass.fctr Federal-gov`                     2.904e+00  9.072e-01
## `workclass.fctr Local-gov`                       3.428e-01  7.691e-01
## `workclass.fctr ?`                               6.723e-01  9.573e-01
## `workclass.fctr Self-emp-inc`                    1.042e+00  8.233e-01
## `workclass.fctr Without-pay`                    -1.725e+01  1.075e+04
## `workclass.fctr Never-worked`                           NA         NA
## `maritalstatus.fctr Married-civ-spouse`         -1.878e+00  7.345e+03
## `maritalstatus.fctr Divorced`                    4.708e-01  5.824e-01
## `maritalstatus.fctr Married-spouse-absent`      -1.558e+01  2.722e+03
## `maritalstatus.fctr Separated`                   4.013e-01  8.565e-01
## `maritalstatus.fctr Married-AF-spouse`          -1.751e+01  1.302e+04
## `maritalstatus.fctr Widowed`                     1.088e+00  8.744e-01
## `race.fctr Black`                                4.090e-01  4.460e-01
## `race.fctr Asian-Pac-Islander`                   1.380e+00  1.491e+00
## `race.fctr Amer-Indian-Eskimo`                   5.347e-01  1.206e+00
## `race.fctr Other`                                1.267e+00  1.487e+00
## .rnorm                                          -3.095e-02  1.171e-01
## `education.fctr HS-grad`                        -1.559e+00  3.436e-01
## `education.fctr 11th`                           -1.815e+01  2.222e+03
## `education.fctr Masters`                        -2.940e-02  4.506e-01
## `education.fctr 9th`                            -1.666e+00  1.406e+00
## `education.fctr Some-college`                   -8.444e-01  3.563e-01
## `education.fctr Assoc-acdm`                     -6.287e-01  7.288e-01
## `education.fctr 7th-8th`                        -2.846e+00  1.162e+00
## `education.fctr Doctorate`                       1.534e+00  9.430e-01
## `education.fctr Assoc-voc`                      -8.237e-01  5.183e-01
## `education.fctr Prof-school`                     1.781e+00  9.115e-01
## `education.fctr 5th-6th`                        -2.305e+00  1.545e+00
## `education.fctr 10th`                           -1.951e+00  7.429e-01
## `education.fctr 1st-4th`                        -2.408e+00  5.929e+03
## `education.fctr Preschool`                      -2.289e+00  1.097e+04
## `education.fctr 12th`                           -1.821e+01  2.665e+03
## `nativecountry.fctr Cuba`                        2.854e+00  1.496e+00
## `nativecountry.fctr Jamaica`                     1.769e-01  1.363e+00
## `nativecountry.fctr India`                      -1.501e+00  1.781e+00
## `nativecountry.fctr Mexico`                     -1.704e+01  2.148e+03
## `nativecountry.fctr South`                       3.292e-01  1.099e+04
## `nativecountry.fctr Puerto-Rico`                -1.631e+01  7.046e+03
## `nativecountry.fctr Honduras`                           NA         NA
## `nativecountry.fctr England`                     9.965e-01  1.034e+00
## `nativecountry.fctr Canada`                      2.268e+00  2.148e+00
## `nativecountry.fctr Germany`                    -1.790e+01  4.090e+03
## `nativecountry.fctr Iran`                               NA         NA
## `nativecountry.fctr Philippines`                -1.854e+00  2.302e+00
## `nativecountry.fctr Italy`                      -1.582e+01  1.075e+04
## `nativecountry.fctr Poland`                     -1.698e+01  1.075e+04
## `nativecountry.fctr Columbia`                   -1.694e+01  4.154e+03
## `nativecountry.fctr Cambodia`                           NA         NA
## `nativecountry.fctr Thailand`                           NA         NA
## `nativecountry.fctr Ecuador`                    -1.120e+01  1.075e+04
## `nativecountry.fctr Laos`                        5.683e-01  2.207e+00
## `nativecountry.fctr Taiwan`                      1.847e+01  7.390e+03
## `nativecountry.fctr Haiti`                      -1.274e+01  1.075e+04
## `nativecountry.fctr Portugal`                   -1.658e+01  1.075e+04
## `nativecountry.fctr Dominican-Republic`                 NA         NA
## `nativecountry.fctr El-Salvador`                -1.699e+01  3.705e+03
## `nativecountry.fctr France`                     -2.093e+01  6.381e+03
## `nativecountry.fctr Guatemala`                   1.553e+01  1.175e+04
## `nativecountry.fctr China`                      -2.950e+00  2.128e+00
## `nativecountry.fctr Japan`                      -1.679e+01  1.075e+04
## `nativecountry.fctr Yugoslavia`                         NA         NA
## `nativecountry.fctr Peru`                       -1.283e+01  1.075e+04
## `nativecountry.fctr Outlying-US(Guam-USVI-etc)`         NA         NA
## `nativecountry.fctr Scotland`                   -1.651e+01  1.075e+04
## `nativecountry.fctr Trinadad&Tobago`            -1.849e+01  1.075e+04
## `nativecountry.fctr Greece`                     -1.473e+01  1.075e+04
## `nativecountry.fctr Nicaragua`                  -1.745e+01  1.075e+04
## `nativecountry.fctr Vietnam`                    -1.078e+00  7.665e+03
## `nativecountry.fctr Hong`                       -1.907e+01  1.075e+04
## `nativecountry.fctr Ireland`                    -1.925e+01  1.075e+04
## `nativecountry.fctr Hungary`                            NA         NA
## `nativecountry.fctr Holand-Netherlands`                 NA         NA
## `occupation.fctr Exec-managerial`                1.304e+00  4.950e-01
## `occupation.fctr Handlers-cleaners`              2.694e-02  8.208e-01
## `occupation.fctr Prof-specialty`                 1.153e+00  5.194e-01
## `occupation.fctr Other-service`                 -2.126e+00  1.190e+00
## `occupation.fctr Sales`                          1.069e+00  5.227e-01
## `occupation.fctr Transport-moving`               1.505e-01  6.237e-01
## `occupation.fctr Farming-fishing`               -1.718e+01  1.838e+03
## `occupation.fctr Machine-op-inspct`              3.740e-01  6.315e-01
## `occupation.fctr Tech-support`                   1.914e+00  7.198e-01
## `occupation.fctr Craft-repair`                   4.127e-01  5.182e-01
## `occupation.fctr ?`                                     NA         NA
## `occupation.fctr Protective-serv`               -1.078e+00  1.275e+00
## `occupation.fctr Armed-Forces`                          NA         NA
## `occupation.fctr Priv-house-serv`               -1.410e+01  4.167e+03
## `relationship.fctr Husband`                      4.541e+00  7.345e+03
## `relationship.fctr Wife`                         6.133e+00  7.345e+03
## `relationship.fctr Own-child`                   -9.141e-01  1.018e+00
## `relationship.fctr Unmarried`                    1.153e+00  5.278e-01
## `relationship.fctr Other-relative`              -1.367e+01  2.263e+03
## `sex.fctr Female`                               -1.083e+00  4.967e-01
##                                                 z value Pr(>|z|)    
## (Intercept)                                      -5.886 3.95e-09 ***
## age                                               3.507 0.000454 ***
## capitalgain                                       5.857 4.71e-09 ***
## hoursperweek                                      2.828 0.004691 ** 
## capitalloss                                       0.931 0.351975    
## `workclass.fctr Self-emp-not-inc`                 0.215 0.829567    
## `workclass.fctr Private`                          1.657 0.097516 .  
## `workclass.fctr Federal-gov`                      3.201 0.001370 ** 
## `workclass.fctr Local-gov`                        0.446 0.655834    
## `workclass.fctr ?`                                0.702 0.482461    
## `workclass.fctr Self-emp-inc`                     1.265 0.205836    
## `workclass.fctr Without-pay`                     -0.002 0.998720    
## `workclass.fctr Never-worked`                        NA       NA    
## `maritalstatus.fctr Married-civ-spouse`           0.000 0.999796    
## `maritalstatus.fctr Divorced`                     0.808 0.418880    
## `maritalstatus.fctr Married-spouse-absent`       -0.006 0.995434    
## `maritalstatus.fctr Separated`                    0.469 0.639391    
## `maritalstatus.fctr Married-AF-spouse`           -0.001 0.998927    
## `maritalstatus.fctr Widowed`                      1.244 0.213532    
## `race.fctr Black`                                 0.917 0.359078    
## `race.fctr Asian-Pac-Islander`                    0.925 0.354892    
## `race.fctr Amer-Indian-Eskimo`                    0.443 0.657450    
## `race.fctr Other`                                 0.852 0.394420    
## .rnorm                                           -0.264 0.791562    
## `education.fctr HS-grad`                         -4.537 5.71e-06 ***
## `education.fctr 11th`                            -0.008 0.993482    
## `education.fctr Masters`                         -0.065 0.947977    
## `education.fctr 9th`                             -1.185 0.236034    
## `education.fctr Some-college`                    -2.370 0.017790 *  
## `education.fctr Assoc-acdm`                      -0.863 0.388343    
## `education.fctr 7th-8th`                         -2.450 0.014284 *  
## `education.fctr Doctorate`                        1.627 0.103707    
## `education.fctr Assoc-voc`                       -1.589 0.112020    
## `education.fctr Prof-school`                      1.954 0.050691 .  
## `education.fctr 5th-6th`                         -1.492 0.135688    
## `education.fctr 10th`                            -2.626 0.008635 ** 
## `education.fctr 1st-4th`                          0.000 0.999676    
## `education.fctr Preschool`                        0.000 0.999833    
## `education.fctr 12th`                            -0.007 0.994550    
## `nativecountry.fctr Cuba`                         1.908 0.056379 .  
## `nativecountry.fctr Jamaica`                      0.130 0.896757    
## `nativecountry.fctr India`                       -0.843 0.399391    
## `nativecountry.fctr Mexico`                      -0.008 0.993670    
## `nativecountry.fctr South`                        0.000 0.999976    
## `nativecountry.fctr Puerto-Rico`                 -0.002 0.998153    
## `nativecountry.fctr Honduras`                        NA       NA    
## `nativecountry.fctr England`                      0.964 0.334943    
## `nativecountry.fctr Canada`                       1.056 0.291095    
## `nativecountry.fctr Germany`                     -0.004 0.996508    
## `nativecountry.fctr Iran`                            NA       NA    
## `nativecountry.fctr Philippines`                 -0.805 0.420791    
## `nativecountry.fctr Italy`                       -0.001 0.998826    
## `nativecountry.fctr Poland`                      -0.002 0.998740    
## `nativecountry.fctr Columbia`                    -0.004 0.996746    
## `nativecountry.fctr Cambodia`                        NA       NA    
## `nativecountry.fctr Thailand`                        NA       NA    
## `nativecountry.fctr Ecuador`                     -0.001 0.999169    
## `nativecountry.fctr Laos`                         0.258 0.796765    
## `nativecountry.fctr Taiwan`                       0.002 0.998006    
## `nativecountry.fctr Haiti`                       -0.001 0.999054    
## `nativecountry.fctr Portugal`                    -0.002 0.998770    
## `nativecountry.fctr Dominican-Republic`              NA       NA    
## `nativecountry.fctr El-Salvador`                 -0.005 0.996341    
## `nativecountry.fctr France`                      -0.003 0.997383    
## `nativecountry.fctr Guatemala`                    0.001 0.998945    
## `nativecountry.fctr China`                       -1.386 0.165625    
## `nativecountry.fctr Japan`                       -0.002 0.998754    
## `nativecountry.fctr Yugoslavia`                      NA       NA    
## `nativecountry.fctr Peru`                        -0.001 0.999048    
## `nativecountry.fctr Outlying-US(Guam-USVI-etc)`      NA       NA    
## `nativecountry.fctr Scotland`                    -0.002 0.998775    
## `nativecountry.fctr Trinadad&Tobago`             -0.002 0.998628    
## `nativecountry.fctr Greece`                      -0.001 0.998907    
## `nativecountry.fctr Nicaragua`                   -0.002 0.998705    
## `nativecountry.fctr Vietnam`                      0.000 0.999888    
## `nativecountry.fctr Hong`                        -0.002 0.998585    
## `nativecountry.fctr Ireland`                     -0.002 0.998572    
## `nativecountry.fctr Hungary`                         NA       NA    
## `nativecountry.fctr Holand-Netherlands`              NA       NA    
## `occupation.fctr Exec-managerial`                 2.634 0.008448 ** 
## `occupation.fctr Handlers-cleaners`               0.033 0.973816    
## `occupation.fctr Prof-specialty`                  2.220 0.026412 *  
## `occupation.fctr Other-service`                  -1.786 0.074026 .  
## `occupation.fctr Sales`                           2.045 0.040865 *  
## `occupation.fctr Transport-moving`                0.241 0.809342    
## `occupation.fctr Farming-fishing`                -0.009 0.992542    
## `occupation.fctr Machine-op-inspct`               0.592 0.553673    
## `occupation.fctr Tech-support`                    2.658 0.007851 ** 
## `occupation.fctr Craft-repair`                    0.796 0.425807    
## `occupation.fctr ?`                                  NA       NA    
## `occupation.fctr Protective-serv`                -0.846 0.397550    
## `occupation.fctr Armed-Forces`                       NA       NA    
## `occupation.fctr Priv-house-serv`                -0.003 0.997300    
## `relationship.fctr Husband`                       0.001 0.999507    
## `relationship.fctr Wife`                          0.001 0.999334    
## `relationship.fctr Own-child`                    -0.898 0.369177    
## `relationship.fctr Unmarried`                     2.183 0.029005 *  
## `relationship.fctr Other-relative`               -0.006 0.995180    
## `sex.fctr Female`                                -2.180 0.029227 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1104.46  on 999  degrees of freedom
## Residual deviance:  545.16  on 913  degrees of freedom
## AIC: 719.16
## 
## Number of Fisher Scoring iterations: 18
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](USCensus_Earnings_files/figure-html/fit.models_0-24.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-25.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                    0
## 2            Y                                    0
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                  759
## 2                                  241
##           Reference
## Prediction   N   Y
##          N 523   9
##          Y 236 232
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  523
## 2            Y                                    9
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                  236
## 2                                  232
##           Reference
## Prediction   N   Y
##          N 599  25
##          Y 160 216
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  599
## 2            Y                                   25
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                  160
## 2                                  216
##           Reference
## Prediction   N   Y
##          N 656  41
##          Y 103 200
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  656
## 2            Y                                   41
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                  103
## 2                                  200
##           Reference
## Prediction   N   Y
##          N 684  57
##          Y  75 184
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  684
## 2            Y                                   57
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   75
## 2                                  184
##           Reference
## Prediction   N   Y
##          N 701  82
##          Y  58 159
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  701
## 2            Y                                   82
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   58
## 2                                  159
##           Reference
## Prediction   N   Y
##          N 728  96
##          Y  31 145
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  728
## 2            Y                                   96
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   31
## 2                                  145
##           Reference
## Prediction   N   Y
##          N 741 123
##          Y  18 118
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  741
## 2            Y                                  123
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   18
## 2                                  118
##           Reference
## Prediction   N   Y
##          N 753 158
##          Y   6  83
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  753
## 2            Y                                  158
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                    6
## 2                                   83
##           Reference
## Prediction   N   Y
##          N 758 183
##          Y   1  58
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  758
## 2            Y                                  183
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                    1
## 2                                   58
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  759
## 2            Y                                  241
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                    0
## 2                                    0
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.6544429
## 3        0.2 0.7001621
## 4        0.3 0.7352941
## 5        0.4 0.7360000
## 6        0.5 0.6943231
## 7        0.6 0.6954436
## 8        0.7 0.6259947
## 9        0.8 0.5030303
## 10       0.9 0.3866667
## 11       1.0 0.0000000
```

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  684
## 2            Y                                   57
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   75
## 2                                  184
##           Reference
## Prediction   N   Y
##          N 684  57
##          Y  75 184
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  684
## 2            Y                                   57
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   75
## 2                                  184
##          Prediction
## Reference   N   Y
##         N 684  75
##         Y  57 184
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.680000e-01   6.481520e-01   8.454346e-01   8.883720e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   5.968845e-18   1.389640e-01
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](USCensus_Earnings_files/figure-html/fit.models_0-26.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-27.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                    0
## 2            Y                                    0
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                  759
## 2                                  241
##           Reference
## Prediction   N   Y
##          N 519  36
##          Y 240 205
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  519
## 2            Y                                   36
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                  240
## 2                                  205
##           Reference
## Prediction   N   Y
##          N 601  56
##          Y 158 185
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  601
## 2            Y                                   56
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                  158
## 2                                  185
##           Reference
## Prediction   N   Y
##          N 641  75
##          Y 118 166
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  641
## 2            Y                                   75
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                  118
## 2                                  166
##           Reference
## Prediction   N   Y
##          N 671  86
##          Y  88 155
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  671
## 2            Y                                   86
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   88
## 2                                  155
##           Reference
## Prediction   N   Y
##          N 694  99
##          Y  65 142
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  694
## 2            Y                                   99
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   65
## 2                                  142
##           Reference
## Prediction   N   Y
##          N 715 116
##          Y  44 125
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  715
## 2            Y                                  116
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   44
## 2                                  125
##           Reference
## Prediction   N   Y
##          N 730 139
##          Y  29 102
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  730
## 2            Y                                  139
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   29
## 2                                  102
##           Reference
## Prediction   N   Y
##          N 740 167
##          Y  19  74
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  740
## 2            Y                                  167
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   19
## 2                                   74
##           Reference
## Prediction   N   Y
##          N 747 192
##          Y  12  49
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  747
## 2            Y                                  192
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   12
## 2                                   49
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  759
## 2            Y                                  241
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                    0
## 2                                    0
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.5976676
## 3        0.2 0.6335616
## 4        0.3 0.6323810
## 5        0.4 0.6404959
## 6        0.5 0.6339286
## 7        0.6 0.6097561
## 8        0.7 0.5483871
## 9        0.8 0.4431138
## 10       0.9 0.3245033
## 11       1.0 0.0000000
```

![](USCensus_Earnings_files/figure-html/fit.models_0-28.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  671
## 2            Y                                   86
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   88
## 2                                  155
##           Reference
## Prediction   N   Y
##          N 671  86
##          Y  88 155
##   over50k.fctr over50k.fctr.predict.Low.cor.X.glm.N
## 1            N                                  671
## 2            Y                                   86
##   over50k.fctr.predict.Low.cor.X.glm.Y
## 1                                   88
## 2                                  155
##          Prediction
## Reference   N   Y
##         N 671  88
##         Y  86 155
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.260000e-01   5.257227e-01   8.010531e-01   8.490099e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   1.789552e-07   9.395704e-01 
##        model_id model_method
## 1 Low.cor.X.glm          glm
##                                                                                                                                                                                  feats
## 1 age, capitalgain, hoursperweek, capitalloss, workclass.fctr, maritalstatus.fctr, race.fctr, .rnorm, education.fctr, nativecountry.fctr, occupation.fctr, relationship.fctr, sex.fctr
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                       2.83                 0.499
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.9325658                    0.4           0.736        0.8299947
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8454346              0.888372     0.5291816    0.835337
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.4       0.6404959            0.826
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.8010531             0.8490099     0.5257227    719.1617
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.05275524       0.1292318
```

```r
# User specified
for (method in glb_models_method_vctr) {
    print(sprintf("iterating over method:%s", method))

    # All X that is not user excluded
    indep_vars_vctr <- setdiff(names(glb_trnent_df), 
        union(glb_rsp_var, glb_exclude_vars_as_features))
    
    # easier to exclude features
#     indep_vars_vctr <- setdiff(names(glb_trnent_df), 
#         union(union(glb_rsp_var, glb_exclude_vars_as_features), 
#               c("<feat1_name>", "<feat2_name>")))
    
    # easier to include features
#     indep_vars_vctr <- c("<feat1_name>", "<feat2_name>")

    # User specified bivariate models
#     indep_vars_vctr_lst <- list()
#     for (feat in setdiff(names(glb_trnent_df), 
#                          union(glb_rsp_var, glb_exclude_vars_as_features)))
#         indep_vars_vctr_lst[["feat"]] <- feat

    # User specified combinatorial models
#     indep_vars_vctr_lst <- list()
#     combn_mtrx <- combn(c("<feat1_name>", "<feat2_name>", "<featn_name>"), 
#                           <num_feats_to_choose>)
#     for (combn_ix in 1:ncol(combn_mtrx))
#         #print(combn_mtrx[, combn_ix])
#         indep_vars_vctr_lst[[combn_ix]] <- combn_mtrx[, combn_ix]

#     glb_sel_mdl <- glb_sel_wlm_mdl <- ret_lst[["model"]]
#     rpart_sel_wlm_mdl <- rpart(reformulate(indep_vars_vctr, response=glb_rsp_var), 
#                                data=glb_trnent_df, method="class", 
#                                control=rpart.control(cp=glb_sel_wlm_mdl$bestTune$cp),
#                            parms=list(loss=glb_model_metric_terms))
#     print("rpart_sel_wlm_mdl"); prp(rpart_sel_wlm_mdl)
# 
    model_id_pfx <- "All.X";
    ret_lst <- myfit_mdl(model_id=paste0(model_id_pfx, ""), model_method=method,
                            indep_vars_vctr=indep_vars_vctr,
                            model_type=glb_model_type,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df)
    
    # Since caret does not optimize rpart well
    if (method == "rpart")
        ret_lst <- myfit_mdl(model_id=paste0(model_id_pfx, ".cp.0"), model_method=method,
                                indep_vars_vctr=indep_vars_vctr,
                                model_type=glb_model_type,
                                rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                                fit_df=glb_trnent_df, OOB_df=glb_newent_df,        
            n_cv_folds=0, tune_models_df=data.frame(parameter="cp", min=0.0, max=0.0, by=0.1))

    # Compare how rf performs w/i & w/o .rnorm
    if (method == "rf")
        ret_lst <- myfit_mdl(model_id=paste0(model_id_pfx, ".no.rnorm"), model_method=method,
                                indep_vars_vctr=setdiff(indep_vars_vctr, c(".rnorm")),
                                model_type=glb_model_type,
                                rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                                fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                    n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df)
    
    # rf is hard-coded in caret to recognize only Accuracy / Kappa evaluation metrics
    #   only for OOB in trainControl ?

#     ret_lst <- myfit_mdl_fn(model_id=paste0(model_id_pfx, ""), model_method=method,
#                             indep_vars_vctr=indep_vars_vctr,
#                             rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
#                             fit_df=glb_trnent_df, OOB_df=glb_newent_df,
#                             n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df,
#                             model_loss_mtrx=glb_model_metric_terms,
#                             model_summaryFunction=glb_model_metric_smmry,
#                             model_metric=glb_model_metric,
#                             model_metric_maximize=glb_model_metric_maximize)
}
```

```
## [1] "iterating over method:glm"
## [1] "fitting model: All.X.glm"
## [1] "    indep_vars: age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr, .rnorm"
## + Fold1: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold1: parameter=none 
## + Fold2: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold2: parameter=none 
## + Fold3: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning: not plotting observations with leverage one:
##   118, 160, 182, 194, 239, 293, 371, 422, 477, 518, 535, 638, 739, 798, 805, 808, 811, 921
```

![](USCensus_Earnings_files/figure-html/fit.models_0-29.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-30.png) 

```
## Warning: not plotting observations with leverage one:
##   118, 160, 182, 194, 239, 293, 371, 422, 477, 518, 535, 638, 739, 798, 805, 808, 811, 921
```

![](USCensus_Earnings_files/figure-html/fit.models_0-31.png) 

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##      Min        1Q    Median        3Q       Max  
## -2.24059  -0.43840  -0.08362   0.00000   2.62797  
## 
## Coefficients: (12 not defined because of singularities)
##                                                   Estimate Std. Error
## (Intercept)                                     -6.821e+00  1.159e+00
## age                                              3.818e-02  1.089e-02
## capitalgain                                      4.161e-04  7.105e-05
## capitalloss                                      2.194e-04  2.357e-04
## hoursperweek                                     2.970e-02  1.051e-02
## `workclass.fctr Self-emp-not-inc`                1.610e-01  7.478e-01
## `workclass.fctr Private`                         1.125e+00  6.789e-01
## `workclass.fctr Federal-gov`                     2.904e+00  9.072e-01
## `workclass.fctr Local-gov`                       3.428e-01  7.691e-01
## `workclass.fctr ?`                               6.723e-01  9.573e-01
## `workclass.fctr Self-emp-inc`                    1.042e+00  8.233e-01
## `workclass.fctr Without-pay`                    -1.725e+01  1.075e+04
## `workclass.fctr Never-worked`                           NA         NA
## `education.fctr HS-grad`                        -1.559e+00  3.436e-01
## `education.fctr 11th`                           -1.815e+01  2.222e+03
## `education.fctr Masters`                        -2.940e-02  4.506e-01
## `education.fctr 9th`                            -1.666e+00  1.406e+00
## `education.fctr Some-college`                   -8.444e-01  3.563e-01
## `education.fctr Assoc-acdm`                     -6.287e-01  7.288e-01
## `education.fctr 7th-8th`                        -2.846e+00  1.162e+00
## `education.fctr Doctorate`                       1.534e+00  9.430e-01
## `education.fctr Assoc-voc`                      -8.237e-01  5.183e-01
## `education.fctr Prof-school`                     1.781e+00  9.115e-01
## `education.fctr 5th-6th`                        -2.305e+00  1.545e+00
## `education.fctr 10th`                           -1.951e+00  7.429e-01
## `education.fctr 1st-4th`                        -2.408e+00  5.929e+03
## `education.fctr Preschool`                      -2.289e+00  1.097e+04
## `education.fctr 12th`                           -1.821e+01  2.665e+03
## `maritalstatus.fctr Married-civ-spouse`         -1.878e+00  7.345e+03
## `maritalstatus.fctr Divorced`                    4.708e-01  5.824e-01
## `maritalstatus.fctr Married-spouse-absent`      -1.558e+01  2.722e+03
## `maritalstatus.fctr Separated`                   4.013e-01  8.565e-01
## `maritalstatus.fctr Married-AF-spouse`          -1.751e+01  1.302e+04
## `maritalstatus.fctr Widowed`                     1.088e+00  8.744e-01
## `occupation.fctr Exec-managerial`                1.304e+00  4.950e-01
## `occupation.fctr Handlers-cleaners`              2.694e-02  8.208e-01
## `occupation.fctr Prof-specialty`                 1.153e+00  5.194e-01
## `occupation.fctr Other-service`                 -2.126e+00  1.190e+00
## `occupation.fctr Sales`                          1.069e+00  5.227e-01
## `occupation.fctr Transport-moving`               1.505e-01  6.237e-01
## `occupation.fctr Farming-fishing`               -1.718e+01  1.838e+03
## `occupation.fctr Machine-op-inspct`              3.740e-01  6.315e-01
## `occupation.fctr Tech-support`                   1.914e+00  7.198e-01
## `occupation.fctr Craft-repair`                   4.127e-01  5.182e-01
## `occupation.fctr ?`                                     NA         NA
## `occupation.fctr Protective-serv`               -1.078e+00  1.275e+00
## `occupation.fctr Armed-Forces`                          NA         NA
## `occupation.fctr Priv-house-serv`               -1.410e+01  4.167e+03
## `relationship.fctr Husband`                      4.541e+00  7.345e+03
## `relationship.fctr Wife`                         6.133e+00  7.345e+03
## `relationship.fctr Own-child`                   -9.141e-01  1.018e+00
## `relationship.fctr Unmarried`                    1.153e+00  5.278e-01
## `relationship.fctr Other-relative`              -1.367e+01  2.263e+03
## `race.fctr Black`                                4.090e-01  4.460e-01
## `race.fctr Asian-Pac-Islander`                   1.380e+00  1.491e+00
## `race.fctr Amer-Indian-Eskimo`                   5.347e-01  1.206e+00
## `race.fctr Other`                                1.267e+00  1.487e+00
## `sex.fctr Female`                               -1.083e+00  4.967e-01
## `nativecountry.fctr Cuba`                        2.854e+00  1.496e+00
## `nativecountry.fctr Jamaica`                     1.769e-01  1.363e+00
## `nativecountry.fctr India`                      -1.501e+00  1.781e+00
## `nativecountry.fctr Mexico`                     -1.704e+01  2.148e+03
## `nativecountry.fctr South`                       3.292e-01  1.099e+04
## `nativecountry.fctr Puerto-Rico`                -1.631e+01  7.046e+03
## `nativecountry.fctr Honduras`                           NA         NA
## `nativecountry.fctr England`                     9.965e-01  1.034e+00
## `nativecountry.fctr Canada`                      2.268e+00  2.148e+00
## `nativecountry.fctr Germany`                    -1.790e+01  4.090e+03
## `nativecountry.fctr Iran`                               NA         NA
## `nativecountry.fctr Philippines`                -1.854e+00  2.302e+00
## `nativecountry.fctr Italy`                      -1.582e+01  1.075e+04
## `nativecountry.fctr Poland`                     -1.698e+01  1.075e+04
## `nativecountry.fctr Columbia`                   -1.694e+01  4.154e+03
## `nativecountry.fctr Cambodia`                           NA         NA
## `nativecountry.fctr Thailand`                           NA         NA
## `nativecountry.fctr Ecuador`                    -1.120e+01  1.075e+04
## `nativecountry.fctr Laos`                        5.683e-01  2.207e+00
## `nativecountry.fctr Taiwan`                      1.847e+01  7.390e+03
## `nativecountry.fctr Haiti`                      -1.274e+01  1.075e+04
## `nativecountry.fctr Portugal`                   -1.658e+01  1.075e+04
## `nativecountry.fctr Dominican-Republic`                 NA         NA
## `nativecountry.fctr El-Salvador`                -1.699e+01  3.705e+03
## `nativecountry.fctr France`                     -2.093e+01  6.381e+03
## `nativecountry.fctr Guatemala`                   1.553e+01  1.175e+04
## `nativecountry.fctr China`                      -2.950e+00  2.128e+00
## `nativecountry.fctr Japan`                      -1.679e+01  1.075e+04
## `nativecountry.fctr Yugoslavia`                         NA         NA
## `nativecountry.fctr Peru`                       -1.283e+01  1.075e+04
## `nativecountry.fctr Outlying-US(Guam-USVI-etc)`         NA         NA
## `nativecountry.fctr Scotland`                   -1.651e+01  1.075e+04
## `nativecountry.fctr Trinadad&Tobago`            -1.849e+01  1.075e+04
## `nativecountry.fctr Greece`                     -1.473e+01  1.075e+04
## `nativecountry.fctr Nicaragua`                  -1.745e+01  1.075e+04
## `nativecountry.fctr Vietnam`                    -1.078e+00  7.665e+03
## `nativecountry.fctr Hong`                       -1.907e+01  1.075e+04
## `nativecountry.fctr Ireland`                    -1.925e+01  1.075e+04
## `nativecountry.fctr Hungary`                            NA         NA
## `nativecountry.fctr Holand-Netherlands`                 NA         NA
## .rnorm                                          -3.095e-02  1.171e-01
##                                                 z value Pr(>|z|)    
## (Intercept)                                      -5.886 3.95e-09 ***
## age                                               3.507 0.000454 ***
## capitalgain                                       5.857 4.71e-09 ***
## capitalloss                                       0.931 0.351975    
## hoursperweek                                      2.828 0.004691 ** 
## `workclass.fctr Self-emp-not-inc`                 0.215 0.829567    
## `workclass.fctr Private`                          1.657 0.097516 .  
## `workclass.fctr Federal-gov`                      3.201 0.001370 ** 
## `workclass.fctr Local-gov`                        0.446 0.655834    
## `workclass.fctr ?`                                0.702 0.482461    
## `workclass.fctr Self-emp-inc`                     1.265 0.205836    
## `workclass.fctr Without-pay`                     -0.002 0.998720    
## `workclass.fctr Never-worked`                        NA       NA    
## `education.fctr HS-grad`                         -4.537 5.71e-06 ***
## `education.fctr 11th`                            -0.008 0.993482    
## `education.fctr Masters`                         -0.065 0.947977    
## `education.fctr 9th`                             -1.185 0.236034    
## `education.fctr Some-college`                    -2.370 0.017790 *  
## `education.fctr Assoc-acdm`                      -0.863 0.388343    
## `education.fctr 7th-8th`                         -2.450 0.014284 *  
## `education.fctr Doctorate`                        1.627 0.103707    
## `education.fctr Assoc-voc`                       -1.589 0.112020    
## `education.fctr Prof-school`                      1.954 0.050691 .  
## `education.fctr 5th-6th`                         -1.492 0.135688    
## `education.fctr 10th`                            -2.626 0.008635 ** 
## `education.fctr 1st-4th`                          0.000 0.999676    
## `education.fctr Preschool`                        0.000 0.999833    
## `education.fctr 12th`                            -0.007 0.994550    
## `maritalstatus.fctr Married-civ-spouse`           0.000 0.999796    
## `maritalstatus.fctr Divorced`                     0.808 0.418880    
## `maritalstatus.fctr Married-spouse-absent`       -0.006 0.995434    
## `maritalstatus.fctr Separated`                    0.469 0.639391    
## `maritalstatus.fctr Married-AF-spouse`           -0.001 0.998927    
## `maritalstatus.fctr Widowed`                      1.244 0.213532    
## `occupation.fctr Exec-managerial`                 2.634 0.008448 ** 
## `occupation.fctr Handlers-cleaners`               0.033 0.973816    
## `occupation.fctr Prof-specialty`                  2.220 0.026412 *  
## `occupation.fctr Other-service`                  -1.786 0.074026 .  
## `occupation.fctr Sales`                           2.045 0.040865 *  
## `occupation.fctr Transport-moving`                0.241 0.809342    
## `occupation.fctr Farming-fishing`                -0.009 0.992542    
## `occupation.fctr Machine-op-inspct`               0.592 0.553673    
## `occupation.fctr Tech-support`                    2.658 0.007851 ** 
## `occupation.fctr Craft-repair`                    0.796 0.425807    
## `occupation.fctr ?`                                  NA       NA    
## `occupation.fctr Protective-serv`                -0.846 0.397550    
## `occupation.fctr Armed-Forces`                       NA       NA    
## `occupation.fctr Priv-house-serv`                -0.003 0.997300    
## `relationship.fctr Husband`                       0.001 0.999507    
## `relationship.fctr Wife`                          0.001 0.999334    
## `relationship.fctr Own-child`                    -0.898 0.369177    
## `relationship.fctr Unmarried`                     2.183 0.029005 *  
## `relationship.fctr Other-relative`               -0.006 0.995180    
## `race.fctr Black`                                 0.917 0.359078    
## `race.fctr Asian-Pac-Islander`                    0.925 0.354892    
## `race.fctr Amer-Indian-Eskimo`                    0.443 0.657450    
## `race.fctr Other`                                 0.852 0.394420    
## `sex.fctr Female`                                -2.180 0.029227 *  
## `nativecountry.fctr Cuba`                         1.908 0.056379 .  
## `nativecountry.fctr Jamaica`                      0.130 0.896757    
## `nativecountry.fctr India`                       -0.843 0.399391    
## `nativecountry.fctr Mexico`                      -0.008 0.993670    
## `nativecountry.fctr South`                        0.000 0.999976    
## `nativecountry.fctr Puerto-Rico`                 -0.002 0.998153    
## `nativecountry.fctr Honduras`                        NA       NA    
## `nativecountry.fctr England`                      0.964 0.334943    
## `nativecountry.fctr Canada`                       1.056 0.291095    
## `nativecountry.fctr Germany`                     -0.004 0.996508    
## `nativecountry.fctr Iran`                            NA       NA    
## `nativecountry.fctr Philippines`                 -0.805 0.420791    
## `nativecountry.fctr Italy`                       -0.001 0.998826    
## `nativecountry.fctr Poland`                      -0.002 0.998740    
## `nativecountry.fctr Columbia`                    -0.004 0.996746    
## `nativecountry.fctr Cambodia`                        NA       NA    
## `nativecountry.fctr Thailand`                        NA       NA    
## `nativecountry.fctr Ecuador`                     -0.001 0.999169    
## `nativecountry.fctr Laos`                         0.258 0.796765    
## `nativecountry.fctr Taiwan`                       0.002 0.998006    
## `nativecountry.fctr Haiti`                       -0.001 0.999054    
## `nativecountry.fctr Portugal`                    -0.002 0.998770    
## `nativecountry.fctr Dominican-Republic`              NA       NA    
## `nativecountry.fctr El-Salvador`                 -0.005 0.996341    
## `nativecountry.fctr France`                      -0.003 0.997383    
## `nativecountry.fctr Guatemala`                    0.001 0.998945    
## `nativecountry.fctr China`                       -1.386 0.165625    
## `nativecountry.fctr Japan`                       -0.002 0.998754    
## `nativecountry.fctr Yugoslavia`                      NA       NA    
## `nativecountry.fctr Peru`                        -0.001 0.999048    
## `nativecountry.fctr Outlying-US(Guam-USVI-etc)`      NA       NA    
## `nativecountry.fctr Scotland`                    -0.002 0.998775    
## `nativecountry.fctr Trinadad&Tobago`             -0.002 0.998628    
## `nativecountry.fctr Greece`                      -0.001 0.998907    
## `nativecountry.fctr Nicaragua`                   -0.002 0.998705    
## `nativecountry.fctr Vietnam`                      0.000 0.999888    
## `nativecountry.fctr Hong`                        -0.002 0.998585    
## `nativecountry.fctr Ireland`                     -0.002 0.998572    
## `nativecountry.fctr Hungary`                         NA       NA    
## `nativecountry.fctr Holand-Netherlands`              NA       NA    
## .rnorm                                           -0.264 0.791562    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1104.46  on 999  degrees of freedom
## Residual deviance:  545.16  on 913  degrees of freedom
## AIC: 719.16
## 
## Number of Fisher Scoring iterations: 18
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](USCensus_Earnings_files/figure-html/fit.models_0-32.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-33.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                                0
## 2            Y                                0
##   over50k.fctr.predict.All.X.glm.Y
## 1                              759
## 2                              241
##           Reference
## Prediction   N   Y
##          N 523   9
##          Y 236 232
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              523
## 2            Y                                9
##   over50k.fctr.predict.All.X.glm.Y
## 1                              236
## 2                              232
##           Reference
## Prediction   N   Y
##          N 599  25
##          Y 160 216
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              599
## 2            Y                               25
##   over50k.fctr.predict.All.X.glm.Y
## 1                              160
## 2                              216
##           Reference
## Prediction   N   Y
##          N 656  41
##          Y 103 200
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              656
## 2            Y                               41
##   over50k.fctr.predict.All.X.glm.Y
## 1                              103
## 2                              200
##           Reference
## Prediction   N   Y
##          N 684  57
##          Y  75 184
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              684
## 2            Y                               57
##   over50k.fctr.predict.All.X.glm.Y
## 1                               75
## 2                              184
##           Reference
## Prediction   N   Y
##          N 701  82
##          Y  58 159
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              701
## 2            Y                               82
##   over50k.fctr.predict.All.X.glm.Y
## 1                               58
## 2                              159
##           Reference
## Prediction   N   Y
##          N 728  96
##          Y  31 145
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              728
## 2            Y                               96
##   over50k.fctr.predict.All.X.glm.Y
## 1                               31
## 2                              145
##           Reference
## Prediction   N   Y
##          N 741 123
##          Y  18 118
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              741
## 2            Y                              123
##   over50k.fctr.predict.All.X.glm.Y
## 1                               18
## 2                              118
##           Reference
## Prediction   N   Y
##          N 753 158
##          Y   6  83
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              753
## 2            Y                              158
##   over50k.fctr.predict.All.X.glm.Y
## 1                                6
## 2                               83
##           Reference
## Prediction   N   Y
##          N 758 183
##          Y   1  58
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              758
## 2            Y                              183
##   over50k.fctr.predict.All.X.glm.Y
## 1                                1
## 2                               58
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              759
## 2            Y                              241
##   over50k.fctr.predict.All.X.glm.Y
## 1                                0
## 2                                0
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.6544429
## 3        0.2 0.7001621
## 4        0.3 0.7352941
## 5        0.4 0.7360000
## 6        0.5 0.6943231
## 7        0.6 0.6954436
## 8        0.7 0.6259947
## 9        0.8 0.5030303
## 10       0.9 0.3866667
## 11       1.0 0.0000000
```

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              684
## 2            Y                               57
##   over50k.fctr.predict.All.X.glm.Y
## 1                               75
## 2                              184
##           Reference
## Prediction   N   Y
##          N 684  57
##          Y  75 184
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              684
## 2            Y                               57
##   over50k.fctr.predict.All.X.glm.Y
## 1                               75
## 2                              184
##          Prediction
## Reference   N   Y
##         N 684  75
##         Y  57 184
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.680000e-01   6.481520e-01   8.454346e-01   8.883720e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   5.968845e-18   1.389640e-01
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](USCensus_Earnings_files/figure-html/fit.models_0-34.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-35.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                                0
## 2            Y                                0
##   over50k.fctr.predict.All.X.glm.Y
## 1                              759
## 2                              241
##           Reference
## Prediction   N   Y
##          N 519  36
##          Y 240 205
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              519
## 2            Y                               36
##   over50k.fctr.predict.All.X.glm.Y
## 1                              240
## 2                              205
##           Reference
## Prediction   N   Y
##          N 601  56
##          Y 158 185
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              601
## 2            Y                               56
##   over50k.fctr.predict.All.X.glm.Y
## 1                              158
## 2                              185
##           Reference
## Prediction   N   Y
##          N 641  75
##          Y 118 166
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              641
## 2            Y                               75
##   over50k.fctr.predict.All.X.glm.Y
## 1                              118
## 2                              166
##           Reference
## Prediction   N   Y
##          N 671  86
##          Y  88 155
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              671
## 2            Y                               86
##   over50k.fctr.predict.All.X.glm.Y
## 1                               88
## 2                              155
##           Reference
## Prediction   N   Y
##          N 694  99
##          Y  65 142
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              694
## 2            Y                               99
##   over50k.fctr.predict.All.X.glm.Y
## 1                               65
## 2                              142
##           Reference
## Prediction   N   Y
##          N 715 116
##          Y  44 125
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              715
## 2            Y                              116
##   over50k.fctr.predict.All.X.glm.Y
## 1                               44
## 2                              125
##           Reference
## Prediction   N   Y
##          N 730 139
##          Y  29 102
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              730
## 2            Y                              139
##   over50k.fctr.predict.All.X.glm.Y
## 1                               29
## 2                              102
##           Reference
## Prediction   N   Y
##          N 740 167
##          Y  19  74
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              740
## 2            Y                              167
##   over50k.fctr.predict.All.X.glm.Y
## 1                               19
## 2                               74
##           Reference
## Prediction   N   Y
##          N 747 192
##          Y  12  49
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              747
## 2            Y                              192
##   over50k.fctr.predict.All.X.glm.Y
## 1                               12
## 2                               49
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              759
## 2            Y                              241
##   over50k.fctr.predict.All.X.glm.Y
## 1                                0
## 2                                0
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.5976676
## 3        0.2 0.6335616
## 4        0.3 0.6323810
## 5        0.4 0.6404959
## 6        0.5 0.6339286
## 7        0.6 0.6097561
## 8        0.7 0.5483871
## 9        0.8 0.4431138
## 10       0.9 0.3245033
## 11       1.0 0.0000000
```

![](USCensus_Earnings_files/figure-html/fit.models_0-36.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              671
## 2            Y                               86
##   over50k.fctr.predict.All.X.glm.Y
## 1                               88
## 2                              155
##           Reference
## Prediction   N   Y
##          N 671  86
##          Y  88 155
##   over50k.fctr over50k.fctr.predict.All.X.glm.N
## 1            N                              671
## 2            Y                               86
##   over50k.fctr.predict.All.X.glm.Y
## 1                               88
## 2                              155
##          Prediction
## Reference   N   Y
##         N 671  88
##         Y  86 155
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.260000e-01   5.257227e-01   8.010531e-01   8.490099e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   1.789552e-07   9.395704e-01 
##    model_id model_method
## 1 All.X.glm          glm
##                                                                                                                                                                                  feats
## 1 age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr, .rnorm
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                      3.119                 0.496
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.9325658                    0.4           0.736        0.8299947
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8454346              0.888372     0.5291816    0.835337
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.4       0.6404959            0.826
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.8010531             0.8490099     0.5257227    719.1617
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.05275524       0.1292318
## [1] "iterating over method:rpart"
## [1] "fitting model: All.X.rpart"
## [1] "    indep_vars: age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr"
## + Fold1: cp=0.03734 
## - Fold1: cp=0.03734 
## + Fold2: cp=0.03734 
## - Fold2: cp=0.03734 
## + Fold3: cp=0.03734 
## - Fold3: cp=0.03734 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.0373 on full training set
```

```
## Warning in myfit_mdl(model_id = paste0(model_id_pfx, ""), model_method =
## method, : model's bestTune found at an extreme of tuneGrid for parameter:
## cp
```

![](USCensus_Earnings_files/figure-html/fit.models_0-37.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-38.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1000 
## 
##           CP nsplit rel error
## 1 0.07883817      0 1.0000000
## 2 0.04356846      2 0.8423237
## 3 0.03734440      5 0.6763485
## 
## Variable importance
## maritalstatus.fctr Married-civ-spouse 
##                                    28 
##             relationship.fctr Husband 
##                                    25 
##                                   age 
##                                    11 
##                           capitalgain 
##                                    10 
##                       sex.fctr Female 
##                                     9 
##           relationship.fctr Own-child 
##                                     5 
##           maritalstatus.fctr Divorced 
##                                     4 
##        occupation.fctr Prof-specialty 
##                                     4 
##       occupation.fctr Exec-managerial 
##                                     3 
## 
## Node number 1: 1000 observations,    complexity param=0.07883817
##   predicted class=N  expected loss=0.241  P(node) =1
##     class counts:   759   241
##    probabilities: 0.759 0.241 
##   left son=2 (539 obs) right son=3 (461 obs)
##   Primary splits:
##       maritalstatus.fctr Married-civ-spouse < 0.5    to the left,  improve=72.48758, (0 missing)
##       relationship.fctr Husband             < 0.5    to the left,  improve=56.59330, (0 missing)
##       capitalgain                           < 6897.5 to the left,  improve=55.55498, (0 missing)
##       age                                   < 28.5   to the left,  improve=33.75374, (0 missing)
##       hoursperweek                          < 49.5   to the left,  improve=21.28575, (0 missing)
##   Surrogate splits:
##       relationship.fctr Husband   < 0.5    to the left,  agree=0.947, adj=0.885, (0 split)
##       sex.fctr Female             < 0.5    to the right, agree=0.695, adj=0.338, (0 split)
##       age                         < 31.5   to the left,  agree=0.644, adj=0.228, (0 split)
##       relationship.fctr Own-child < 0.5    to the right, agree=0.620, adj=0.176, (0 split)
##       maritalstatus.fctr Divorced < 0.5    to the right, agree=0.609, adj=0.152, (0 split)
## 
## Node number 2: 539 observations
##   predicted class=N  expected loss=0.06493506  P(node) =0.539
##     class counts:   504    35
##    probabilities: 0.935 0.065 
## 
## Node number 3: 461 observations,    complexity param=0.07883817
##   predicted class=N  expected loss=0.4468547  P(node) =0.461
##     class counts:   255   206
##    probabilities: 0.553 0.447 
##   left son=6 (423 obs) right son=7 (38 obs)
##   Primary splits:
##       capitalgain                    < 5095.5 to the left,  improve=25.342690, (0 missing)
##       age                            < 28.5   to the left,  improve=13.026120, (0 missing)
##       occupation.fctr Prof-specialty < 0.5    to the left,  improve=12.368610, (0 missing)
##       education.fctr HS-grad         < 0.5    to the right, improve=10.892940, (0 missing)
##       hoursperweek                   < 49     to the left,  improve= 8.044688, (0 missing)
## 
## Node number 6: 423 observations,    complexity param=0.04356846
##   predicted class=N  expected loss=0.3971631  P(node) =0.423
##     class counts:   255   168
##    probabilities: 0.603 0.397 
##   left son=12 (41 obs) right son=13 (382 obs)
##   Primary splits:
##       age                             < 28.5   to the left,  improve=11.020560, (0 missing)
##       education.fctr HS-grad          < 0.5    to the right, improve= 8.436452, (0 missing)
##       occupation.fctr Prof-specialty  < 0.5    to the left,  improve= 7.792584, (0 missing)
##       capitalloss                     < 1813.5 to the left,  improve= 7.227105, (0 missing)
##       occupation.fctr Exec-managerial < 0.5    to the left,  improve= 6.992129, (0 missing)
## 
## Node number 7: 38 observations
##   predicted class=Y  expected loss=0  P(node) =0.038
##     class counts:     0    38
##    probabilities: 0.000 1.000 
## 
## Node number 12: 41 observations
##   predicted class=N  expected loss=0.04878049  P(node) =0.041
##     class counts:    39     2
##    probabilities: 0.951 0.049 
## 
## Node number 13: 382 observations,    complexity param=0.04356846
##   predicted class=N  expected loss=0.434555  P(node) =0.382
##     class counts:   216   166
##    probabilities: 0.565 0.435 
##   left son=26 (315 obs) right son=27 (67 obs)
##   Primary splits:
##       occupation.fctr Exec-managerial < 0.5    to the left,  improve=8.020381, (0 missing)
##       education.fctr HS-grad          < 0.5    to the right, improve=7.467051, (0 missing)
##       occupation.fctr Prof-specialty  < 0.5    to the left,  improve=7.368971, (0 missing)
##       occupation.fctr Other-service   < 0.5    to the right, improve=6.306984, (0 missing)
##       capitalloss                     < 1813.5 to the left,  improve=5.929517, (0 missing)
## 
## Node number 26: 315 observations,    complexity param=0.04356846
##   predicted class=N  expected loss=0.3873016  P(node) =0.315
##     class counts:   193   122
##    probabilities: 0.613 0.387 
##   left son=52 (262 obs) right son=53 (53 obs)
##   Primary splits:
##       occupation.fctr Prof-specialty < 0.5    to the left,  improve=10.862090, (0 missing)
##       education.fctr HS-grad         < 0.5    to the right, improve= 5.445683, (0 missing)
##       workclass.fctr Federal-gov     < 0.5    to the left,  improve= 5.375036, (0 missing)
##       occupation.fctr Other-service  < 0.5    to the right, improve= 5.056941, (0 missing)
##       occupation.fctr Sales          < 0.5    to the left,  improve= 4.800413, (0 missing)
##   Surrogate splits:
##       education.fctr Prof-school < 0.5    to the left,  agree=0.851, adj=0.113, (0 split)
##       education.fctr Masters     < 0.5    to the left,  agree=0.848, adj=0.094, (0 split)
##       education.fctr Doctorate   < 0.5    to the left,  agree=0.848, adj=0.094, (0 split)
##       capitalloss                < 1894.5 to the left,  agree=0.838, adj=0.038, (0 split)
## 
## Node number 27: 67 observations
##   predicted class=Y  expected loss=0.3432836  P(node) =0.067
##     class counts:    23    44
##    probabilities: 0.343 0.657 
## 
## Node number 52: 262 observations
##   predicted class=N  expected loss=0.3282443  P(node) =0.262
##     class counts:   176    86
##    probabilities: 0.672 0.328 
## 
## Node number 53: 53 observations
##   predicted class=Y  expected loss=0.3207547  P(node) =0.053
##     class counts:    17    36
##    probabilities: 0.321 0.679 
## 
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 1000 241 N (0.75900000 0.24100000)  
##    2) maritalstatus.fctr Married-civ-spouse< 0.5 539  35 N (0.93506494 0.06493506) *
##    3) maritalstatus.fctr Married-civ-spouse>=0.5 461 206 N (0.55314534 0.44685466)  
##      6) capitalgain< 5095.5 423 168 N (0.60283688 0.39716312)  
##       12) age< 28.5 41   2 N (0.95121951 0.04878049) *
##       13) age>=28.5 382 166 N (0.56544503 0.43455497)  
##         26) occupation.fctr Exec-managerial< 0.5 315 122 N (0.61269841 0.38730159)  
##           52) occupation.fctr Prof-specialty< 0.5 262  86 N (0.67175573 0.32824427) *
##           53) occupation.fctr Prof-specialty>=0.5 53  17 Y (0.32075472 0.67924528) *
##         27) occupation.fctr Exec-managerial>=0.5 67  23 Y (0.34328358 0.65671642) *
##      7) capitalgain>=5095.5 38   0 Y (0.00000000 1.00000000) *
```

![](USCensus_Earnings_files/figure-html/fit.models_0-39.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                  0
## 2            Y                                  0
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                759
## 2                                241
##           Reference
## Prediction   N   Y
##          N 543  37
##          Y 216 204
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                543
## 2            Y                                 37
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                216
## 2                                204
##           Reference
## Prediction   N   Y
##          N 543  37
##          Y 216 204
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                543
## 2            Y                                 37
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                216
## 2                                204
##           Reference
## Prediction   N   Y
##          N 543  37
##          Y 216 204
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                543
## 2            Y                                 37
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                216
## 2                                204
##           Reference
## Prediction   N   Y
##          N 719 123
##          Y  40 118
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                719
## 2            Y                                123
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                 40
## 2                                118
##           Reference
## Prediction   N   Y
##          N 719 123
##          Y  40 118
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                719
## 2            Y                                123
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                 40
## 2                                118
##           Reference
## Prediction   N   Y
##          N 719 123
##          Y  40 118
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                719
## 2            Y                                123
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                 40
## 2                                118
##           Reference
## Prediction   N   Y
##          N 759 203
##          Y   0  38
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                759
## 2            Y                                203
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                  0
## 2                                 38
##           Reference
## Prediction   N   Y
##          N 759 203
##          Y   0  38
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                759
## 2            Y                                203
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                  0
## 2                                 38
##           Reference
## Prediction   N   Y
##          N 759 203
##          Y   0  38
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                759
## 2            Y                                203
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                  0
## 2                                 38
##           Reference
## Prediction   N   Y
##          N 759 203
##          Y   0  38
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                759
## 2            Y                                203
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                  0
## 2                                 38
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.6172466
## 3        0.2 0.6172466
## 4        0.3 0.6172466
## 5        0.4 0.5914787
## 6        0.5 0.5914787
## 7        0.6 0.5914787
## 8        0.7 0.2724014
## 9        0.8 0.2724014
## 10       0.9 0.2724014
## 11       1.0 0.2724014
```

![](USCensus_Earnings_files/figure-html/fit.models_0-40.png) 

```
## [1] "Classifier Probability Threshold: 0.3000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                543
## 2            Y                                 37
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                216
## 2                                204
##           Reference
## Prediction   N   Y
##          N 543  37
##          Y 216 204
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                543
## 2            Y                                 37
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                216
## 2                                204
##          Prediction
## Reference   N   Y
##         N 543 216
##         Y  37 204
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.470000e-01   4.482729e-01   7.188527e-01   7.736848e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   8.225678e-01   4.525170e-29
```

![](USCensus_Earnings_files/figure-html/fit.models_0-41.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                  0
## 2            Y                                  0
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                759
## 2                                241
##           Reference
## Prediction   N   Y
##          N 539  44
##          Y 220 197
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                539
## 2            Y                                 44
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                220
## 2                                197
##           Reference
## Prediction   N   Y
##          N 539  44
##          Y 220 197
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                539
## 2            Y                                 44
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                220
## 2                                197
##           Reference
## Prediction   N   Y
##          N 539  44
##          Y 220 197
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                539
## 2            Y                                 44
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                220
## 2                                197
##           Reference
## Prediction   N   Y
##          N 714 123
##          Y  45 118
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                714
## 2            Y                                123
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                 45
## 2                                118
##           Reference
## Prediction   N   Y
##          N 714 123
##          Y  45 118
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                714
## 2            Y                                123
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                 45
## 2                                118
##           Reference
## Prediction   N   Y
##          N 714 123
##          Y  45 118
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                714
## 2            Y                                123
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                 45
## 2                                118
##           Reference
## Prediction   N   Y
##          N 759 210
##          Y   0  31
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                759
## 2            Y                                210
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                  0
## 2                                 31
##           Reference
## Prediction   N   Y
##          N 759 210
##          Y   0  31
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                759
## 2            Y                                210
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                  0
## 2                                 31
##           Reference
## Prediction   N   Y
##          N 759 210
##          Y   0  31
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                759
## 2            Y                                210
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                  0
## 2                                 31
##           Reference
## Prediction   N   Y
##          N 759 210
##          Y   0  31
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                759
## 2            Y                                210
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                  0
## 2                                 31
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.5987842
## 3        0.2 0.5987842
## 4        0.3 0.5987842
## 5        0.4 0.5841584
## 6        0.5 0.5841584
## 7        0.6 0.5841584
## 8        0.7 0.2279412
## 9        0.8 0.2279412
## 10       0.9 0.2279412
## 11       1.0 0.2279412
```

![](USCensus_Earnings_files/figure-html/fit.models_0-42.png) 

```
## [1] "Classifier Probability Threshold: 0.3000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                539
## 2            Y                                 44
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                220
## 2                                197
##           Reference
## Prediction   N   Y
##          N 539  44
##          Y 220 197
##   over50k.fctr over50k.fctr.predict.All.X.rpart.N
## 1            N                                539
## 2            Y                                 44
##   over50k.fctr.predict.All.X.rpart.Y
## 1                                220
## 2                                197
##          Prediction
## Reference   N   Y
##         N 539 220
##         Y  44 197
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.360000e-01   4.223271e-01   7.075104e-01   7.630926e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   9.577710e-01   4.743896e-27 
##      model_id model_method
## 1 All.X.rpart        rpart
##                                                                                                                                                                          feats
## 1 age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                      2.392                 0.247
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.8336586                    0.3       0.6172466        0.8020056
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.7188527             0.7736848     0.3405157   0.8063651
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.3       0.5987842            0.736
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.7075104             0.7630926     0.4223271
##   max.AccuracySD.fit max.KappaSD.fit
## 1        0.005703043       0.1004617
## [1] "fitting model: All.X.cp.0.rpart"
## [1] "    indep_vars: age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr"
## Fitting cp = 0 on full training set
```

![](USCensus_Earnings_files/figure-html/fit.models_0-43.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 1000 
## 
##            CP nsplit rel error
## 1 0.078838174      0 1.0000000
## 2 0.043568465      2 0.8423237
## 3 0.037344398      5 0.6763485
## 4 0.020746888      7 0.6016598
## 5 0.016597510      8 0.5809129
## 6 0.012448133      9 0.5643154
## 7 0.004149378     10 0.5518672
## 8 0.002074689     17 0.5186722
## 9 0.000000000     23 0.5062241
## 
## Variable importance
## maritalstatus.fctr Married-civ-spouse 
##                                    23 
##             relationship.fctr Husband 
##                                    20 
##                           capitalgain 
##                                    13 
##                                   age 
##                                    10 
##                       sex.fctr Female 
##                                     8 
##           relationship.fctr Own-child 
##                                     4 
##           maritalstatus.fctr Divorced 
##                                     3 
##        occupation.fctr Prof-specialty 
##                                     3 
##                 occupation.fctr Sales 
##                                     3 
##       occupation.fctr Exec-managerial 
##                                     3 
##                          hoursperweek 
##                                     2 
##          occupation.fctr Tech-support 
##                                     2 
##                           capitalloss 
##                                     1 
##       workclass.fctr Self-emp-not-inc 
##                                     1 
##                education.fctr HS-grad 
##                                     1 
##         occupation.fctr Other-service 
##                                     1 
##          occupation.fctr Craft-repair 
##                                     1 
##           workclass.fctr Self-emp-inc 
##                                     1 
## 
## Node number 1: 1000 observations,    complexity param=0.07883817
##   predicted class=N  expected loss=0.241  P(node) =1
##     class counts:   759   241
##    probabilities: 0.759 0.241 
##   left son=2 (539 obs) right son=3 (461 obs)
##   Primary splits:
##       maritalstatus.fctr Married-civ-spouse < 0.5    to the left,  improve=72.48758, (0 missing)
##       relationship.fctr Husband             < 0.5    to the left,  improve=56.59330, (0 missing)
##       capitalgain                           < 6897.5 to the left,  improve=55.55498, (0 missing)
##       age                                   < 28.5   to the left,  improve=33.75374, (0 missing)
##       hoursperweek                          < 49.5   to the left,  improve=21.28575, (0 missing)
##   Surrogate splits:
##       relationship.fctr Husband   < 0.5    to the left,  agree=0.947, adj=0.885, (0 split)
##       sex.fctr Female             < 0.5    to the right, agree=0.695, adj=0.338, (0 split)
##       age                         < 31.5   to the left,  agree=0.644, adj=0.228, (0 split)
##       relationship.fctr Own-child < 0.5    to the right, agree=0.620, adj=0.176, (0 split)
##       maritalstatus.fctr Divorced < 0.5    to the right, agree=0.609, adj=0.152, (0 split)
## 
## Node number 2: 539 observations,    complexity param=0.0373444
##   predicted class=N  expected loss=0.06493506  P(node) =0.539
##     class counts:   504    35
##    probabilities: 0.935 0.065 
##   left son=4 (526 obs) right son=5 (13 obs)
##   Primary splits:
##       capitalgain                     < 4718.5 to the left,  improve=16.260040, (0 missing)
##       hoursperweek                    < 44.5   to the left,  improve= 4.598427, (0 missing)
##       occupation.fctr Exec-managerial < 0.5    to the left,  improve= 3.861460, (0 missing)
##       age                             < 29.5   to the left,  improve= 2.470557, (0 missing)
##       education.fctr Masters          < 0.5    to the left,  improve= 1.310401, (0 missing)
## 
## Node number 3: 461 observations,    complexity param=0.07883817
##   predicted class=N  expected loss=0.4468547  P(node) =0.461
##     class counts:   255   206
##    probabilities: 0.553 0.447 
##   left son=6 (423 obs) right son=7 (38 obs)
##   Primary splits:
##       capitalgain                    < 5095.5 to the left,  improve=25.342690, (0 missing)
##       age                            < 28.5   to the left,  improve=13.026120, (0 missing)
##       occupation.fctr Prof-specialty < 0.5    to the left,  improve=12.368610, (0 missing)
##       education.fctr HS-grad         < 0.5    to the right, improve=10.892940, (0 missing)
##       hoursperweek                   < 49     to the left,  improve= 8.044688, (0 missing)
## 
## Node number 4: 526 observations
##   predicted class=N  expected loss=0.04562738  P(node) =0.526
##     class counts:   502    24
##    probabilities: 0.954 0.046 
## 
## Node number 5: 13 observations
##   predicted class=Y  expected loss=0.1538462  P(node) =0.013
##     class counts:     2    11
##    probabilities: 0.154 0.846 
## 
## Node number 6: 423 observations,    complexity param=0.04356846
##   predicted class=N  expected loss=0.3971631  P(node) =0.423
##     class counts:   255   168
##    probabilities: 0.603 0.397 
##   left son=12 (41 obs) right son=13 (382 obs)
##   Primary splits:
##       age                             < 28.5   to the left,  improve=11.020560, (0 missing)
##       education.fctr HS-grad          < 0.5    to the right, improve= 8.436452, (0 missing)
##       occupation.fctr Prof-specialty  < 0.5    to the left,  improve= 7.792584, (0 missing)
##       capitalloss                     < 1813.5 to the left,  improve= 7.227105, (0 missing)
##       occupation.fctr Exec-managerial < 0.5    to the left,  improve= 6.992129, (0 missing)
## 
## Node number 7: 38 observations
##   predicted class=Y  expected loss=0  P(node) =0.038
##     class counts:     0    38
##    probabilities: 0.000 1.000 
## 
## Node number 12: 41 observations
##   predicted class=N  expected loss=0.04878049  P(node) =0.041
##     class counts:    39     2
##    probabilities: 0.951 0.049 
## 
## Node number 13: 382 observations,    complexity param=0.04356846
##   predicted class=N  expected loss=0.434555  P(node) =0.382
##     class counts:   216   166
##    probabilities: 0.565 0.435 
##   left son=26 (315 obs) right son=27 (67 obs)
##   Primary splits:
##       occupation.fctr Exec-managerial < 0.5    to the left,  improve=8.020381, (0 missing)
##       education.fctr HS-grad          < 0.5    to the right, improve=7.467051, (0 missing)
##       occupation.fctr Prof-specialty  < 0.5    to the left,  improve=7.368971, (0 missing)
##       occupation.fctr Other-service   < 0.5    to the right, improve=6.306984, (0 missing)
##       capitalloss                     < 1813.5 to the left,  improve=5.929517, (0 missing)
## 
## Node number 26: 315 observations,    complexity param=0.04356846
##   predicted class=N  expected loss=0.3873016  P(node) =0.315
##     class counts:   193   122
##    probabilities: 0.613 0.387 
##   left son=52 (262 obs) right son=53 (53 obs)
##   Primary splits:
##       occupation.fctr Prof-specialty < 0.5    to the left,  improve=10.862090, (0 missing)
##       education.fctr HS-grad         < 0.5    to the right, improve= 5.445683, (0 missing)
##       workclass.fctr Federal-gov     < 0.5    to the left,  improve= 5.375036, (0 missing)
##       occupation.fctr Other-service  < 0.5    to the right, improve= 5.056941, (0 missing)
##       occupation.fctr Sales          < 0.5    to the left,  improve= 4.800413, (0 missing)
##   Surrogate splits:
##       education.fctr Prof-school < 0.5    to the left,  agree=0.851, adj=0.113, (0 split)
##       education.fctr Masters     < 0.5    to the left,  agree=0.848, adj=0.094, (0 split)
##       education.fctr Doctorate   < 0.5    to the left,  agree=0.848, adj=0.094, (0 split)
##       capitalloss                < 1894.5 to the left,  agree=0.838, adj=0.038, (0 split)
## 
## Node number 27: 67 observations,    complexity param=0.01244813
##   predicted class=Y  expected loss=0.3432836  P(node) =0.067
##     class counts:    23    44
##    probabilities: 0.343 0.657 
##   left son=54 (7 obs) right son=55 (60 obs)
##   Primary splits:
##       workclass.fctr Self-emp-not-inc < 0.5    to the right, improve=2.1518120, (0 missing)
##       capitalloss                     < 1649   to the left,  improve=2.1411590, (0 missing)
##       education.fctr HS-grad          < 0.5    to the right, improve=1.4004450, (0 missing)
##       workclass.fctr Private          < 0.5    to the left,  improve=0.8810273, (0 missing)
##       hoursperweek                    < 33.5   to the left,  improve=0.8137171, (0 missing)
## 
## Node number 52: 262 observations,    complexity param=0.0373444
##   predicted class=N  expected loss=0.3282443  P(node) =0.262
##     class counts:   176    86
##    probabilities: 0.672 0.328 
##   left son=104 (215 obs) right son=105 (47 obs)
##   Primary splits:
##       occupation.fctr Sales         < 0.5    to the left,  improve=8.196710, (0 missing)
##       occupation.fctr Tech-support  < 0.5    to the left,  improve=3.766842, (0 missing)
##       occupation.fctr Other-service < 0.5    to the right, improve=3.672066, (0 missing)
##       hoursperweek                  < 33.5   to the left,  improve=2.574169, (0 missing)
##       workclass.fctr Self-emp-inc   < 0.5    to the left,  improve=2.255608, (0 missing)
##   Surrogate splits:
##       workclass.fctr Self-emp-inc < 0.5    to the left,  agree=0.840, adj=0.106, (0 split)
##       hoursperweek                < 77.5   to the left,  agree=0.832, adj=0.064, (0 split)
## 
## Node number 53: 53 observations,    complexity param=0.002074689
##   predicted class=Y  expected loss=0.3207547  P(node) =0.053
##     class counts:    17    36
##    probabilities: 0.321 0.679 
##   left son=106 (45 obs) right son=107 (8 obs)
##   Primary splits:
##       capitalloss            < 951    to the left,  improve=1.9387840, (0 missing)
##       hoursperweek           < 42.5   to the left,  improve=1.4521110, (0 missing)
##       age                    < 34.5   to the left,  improve=1.1953500, (0 missing)
##       relationship.fctr Wife < 0.5    to the left,  improve=0.9529255, (0 missing)
##       sex.fctr Female        < 0.5    to the left,  improve=0.9529255, (0 missing)
## 
## Node number 54: 7 observations
##   predicted class=N  expected loss=0.2857143  P(node) =0.007
##     class counts:     5     2
##    probabilities: 0.714 0.286 
## 
## Node number 55: 60 observations,    complexity param=0.002074689
##   predicted class=Y  expected loss=0.3  P(node) =0.06
##     class counts:    18    42
##    probabilities: 0.300 0.700 
##   left son=110 (52 obs) right son=111 (8 obs)
##   Primary splits:
##       capitalloss            < 1649   to the left,  improve=1.6615380, (0 missing)
##       hoursperweek           < 36.5   to the left,  improve=1.1676550, (0 missing)
##       education.fctr HS-grad < 0.5    to the right, improve=1.0730160, (0 missing)
##       age                    < 55     to the right, improve=0.9600000, (0 missing)
##       sex.fctr Female        < 0.5    to the right, improve=0.2619946, (0 missing)
## 
## Node number 104: 215 observations,    complexity param=0.02074689
##   predicted class=N  expected loss=0.2697674  P(node) =0.215
##     class counts:   157    58
##    probabilities: 0.730 0.270 
##   left son=208 (206 obs) right son=209 (9 obs)
##   Primary splits:
##       occupation.fctr Tech-support    < 0.5    to the left,  improve=4.848293, (0 missing)
##       occupation.fctr Other-service   < 0.5    to the right, improve=2.516022, (0 missing)
##       hoursperweek                    < 33.5   to the left,  improve=2.199854, (0 missing)
##       occupation.fctr Farming-fishing < 0.5    to the right, improve=1.526489, (0 missing)
##       race.fctr Black                 < 0.5    to the left,  improve=1.432618, (0 missing)
## 
## Node number 105: 47 observations,    complexity param=0.01659751
##   predicted class=Y  expected loss=0.4042553  P(node) =0.047
##     class counts:    19    28
##    probabilities: 0.404 0.596 
##   left son=210 (12 obs) right son=211 (35 obs)
##   Primary splits:
##       education.fctr HS-grad      < 0.5    to the right, improve=2.2192500, (0 missing)
##       workclass.fctr Self-emp-inc < 0.5    to the left,  improve=1.9131520, (0 missing)
##       workclass.fctr Private      < 0.5    to the right, improve=0.5603758, (0 missing)
##       hoursperweek                < 38     to the left,  improve=0.4597264, (0 missing)
##       age                         < 38.5   to the left,  improve=0.4447495, (0 missing)
##   Surrogate splits:
##       age                             < 62.5   to the right, agree=0.766, adj=0.083, (0 split)
##       workclass.fctr Self-emp-not-inc < 0.5    to the right, agree=0.766, adj=0.083, (0 split)
## 
## Node number 106: 45 observations,    complexity param=0.002074689
##   predicted class=Y  expected loss=0.3777778  P(node) =0.045
##     class counts:    17    28
##    probabilities: 0.378 0.622 
##   left son=212 (27 obs) right son=213 (18 obs)
##   Primary splits:
##       hoursperweek              < 42.5   to the left,  improve=1.4518520, (0 missing)
##       relationship.fctr Wife    < 0.5    to the left,  improve=1.2433930, (0 missing)
##       sex.fctr Female           < 0.5    to the left,  improve=1.2433930, (0 missing)
##       age                       < 34.5   to the left,  improve=1.1893390, (0 missing)
##       relationship.fctr Husband < 0.5    to the right, improve=0.5444444, (0 missing)
##   Surrogate splits:
##       age                         < 60.5   to the left,  agree=0.644, adj=0.111, (0 split)
##       workclass.fctr Self-emp-inc < 0.5    to the left,  agree=0.644, adj=0.111, (0 split)
##       workclass.fctr Federal-gov  < 0.5    to the left,  agree=0.622, adj=0.056, (0 split)
##       education.fctr Prof-school  < 0.5    to the left,  agree=0.622, adj=0.056, (0 split)
## 
## Node number 107: 8 observations
##   predicted class=Y  expected loss=0  P(node) =0.008
##     class counts:     0     8
##    probabilities: 0.000 1.000 
## 
## Node number 110: 52 observations,    complexity param=0.002074689
##   predicted class=Y  expected loss=0.3461538  P(node) =0.052
##     class counts:    18    34
##    probabilities: 0.346 0.654 
##   left son=220 (15 obs) right son=221 (37 obs)
##   Primary splits:
##       education.fctr HS-grad < 0.5    to the right, improve=1.4772000, (0 missing)
##       hoursperweek           < 51.5   to the left,  improve=1.2820510, (0 missing)
##       age                    < 55     to the right, improve=0.9544822, (0 missing)
##       workclass.fctr Private < 0.5    to the left,  improve=0.1884615, (0 missing)
##       education.fctr Masters < 0.5    to the left,  improve=0.1748252, (0 missing)
##   Surrogate splits:
##       age          < 55     to the right, agree=0.808, adj=0.333, (0 split)
##       hoursperweek < 22     to the left,  agree=0.750, adj=0.133, (0 split)
## 
## Node number 111: 8 observations
##   predicted class=Y  expected loss=0  P(node) =0.008
##     class counts:     0     8
##    probabilities: 0.000 1.000 
## 
## Node number 208: 206 observations,    complexity param=0.004149378
##   predicted class=N  expected loss=0.2475728  P(node) =0.206
##     class counts:   155    51
##    probabilities: 0.752 0.248 
##   left son=416 (16 obs) right son=417 (190 obs)
##   Primary splits:
##       occupation.fctr Other-service   < 0.5    to the right, improve=2.126520, (0 missing)
##       hoursperweek                    < 33.5   to the left,  improve=1.733043, (0 missing)
##       workclass.fctr Self-emp-not-inc < 0.5    to the right, improve=1.335836, (0 missing)
##       occupation.fctr Farming-fishing < 0.5    to the right, improve=1.288389, (0 missing)
##       race.fctr Black                 < 0.5    to the left,  improve=1.259816, (0 missing)
##   Surrogate splits:
##       capitalgain < 3941   to the right, agree=0.927, adj=0.062, (0 split)
## 
## Node number 209: 9 observations
##   predicted class=Y  expected loss=0.2222222  P(node) =0.009
##     class counts:     2     7
##    probabilities: 0.222 0.778 
## 
## Node number 210: 12 observations
##   predicted class=N  expected loss=0.3333333  P(node) =0.012
##     class counts:     8     4
##    probabilities: 0.667 0.333 
## 
## Node number 211: 35 observations
##   predicted class=Y  expected loss=0.3142857  P(node) =0.035
##     class counts:    11    24
##    probabilities: 0.314 0.686 
## 
## Node number 212: 27 observations,    complexity param=0.002074689
##   predicted class=Y  expected loss=0.4814815  P(node) =0.027
##     class counts:    13    14
##    probabilities: 0.481 0.519 
##   left son=424 (7 obs) right son=425 (20 obs)
##   Primary splits:
##       age                      < 49     to the right, improve=0.15291010, (0 missing)
##       education.fctr Masters   < 0.5    to the left,  improve=0.03703704, (0 missing)
##       workclass.fctr Private   < 0.5    to the left,  improve=0.02693603, (0 missing)
##       workclass.fctr Local-gov < 0.5    to the left,  improve=0.02693603, (0 missing)
##   Surrogate splits:
##       workclass.fctr Self-emp-not-inc < 0.5    to the right, agree=0.778, adj=0.143, (0 split)
## 
## Node number 213: 18 observations
##   predicted class=Y  expected loss=0.2222222  P(node) =0.018
##     class counts:     4    14
##    probabilities: 0.222 0.778 
## 
## Node number 220: 15 observations
##   predicted class=N  expected loss=0.4666667  P(node) =0.015
##     class counts:     8     7
##    probabilities: 0.533 0.467 
## 
## Node number 221: 37 observations
##   predicted class=Y  expected loss=0.2702703  P(node) =0.037
##     class counts:    10    27
##    probabilities: 0.270 0.730 
## 
## Node number 416: 16 observations
##   predicted class=N  expected loss=0  P(node) =0.016
##     class counts:    16     0
##    probabilities: 1.000 0.000 
## 
## Node number 417: 190 observations,    complexity param=0.004149378
##   predicted class=N  expected loss=0.2684211  P(node) =0.19
##     class counts:   139    51
##    probabilities: 0.732 0.268 
##   left son=834 (22 obs) right son=835 (168 obs)
##   Primary splits:
##       workclass.fctr Self-emp-not-inc < 0.5    to the right, improve=1.568022, (0 missing)
##       occupation.fctr Farming-fishing < 0.5    to the right, improve=1.521053, (0 missing)
##       hoursperweek                    < 33.5   to the left,  improve=1.416206, (0 missing)
##       capitalloss                     < 629    to the right, improve=1.361384, (0 missing)
##       race.fctr Black                 < 0.5    to the left,  improve=1.232164, (0 missing)
## 
## Node number 424: 7 observations
##   predicted class=N  expected loss=0.4285714  P(node) =0.007
##     class counts:     4     3
##    probabilities: 0.571 0.429 
## 
## Node number 425: 20 observations,    complexity param=0.002074689
##   predicted class=Y  expected loss=0.45  P(node) =0.02
##     class counts:     9    11
##    probabilities: 0.450 0.550 
##   left son=850 (9 obs) right son=851 (11 obs)
##   Primary splits:
##       age                      < 38     to the left,  improve=0.364646500, (0 missing)
##       workclass.fctr Local-gov < 0.5    to the right, improve=0.066666670, (0 missing)
##       workclass.fctr Private   < 0.5    to the left,  improve=0.001010101, (0 missing)
##   Surrogate splits:
##       workclass.fctr Private    < 0.5    to the left,  agree=0.70, adj=0.333, (0 split)
##       hoursperweek              < 37.5   to the left,  agree=0.65, adj=0.222, (0 split)
##       workclass.fctr Local-gov  < 0.5    to the right, agree=0.65, adj=0.222, (0 split)
##       education.fctr HS-grad    < 0.5    to the right, agree=0.65, adj=0.222, (0 split)
##       relationship.fctr Husband < 0.5    to the left,  agree=0.60, adj=0.111, (0 split)
## 
## Node number 834: 22 observations
##   predicted class=N  expected loss=0.09090909  P(node) =0.022
##     class counts:    20     2
##    probabilities: 0.909 0.091 
## 
## Node number 835: 168 observations,    complexity param=0.004149378
##   predicted class=N  expected loss=0.2916667  P(node) =0.168
##     class counts:   119    49
##    probabilities: 0.708 0.292 
##   left son=1670 (27 obs) right son=1671 (141 obs)
##   Primary splits:
##       age                             < 60     to the right, improve=1.3252560, (0 missing)
##       capitalloss                     < 629    to the right, improve=1.2427540, (0 missing)
##       occupation.fctr Protective-serv < 0.5    to the right, improve=1.2427540, (0 missing)
##       education.fctr 7th-8th          < 0.5    to the right, improve=0.9488033, (0 missing)
##       race.fctr Black                 < 0.5    to the left,  improve=0.9411111, (0 missing)
##   Surrogate splits:
##       hoursperweek      < 18     to the left,  agree=0.869, adj=0.185, (0 split)
##       capitalloss       < 1706   to the right, agree=0.863, adj=0.148, (0 split)
##       workclass.fctr ?  < 0.5    to the right, agree=0.857, adj=0.111, (0 split)
##       occupation.fctr ? < 0.5    to the right, agree=0.857, adj=0.111, (0 split)
##       capitalgain       < 3467.5 to the right, agree=0.845, adj=0.037, (0 split)
## 
## Node number 850: 9 observations
##   predicted class=N  expected loss=0.4444444  P(node) =0.009
##     class counts:     5     4
##    probabilities: 0.556 0.444 
## 
## Node number 851: 11 observations
##   predicted class=Y  expected loss=0.3636364  P(node) =0.011
##     class counts:     4     7
##    probabilities: 0.364 0.636 
## 
## Node number 1670: 27 observations
##   predicted class=N  expected loss=0.1481481  P(node) =0.027
##     class counts:    23     4
##    probabilities: 0.852 0.148 
## 
## Node number 1671: 141 observations,    complexity param=0.004149378
##   predicted class=N  expected loss=0.3191489  P(node) =0.141
##     class counts:    96    45
##    probabilities: 0.681 0.319 
##   left son=3342 (93 obs) right son=3343 (48 obs)
##   Primary splits:
##       age                          < 44.5   to the left,  improve=1.3841230, (0 missing)
##       race.fctr Black              < 0.5    to the left,  improve=1.1805960, (0 missing)
##       education.fctr 7th-8th       < 0.5    to the right, improve=0.6393777, (0 missing)
##       hoursperweek                 < 52.5   to the right, improve=0.4765957, (0 missing)
##       occupation.fctr Craft-repair < 0.5    to the right, improve=0.4637811, (0 missing)
##   Surrogate splits:
##       workclass.fctr Federal-gov < 0.5    to the left,  agree=0.688, adj=0.083, (0 split)
##       capitalgain                < 2841.5 to the left,  agree=0.674, adj=0.042, (0 split)
##       education.fctr 10th        < 0.5    to the left,  agree=0.667, adj=0.021, (0 split)
## 
## Node number 3342: 93 observations
##   predicted class=N  expected loss=0.2688172  P(node) =0.093
##     class counts:    68    25
##    probabilities: 0.731 0.269 
## 
## Node number 3343: 48 observations,    complexity param=0.004149378
##   predicted class=N  expected loss=0.4166667  P(node) =0.048
##     class counts:    28    20
##    probabilities: 0.583 0.417 
##   left son=6686 (15 obs) right son=6687 (33 obs)
##   Primary splits:
##       occupation.fctr Craft-repair      < 0.5    to the right, improve=2.0484850, (0 missing)
##       hoursperweek                      < 51     to the right, improve=1.2288040, (0 missing)
##       age                               < 52.5   to the right, improve=1.1111110, (0 missing)
##       occupation.fctr Machine-op-inspct < 0.5    to the left,  improve=0.4733825, (0 missing)
##       education.fctr Some-college       < 0.5    to the right, improve=0.2810685, (0 missing)
## 
## Node number 6686: 15 observations
##   predicted class=N  expected loss=0.2  P(node) =0.015
##     class counts:    12     3
##    probabilities: 0.800 0.200 
## 
## Node number 6687: 33 observations,    complexity param=0.004149378
##   predicted class=Y  expected loss=0.4848485  P(node) =0.033
##     class counts:    16    17
##    probabilities: 0.485 0.515 
##   left son=13374 (7 obs) right son=13375 (26 obs)
##   Primary splits:
##       hoursperweek                      < 51     to the right, improve=2.46287000, (0 missing)
##       age                               < 52.5   to the right, improve=1.84638700, (0 missing)
##       occupation.fctr Transport-moving  < 0.5    to the right, improve=0.81818180, (0 missing)
##       race.fctr Black                   < 0.5    to the right, improve=0.13320010, (0 missing)
##       occupation.fctr Machine-op-inspct < 0.5    to the left,  improve=0.03030303, (0 missing)
##   Surrogate splits:
##       workclass.fctr Self-emp-inc < 0.5    to the right, agree=0.848, adj=0.286, (0 split)
##       education.fctr Some-college < 0.5    to the right, agree=0.818, adj=0.143, (0 split)
## 
## Node number 13374: 7 observations
##   predicted class=N  expected loss=0.1428571  P(node) =0.007
##     class counts:     6     1
##    probabilities: 0.857 0.143 
## 
## Node number 13375: 26 observations,    complexity param=0.004149378
##   predicted class=Y  expected loss=0.3846154  P(node) =0.026
##     class counts:    10    16
##    probabilities: 0.385 0.615 
##   left son=26750 (10 obs) right son=26751 (16 obs)
##   Primary splits:
##       age                               < 52.5   to the right, improve=1.507692000, (0 missing)
##       occupation.fctr Transport-moving  < 0.5    to the right, improve=1.335470000, (0 missing)
##       occupation.fctr Machine-op-inspct < 0.5    to the right, improve=0.432692300, (0 missing)
##       education.fctr HS-grad            < 0.5    to the right, improve=0.007692308, (0 missing)
##   Surrogate splits:
##       capitalgain         < 1551.5 to the right, agree=0.731, adj=0.3, (0 split)
##       education.fctr 10th < 0.5    to the right, agree=0.731, adj=0.3, (0 split)
## 
## Node number 26750: 10 observations
##   predicted class=N  expected loss=0.4  P(node) =0.01
##     class counts:     6     4
##    probabilities: 0.600 0.400 
## 
## Node number 26751: 16 observations
##   predicted class=Y  expected loss=0.25  P(node) =0.016
##     class counts:     4    12
##    probabilities: 0.250 0.750 
## 
## n= 1000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##     1) root 1000 241 N (0.75900000 0.24100000)  
##       2) maritalstatus.fctr Married-civ-spouse< 0.5 539  35 N (0.93506494 0.06493506)  
##         4) capitalgain< 4718.5 526  24 N (0.95437262 0.04562738) *
##         5) capitalgain>=4718.5 13   2 Y (0.15384615 0.84615385) *
##       3) maritalstatus.fctr Married-civ-spouse>=0.5 461 206 N (0.55314534 0.44685466)  
##         6) capitalgain< 5095.5 423 168 N (0.60283688 0.39716312)  
##          12) age< 28.5 41   2 N (0.95121951 0.04878049) *
##          13) age>=28.5 382 166 N (0.56544503 0.43455497)  
##            26) occupation.fctr Exec-managerial< 0.5 315 122 N (0.61269841 0.38730159)  
##              52) occupation.fctr Prof-specialty< 0.5 262  86 N (0.67175573 0.32824427)  
##               104) occupation.fctr Sales< 0.5 215  58 N (0.73023256 0.26976744)  
##                 208) occupation.fctr Tech-support< 0.5 206  51 N (0.75242718 0.24757282)  
##                   416) occupation.fctr Other-service>=0.5 16   0 N (1.00000000 0.00000000) *
##                   417) occupation.fctr Other-service< 0.5 190  51 N (0.73157895 0.26842105)  
##                     834) workclass.fctr Self-emp-not-inc>=0.5 22   2 N (0.90909091 0.09090909) *
##                     835) workclass.fctr Self-emp-not-inc< 0.5 168  49 N (0.70833333 0.29166667)  
##                      1670) age>=60 27   4 N (0.85185185 0.14814815) *
##                      1671) age< 60 141  45 N (0.68085106 0.31914894)  
##                        3342) age< 44.5 93  25 N (0.73118280 0.26881720) *
##                        3343) age>=44.5 48  20 N (0.58333333 0.41666667)  
##                          6686) occupation.fctr Craft-repair>=0.5 15   3 N (0.80000000 0.20000000) *
##                          6687) occupation.fctr Craft-repair< 0.5 33  16 Y (0.48484848 0.51515152)  
##                           13374) hoursperweek>=51 7   1 N (0.85714286 0.14285714) *
##                           13375) hoursperweek< 51 26  10 Y (0.38461538 0.61538462)  
##                             26750) age>=52.5 10   4 N (0.60000000 0.40000000) *
##                             26751) age< 52.5 16   4 Y (0.25000000 0.75000000) *
##                 209) occupation.fctr Tech-support>=0.5 9   2 Y (0.22222222 0.77777778) *
##               105) occupation.fctr Sales>=0.5 47  19 Y (0.40425532 0.59574468)  
##                 210) education.fctr HS-grad>=0.5 12   4 N (0.66666667 0.33333333) *
##                 211) education.fctr HS-grad< 0.5 35  11 Y (0.31428571 0.68571429) *
##              53) occupation.fctr Prof-specialty>=0.5 53  17 Y (0.32075472 0.67924528)  
##               106) capitalloss< 951 45  17 Y (0.37777778 0.62222222)  
##                 212) hoursperweek< 42.5 27  13 Y (0.48148148 0.51851852)  
##                   424) age>=49 7   3 N (0.57142857 0.42857143) *
##                   425) age< 49 20   9 Y (0.45000000 0.55000000)  
##                     850) age< 38 9   4 N (0.55555556 0.44444444) *
##                     851) age>=38 11   4 Y (0.36363636 0.63636364) *
##                 213) hoursperweek>=42.5 18   4 Y (0.22222222 0.77777778) *
##               107) capitalloss>=951 8   0 Y (0.00000000 1.00000000) *
##            27) occupation.fctr Exec-managerial>=0.5 67  23 Y (0.34328358 0.65671642)  
##              54) workclass.fctr Self-emp-not-inc>=0.5 7   2 N (0.71428571 0.28571429) *
##              55) workclass.fctr Self-emp-not-inc< 0.5 60  18 Y (0.30000000 0.70000000)  
##               110) capitalloss< 1649 52  18 Y (0.34615385 0.65384615)  
##                 220) education.fctr HS-grad>=0.5 15   7 N (0.53333333 0.46666667) *
##                 221) education.fctr HS-grad< 0.5 37  10 Y (0.27027027 0.72972973) *
##               111) capitalloss>=1649 8   0 Y (0.00000000 1.00000000) *
##         7) capitalgain>=5095.5 38   0 Y (0.00000000 1.00000000) *
```

![](USCensus_Earnings_files/figure-html/fit.models_0-44.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                       0
## 2            Y                                       0
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                     759
## 2                                     241
##           Reference
## Prediction   N   Y
##          N 577  28
##          Y 182 213
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     577
## 2            Y                                      28
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                     182
## 2                                     213
##           Reference
## Prediction   N   Y
##          N 606  33
##          Y 153 208
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     606
## 2            Y                                      33
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                     153
## 2                                     208
##           Reference
## Prediction   N   Y
##          N 691  63
##          Y  68 178
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     691
## 2            Y                                      63
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      68
## 2                                     178
##           Reference
## Prediction   N   Y
##          N 699  67
##          Y  60 174
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     699
## 2            Y                                      67
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      60
## 2                                     174
##           Reference
## Prediction   N   Y
##          N 722  85
##          Y  37 156
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     722
## 2            Y                                      85
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      37
## 2                                     156
##           Reference
## Prediction   N   Y
##          N 722  85
##          Y  37 156
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     722
## 2            Y                                      85
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      37
## 2                                     156
##           Reference
## Prediction   N   Y
##          N 737 116
##          Y  22 125
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     737
## 2            Y                                     116
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      22
## 2                                     125
##           Reference
## Prediction   N   Y
##          N 757 176
##          Y   2  65
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     757
## 2            Y                                     176
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                       2
## 2                                      65
##           Reference
## Prediction   N   Y
##          N 759 187
##          Y   0  54
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     759
## 2            Y                                     187
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                       0
## 2                                      54
##           Reference
## Prediction   N   Y
##          N 759 187
##          Y   0  54
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     759
## 2            Y                                     187
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                       0
## 2                                      54
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.6698113
## 3        0.2 0.6910299
## 4        0.3 0.7310062
## 5        0.4 0.7326316
## 6        0.5 0.7188940
## 7        0.6 0.7188940
## 8        0.7 0.6443299
## 9        0.8 0.4220779
## 10       0.9 0.3661017
## 11       1.0 0.3661017
```

![](USCensus_Earnings_files/figure-html/fit.models_0-45.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     699
## 2            Y                                      67
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      60
## 2                                     174
##           Reference
## Prediction   N   Y
##          N 699  67
##          Y  60 174
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     699
## 2            Y                                      67
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      60
## 2                                     174
##          Prediction
## Reference   N   Y
##         N 699  60
##         Y  67 174
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.730000e-01   6.493766e-01   8.507600e-01   8.930157e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   1.318095e-19   5.944394e-01
```

![](USCensus_Earnings_files/figure-html/fit.models_0-46.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                       0
## 2            Y                                       0
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                     759
## 2                                     241
##           Reference
## Prediction   N   Y
##          N 564  39
##          Y 195 202
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     564
## 2            Y                                      39
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                     195
## 2                                     202
##           Reference
## Prediction   N   Y
##          N 586  48
##          Y 173 193
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     586
## 2            Y                                      48
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                     173
## 2                                     193
##           Reference
## Prediction   N   Y
##          N 664  83
##          Y  95 158
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     664
## 2            Y                                      83
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      95
## 2                                     158
##           Reference
## Prediction   N   Y
##          N 673  88
##          Y  86 153
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     673
## 2            Y                                      88
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      86
## 2                                     153
##           Reference
## Prediction   N   Y
##          N 698 113
##          Y  61 128
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     698
## 2            Y                                     113
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      61
## 2                                     128
##           Reference
## Prediction   N   Y
##          N 698 113
##          Y  61 128
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     698
## 2            Y                                     113
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      61
## 2                                     128
##           Reference
## Prediction   N   Y
##          N 718 134
##          Y  41 107
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     718
## 2            Y                                     134
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      41
## 2                                     107
##           Reference
## Prediction   N   Y
##          N 755 188
##          Y   4  53
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     755
## 2            Y                                     188
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                       4
## 2                                      53
##           Reference
## Prediction   N   Y
##          N 758 199
##          Y   1  42
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     758
## 2            Y                                     199
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                       1
## 2                                      42
##           Reference
## Prediction   N   Y
##          N 758 199
##          Y   1  42
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     758
## 2            Y                                     199
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                       1
## 2                                      42
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.6332288
## 3        0.2 0.6359143
## 4        0.3 0.6396761
## 5        0.4 0.6375000
## 6        0.5 0.5953488
## 7        0.6 0.5953488
## 8        0.7 0.5501285
## 9        0.8 0.3557047
## 10       0.9 0.2957746
## 11       1.0 0.2957746
```

```
## [1] "Classifier Probability Threshold: 0.3000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     664
## 2            Y                                      83
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      95
## 2                                     158
##           Reference
## Prediction   N   Y
##          N 664  83
##          Y  95 158
##   over50k.fctr over50k.fctr.predict.All.X.cp.0.rpart.N
## 1            N                                     664
## 2            Y                                      83
##   over50k.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      95
## 2                                     158
##          Prediction
## Reference   N   Y
##         N 664  95
##         Y  83 158
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.220000e-01   5.215748e-01   7.968549e-01   8.452324e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   9.132554e-07   4.096641e-01 
##           model_id model_method
## 1 All.X.cp.0.rpart        rpart
##                                                                                                                                                                          feats
## 1 age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               0                      0.719                 0.242
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.8936906                    0.4       0.7326316            0.873
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1               0.85076             0.8930157     0.6493766   0.8506005
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.3       0.6396761            0.822
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.7968549             0.8452324     0.5215748
## [1] "iterating over method:rf"
## [1] "fitting model: All.X.rf"
## [1] "    indep_vars: age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr, .rnorm"
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

![](USCensus_Earnings_files/figure-html/fit.models_0-47.png) 

```
## + : mtry= 2 
## - : mtry= 2 
## + : mtry=50 
## - : mtry=50 
## + : mtry=98 
## - : mtry=98 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 98 on full training set
```

```
## Warning in myfit_mdl(model_id = paste0(model_id_pfx, ""), model_method =
## method, : model's bestTune found at an extreme of tuneGrid for parameter:
## mtry
```

![](USCensus_Earnings_files/figure-html/fit.models_0-48.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-49.png) 

```
##                 Length Class      Mode     
## call               4   -none-     call     
## type               1   -none-     character
## predicted       1000   factor     numeric  
## err.rate        1500   -none-     numeric  
## confusion          6   -none-     numeric  
## votes           2000   matrix     numeric  
## oob.times       1000   -none-     numeric  
## classes            2   -none-     character
## importance        98   -none-     numeric  
## importanceSD       0   -none-     NULL     
## localImportance    0   -none-     NULL     
## proximity          0   -none-     NULL     
## ntree              1   -none-     numeric  
## mtry               1   -none-     numeric  
## forest            14   -none-     list     
## y               1000   factor     numeric  
## test               0   -none-     NULL     
## inbag              0   -none-     NULL     
## xNames            98   -none-     character
## problemType        1   -none-     character
## tuneValue          1   data.frame list     
## obsLevels          2   -none-     character
```

![](USCensus_Earnings_files/figure-html/fit.models_0-50.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                               0
## 2            Y                               0
##   over50k.fctr.predict.All.X.rf.Y
## 1                             759
## 2                             241
##           Reference
## Prediction   N   Y
##          N 581   0
##          Y 178 241
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             581
## 2            Y                               0
##   over50k.fctr.predict.All.X.rf.Y
## 1                             178
## 2                             241
##           Reference
## Prediction   N   Y
##          N 706   0
##          Y  53 241
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             706
## 2            Y                               0
##   over50k.fctr.predict.All.X.rf.Y
## 1                              53
## 2                             241
##           Reference
## Prediction   N   Y
##          N 750   0
##          Y   9 241
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             750
## 2            Y                               0
##   over50k.fctr.predict.All.X.rf.Y
## 1                               9
## 2                             241
##           Reference
## Prediction   N   Y
##          N 759   0
##          Y   0 241
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             759
## 2            Y                               0
##   over50k.fctr.predict.All.X.rf.Y
## 1                               0
## 2                             241
##           Reference
## Prediction   N   Y
##          N 759   0
##          Y   0 241
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             759
## 2            Y                               0
##   over50k.fctr.predict.All.X.rf.Y
## 1                               0
## 2                             241
##           Reference
## Prediction   N   Y
##          N 759   0
##          Y   0 241
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             759
## 2            Y                               0
##   over50k.fctr.predict.All.X.rf.Y
## 1                               0
## 2                             241
##           Reference
## Prediction   N   Y
##          N 759  38
##          Y   0 203
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             759
## 2            Y                              38
##   over50k.fctr.predict.All.X.rf.Y
## 1                               0
## 2                             203
##           Reference
## Prediction   N   Y
##          N 759  97
##          Y   0 144
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             759
## 2            Y                              97
##   over50k.fctr.predict.All.X.rf.Y
## 1                               0
## 2                             144
##           Reference
## Prediction   N   Y
##          N 759 155
##          Y   0  86
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             759
## 2            Y                             155
##   over50k.fctr.predict.All.X.rf.Y
## 1                               0
## 2                              86
##           Reference
## Prediction   N   Y
##          N 759 223
##          Y   0  18
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             759
## 2            Y                             223
##   over50k.fctr.predict.All.X.rf.Y
## 1                               0
## 2                              18
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.7303030
## 3        0.2 0.9009346
## 4        0.3 0.9816701
## 5        0.4 1.0000000
## 6        0.5 1.0000000
## 7        0.6 1.0000000
## 8        0.7 0.9144144
## 9        0.8 0.7480519
## 10       0.9 0.5259939
## 11       1.0 0.1389961
```

![](USCensus_Earnings_files/figure-html/fit.models_0-51.png) 

```
## [1] "Classifier Probability Threshold: 0.6000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             759
## 2            Y                              NA
##   over50k.fctr.predict.All.X.rf.Y
## 1                              NA
## 2                             241
##           Reference
## Prediction   N   Y
##          N 759   0
##          Y   0 241
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             759
## 2            Y                               0
##   over50k.fctr.predict.All.X.rf.Y
## 1                               0
## 2                             241
##          Prediction
## Reference   N   Y
##         N 759   0
##         Y   0 241
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   1.000000e+00   1.000000e+00   9.963179e-01   1.000000e+00   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##  1.744922e-120            NaN
```

![](USCensus_Earnings_files/figure-html/fit.models_0-52.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                               0
## 2            Y                               0
##   over50k.fctr.predict.All.X.rf.Y
## 1                             759
## 2                             241
##           Reference
## Prediction   N   Y
##          N 483  19
##          Y 276 222
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             483
## 2            Y                              19
##   over50k.fctr.predict.All.X.rf.Y
## 1                             276
## 2                             222
##           Reference
## Prediction   N   Y
##          N 555  37
##          Y 204 204
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             555
## 2            Y                              37
##   over50k.fctr.predict.All.X.rf.Y
## 1                             204
## 2                             204
##           Reference
## Prediction   N   Y
##          N 608  58
##          Y 151 183
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             608
## 2            Y                              58
##   over50k.fctr.predict.All.X.rf.Y
## 1                             151
## 2                             183
##           Reference
## Prediction   N   Y
##          N 658  77
##          Y 101 164
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             658
## 2            Y                              77
##   over50k.fctr.predict.All.X.rf.Y
## 1                             101
## 2                             164
##           Reference
## Prediction   N   Y
##          N 696  96
##          Y  63 145
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             696
## 2            Y                              96
##   over50k.fctr.predict.All.X.rf.Y
## 1                              63
## 2                             145
##           Reference
## Prediction   N   Y
##          N 718 119
##          Y  41 122
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             718
## 2            Y                             119
##   over50k.fctr.predict.All.X.rf.Y
## 1                              41
## 2                             122
##           Reference
## Prediction   N   Y
##          N 735 144
##          Y  24  97
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             735
## 2            Y                             144
##   over50k.fctr.predict.All.X.rf.Y
## 1                              24
## 2                              97
##           Reference
## Prediction   N   Y
##          N 743 175
##          Y  16  66
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             743
## 2            Y                             175
##   over50k.fctr.predict.All.X.rf.Y
## 1                              16
## 2                              66
##           Reference
## Prediction   N   Y
##          N 755 194
##          Y   4  47
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             755
## 2            Y                             194
##   over50k.fctr.predict.All.X.rf.Y
## 1                               4
## 2                              47
##           Reference
## Prediction   N   Y
##          N 759 234
##          Y   0   7
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             759
## 2            Y                             234
##   over50k.fctr.predict.All.X.rf.Y
## 1                               0
## 2                               7
##    threshold    f.score
## 1        0.0 0.38839645
## 2        0.1 0.60081191
## 3        0.2 0.62865948
## 4        0.3 0.63652174
## 5        0.4 0.64822134
## 6        0.5 0.64587973
## 7        0.6 0.60396040
## 8        0.7 0.53591160
## 9        0.8 0.40866873
## 10       0.9 0.32191781
## 11       1.0 0.05645161
```

![](USCensus_Earnings_files/figure-html/fit.models_0-53.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             658
## 2            Y                              77
##   over50k.fctr.predict.All.X.rf.Y
## 1                             101
## 2                             164
##           Reference
## Prediction   N   Y
##          N 658  77
##          Y 101 164
##   over50k.fctr over50k.fctr.predict.All.X.rf.N
## 1            N                             658
## 2            Y                              77
##   over50k.fctr.predict.All.X.rf.Y
## 1                             101
## 2                             164
##          Prediction
## Reference   N   Y
##         N 658 101
##         Y  77 164
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.220000e-01   5.294366e-01   7.968549e-01   8.452324e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   9.132554e-07   8.472177e-02 
##   model_id model_method
## 1 All.X.rf           rf
##                                                                                                                                                                                  feats
## 1 age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr, .rnorm
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                     23.395                 6.721
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1           1                    0.6               1            0.829
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.9963179                     1     0.4992562   0.8749638
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.4       0.6482213            0.822
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.7968549             0.8452324     0.5294366
## [1] "fitting model: All.X.no.rnorm.rf"
## [1] "    indep_vars: age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr"
## + : mtry= 2 
## - : mtry= 2 
## + : mtry=49 
## - : mtry=49 
## + : mtry=97 
## - : mtry=97 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 49 on full training set
```

![](USCensus_Earnings_files/figure-html/fit.models_0-54.png) ![](USCensus_Earnings_files/figure-html/fit.models_0-55.png) 

```
##                 Length Class      Mode     
## call               4   -none-     call     
## type               1   -none-     character
## predicted       1000   factor     numeric  
## err.rate        1500   -none-     numeric  
## confusion          6   -none-     numeric  
## votes           2000   matrix     numeric  
## oob.times       1000   -none-     numeric  
## classes            2   -none-     character
## importance        97   -none-     numeric  
## importanceSD       0   -none-     NULL     
## localImportance    0   -none-     NULL     
## proximity          0   -none-     NULL     
## ntree              1   -none-     numeric  
## mtry               1   -none-     numeric  
## forest            14   -none-     list     
## y               1000   factor     numeric  
## test               0   -none-     NULL     
## inbag              0   -none-     NULL     
## xNames            97   -none-     character
## problemType        1   -none-     character
## tuneValue          1   data.frame list     
## obsLevels          2   -none-     character
```

![](USCensus_Earnings_files/figure-html/fit.models_0-56.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                        0
## 2            Y                                        0
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                      759
## 2                                      241
##           Reference
## Prediction   N   Y
##          N 590   0
##          Y 169 241
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      590
## 2            Y                                        0
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                      169
## 2                                      241
##           Reference
## Prediction   N   Y
##          N 689   0
##          Y  70 241
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      689
## 2            Y                                        0
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                       70
## 2                                      241
##           Reference
## Prediction   N   Y
##          N 737   0
##          Y  22 241
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      737
## 2            Y                                        0
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                       22
## 2                                      241
##           Reference
## Prediction   N   Y
##          N 758   1
##          Y   1 240
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      758
## 2            Y                                        1
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                        1
## 2                                      240
##           Reference
## Prediction   N   Y
##          N 759   2
##          Y   0 239
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      759
## 2            Y                                        2
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                        0
## 2                                      239
##           Reference
## Prediction   N   Y
##          N 759   8
##          Y   0 233
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      759
## 2            Y                                        8
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                        0
## 2                                      233
##           Reference
## Prediction   N   Y
##          N 759  45
##          Y   0 196
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      759
## 2            Y                                       45
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                        0
## 2                                      196
##           Reference
## Prediction   N   Y
##          N 759 101
##          Y   0 140
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      759
## 2            Y                                      101
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                        0
## 2                                      140
##           Reference
## Prediction   N   Y
##          N 759 153
##          Y   0  88
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      759
## 2            Y                                      153
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                        0
## 2                                       88
##           Reference
## Prediction   N   Y
##          N 759 240
##          Y   0   1
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      759
## 2            Y                                      240
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                        0
## 2                                        1
##    threshold     f.score
## 1        0.0 0.388396454
## 2        0.1 0.740399386
## 3        0.2 0.873188406
## 4        0.3 0.956349206
## 5        0.4 0.995850622
## 6        0.5 0.995833333
## 7        0.6 0.983122363
## 8        0.7 0.897025172
## 9        0.8 0.734908136
## 10       0.9 0.534954407
## 11       1.0 0.008264463
```

![](USCensus_Earnings_files/figure-html/fit.models_0-57.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      758
## 2            Y                                        1
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                        1
## 2                                      240
##           Reference
## Prediction   N   Y
##          N 758   1
##          Y   1 240
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      758
## 2            Y                                        1
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                        1
## 2                                      240
##          Prediction
## Reference   N   Y
##         N 758   1
##         Y   1 240
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.980000e-01   9.945331e-01   9.927942e-01   9.997577e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##  8.843011e-116   1.000000e+00
```

![](USCensus_Earnings_files/figure-html/fit.models_0-58.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                        0
## 2            Y                                        0
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                      759
## 2                                      241
##           Reference
## Prediction   N   Y
##          N 496  22
##          Y 263 219
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      496
## 2            Y                                       22
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                      263
## 2                                      219
##           Reference
## Prediction   N   Y
##          N 557  37
##          Y 202 204
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      557
## 2            Y                                       37
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                      202
## 2                                      204
##           Reference
## Prediction   N   Y
##          N 620  52
##          Y 139 189
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      620
## 2            Y                                       52
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                      139
## 2                                      189
##           Reference
## Prediction   N   Y
##          N 661  77
##          Y  98 164
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      661
## 2            Y                                       77
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                       98
## 2                                      164
##           Reference
## Prediction   N   Y
##          N 699  92
##          Y  60 149
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      699
## 2            Y                                       92
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                       60
## 2                                      149
##           Reference
## Prediction   N   Y
##          N 717 117
##          Y  42 124
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      717
## 2            Y                                      117
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                       42
## 2                                      124
##           Reference
## Prediction   N   Y
##          N 738 141
##          Y  21 100
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      738
## 2            Y                                      141
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                       21
## 2                                      100
##           Reference
## Prediction   N   Y
##          N 742 164
##          Y  17  77
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      742
## 2            Y                                      164
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                       17
## 2                                       77
##           Reference
## Prediction   N   Y
##          N 753 192
##          Y   6  49
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      753
## 2            Y                                      192
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                        6
## 2                                       49
##           Reference
## Prediction   N   Y
##          N 759 239
##          Y   0   2
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      759
## 2            Y                                      239
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                        0
## 2                                        2
##    threshold    f.score
## 1        0.0 0.38839645
## 2        0.1 0.60580913
## 3        0.2 0.63060278
## 4        0.3 0.66432337
## 5        0.4 0.65208748
## 6        0.5 0.66222222
## 7        0.6 0.60933661
## 8        0.7 0.55248619
## 9        0.8 0.45970149
## 10       0.9 0.33108108
## 11       1.0 0.01646091
```

![](USCensus_Earnings_files/figure-html/fit.models_0-59.png) 

```
## [1] "Classifier Probability Threshold: 0.3000 to maximize f.score.OOB"
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      620
## 2            Y                                       52
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                      139
## 2                                      189
##           Reference
## Prediction   N   Y
##          N 620  52
##          Y 139 189
##   over50k.fctr over50k.fctr.predict.All.X.no.rnorm.rf.N
## 1            N                                      620
## 2            Y                                       52
##   over50k.fctr.predict.All.X.no.rnorm.rf.Y
## 1                                      139
## 2                                      189
##          Prediction
## Reference   N   Y
##         N 620 139
##         Y  52 189
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.090000e-01   5.351712e-01   7.832406e-01   8.329252e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   8.884330e-05   4.885416e-10 
##            model_id model_method
## 1 All.X.no.rnorm.rf           rf
##                                                                                                                                                                          feats
## 1 age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                     20.399                 5.049
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.9999891                    0.4       0.9958506            0.828
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.9927942             0.9997577     0.5147302   0.8805182
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.3       0.6643234            0.809
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.7832406             0.8329252     0.5351712
```

```r
# Simplify a model
# fit_df <- glb_trnent_df; glb_mdl <- step(<complex>_mdl)

print(glb_models_df)
```

```
##                     model_id     model_method
## 1          MFO.myMFO_classfr    myMFO_classfr
## 2    Random.myrandom_classfr myrandom_classfr
## 3       Max.cor.Y.cv.0.rpart            rpart
## 4  Max.cor.Y.cv.0.cp.0.rpart            rpart
## 5            Max.cor.Y.rpart            rpart
## 6              Max.cor.Y.glm              glm
## 7              Low.cor.X.glm              glm
## 8                  All.X.glm              glm
## 9                All.X.rpart            rpart
## 10          All.X.cp.0.rpart            rpart
## 11                  All.X.rf               rf
## 12         All.X.no.rnorm.rf               rf
##                                                                                                                                                                                   feats
## 1                                                                                                                                                                                .rnorm
## 2                                                                                                                                                                                .rnorm
## 3                                                                                                                                                                                   age
## 4                                                                                                                                                                                   age
## 5                                                                                                                                                                                   age
## 6                                                                                                                                                                                   age
## 7  age, capitalgain, hoursperweek, capitalloss, workclass.fctr, maritalstatus.fctr, race.fctr, .rnorm, education.fctr, nativecountry.fctr, occupation.fctr, relationship.fctr, sex.fctr
## 8  age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr, .rnorm
## 9          age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr
## 10         age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr
## 11 age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr, .rnorm
## 12         age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr
##    max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1                0                      0.372                 0.002
## 2                0                      0.223                 0.001
## 3                0                      0.779                 0.020
## 4                0                      0.668                 0.018
## 5                3                      0.978                 0.021
## 6                1                      1.074                 0.019
## 7                1                      2.830                 0.499
## 8                1                      3.119                 0.496
## 9                3                      2.392                 0.247
## 10               0                      0.719                 0.242
## 11               3                     23.395                 6.721
## 12               3                     20.399                 5.049
##    max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1    0.5000000                    0.5       0.0000000        0.7590000
## 2    0.5010961                    0.2       0.3883965        0.2410000
## 3    0.5000000                    0.5       0.0000000        0.7590000
## 4    0.7183097                    0.2       0.4791878        0.4870000
## 5    0.5000000                    0.5       0.0000000        0.7459975
## 6    0.6880368                    0.2       0.4728192        0.7550095
## 7    0.9325658                    0.4       0.7360000        0.8299947
## 8    0.9325658                    0.4       0.7360000        0.8299947
## 9    0.8336586                    0.3       0.6172466        0.8020056
## 10   0.8936906                    0.4       0.7326316        0.8730000
## 11   1.0000000                    0.6       1.0000000        0.8290000
## 12   0.9999891                    0.4       0.9958506        0.8280000
##    max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.7312521             0.7852137    0.00000000   0.5000000
## 2              0.2147863             0.2687479    0.00000000   0.4879209
## 3              0.7312521             0.7852137    0.00000000   0.5000000
## 4              0.4555980             0.5184787    0.18102402   0.6837507
## 5              0.7312521             0.7852137    0.05249700   0.5000000
## 6              0.5517330             0.6137769    0.02934095   0.6937196
## 7              0.8454346             0.8883720    0.52918164   0.8353370
## 8              0.8454346             0.8883720    0.52918164   0.8353370
## 9              0.7188527             0.7736848    0.34051565   0.8063651
## 10             0.8507600             0.8930157    0.64937661   0.8506005
## 11             0.9963179             1.0000000    0.49925621   0.8749638
## 12             0.9927942             0.9997577    0.51473020   0.8805182
##    opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                     0.5       0.0000000            0.759
## 2                     0.2       0.3883965            0.241
## 3                     0.5       0.0000000            0.759
## 4                     0.2       0.4763903            0.501
## 5                     0.5       0.0000000            0.759
## 6                     0.2       0.4916773            0.603
## 7                     0.4       0.6404959            0.826
## 8                     0.4       0.6404959            0.826
## 9                     0.3       0.5987842            0.736
## 10                    0.3       0.6396761            0.822
## 11                    0.4       0.6482213            0.822
## 12                    0.3       0.6643234            0.809
##    max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1              0.7312521             0.7852137     0.0000000
## 2              0.2147863             0.2687479     0.0000000
## 3              0.7312521             0.7852137     0.0000000
## 4              0.4695463             0.5324478     0.1817204
## 5              0.7312521             0.7852137     0.0000000
## 6              0.5719080             0.6334837     0.2375941
## 7              0.8010531             0.8490099     0.5257227
## 8              0.8010531             0.8490099     0.5257227
## 9              0.7075104             0.7630926     0.4223271
## 10             0.7968549             0.8452324     0.5215748
## 11             0.7968549             0.8452324     0.5294366
## 12             0.7832406             0.8329252     0.5351712
##    max.AccuracySD.fit max.KappaSD.fit min.aic.fit
## 1                  NA              NA          NA
## 2                  NA              NA          NA
## 3                  NA              NA          NA
## 4                  NA              NA          NA
## 5         0.012206415      0.02765327          NA
## 6         0.008758591      0.03529993   1048.0810
## 7         0.052755235      0.12923177    719.1617
## 8         0.052755235      0.12923177    719.1617
## 9         0.005703043      0.10046165          NA
## 10                 NA              NA          NA
## 11                 NA              NA          NA
## 12                 NA              NA          NA
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.models", 
    chunk_step_major=glb_script_df[nrow(glb_script_df), "chunk_step_major"], 
    chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,                              
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##          chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed8  fit.models                5                0   7.989
## elapsed9  fit.models                5                1  98.841
```


```r
if (!is.null(glb_model_metric_smmry)) {
    stats_df <- glb_models_df[, "model_id", FALSE]

    stats_mdl_df <- data.frame()
    for (model_id in stats_df$model_id) {
        stats_mdl_df <- rbind(stats_mdl_df, 
            mypredict_mdl(glb_models_lst[[model_id]], glb_trnent_df, glb_rsp_var, 
                          glb_rsp_var_out, model_id, "fit",
        						glb_model_metric_smmry, glb_model_metric, 
        						glb_model_metric_maximize, ret_type="stats"))
    }
    stats_df <- merge(stats_df, stats_mdl_df, all.x=TRUE)
    
    stats_mdl_df <- data.frame()
    for (model_id in stats_df$model_id) {
        stats_mdl_df <- rbind(stats_mdl_df, 
            mypredict_mdl(glb_models_lst[[model_id]], glb_newent_df, glb_rsp_var, 
                          glb_rsp_var_out, model_id, "OOB",
            					glb_model_metric_smmry, glb_model_metric, 
        						glb_model_metric_maximize, ret_type="stats"))
    }
    stats_df <- merge(stats_df, stats_mdl_df, all.x=TRUE)
    
#     tmp_models_df <- orderBy(~model_id, glb_models_df)
#     rownames(tmp_models_df) <- seq(1, nrow(tmp_models_df))
#     all.equal(subset(tmp_models_df[, names(stats_df)], model_id != "Random.myrandom_classfr"),
#               subset(stats_df, model_id != "Random.myrandom_classfr"))
#     print(subset(tmp_models_df[, names(stats_df)], model_id != "Random.myrandom_classfr")[, c("model_id", "max.Accuracy.fit")])
#     print(subset(stats_df, model_id != "Random.myrandom_classfr")[, c("model_id", "max.Accuracy.fit")])

    print("Merging following data into glb_models_df:")
    print(stats_mrg_df <- stats_df[, c(1, grep(glb_model_metric, names(stats_df)))])
    print(tmp_models_df <- orderBy(~model_id, glb_models_df[, c("model_id", grep(glb_model_metric, names(stats_df), value=TRUE))]))

    tmp2_models_df <- glb_models_df[, c("model_id", setdiff(names(glb_models_df), grep(glb_model_metric, names(stats_df), value=TRUE)))]
    tmp3_models_df <- merge(tmp2_models_df, stats_mrg_df, all.x=TRUE, sort=FALSE)
    print(tmp3_models_df)
    print(names(tmp3_models_df))
    print(glb_models_df <- subset(tmp3_models_df, select=-model_id.1))
}

plt_models_df <- glb_models_df[, -grep("SD|Upper|Lower", names(glb_models_df))]
for (var in grep("^min.", names(plt_models_df), value=TRUE)) {
    plt_models_df[, sub("min.", "inv.", var)] <- 
        #ifelse(all(is.na(tmp <- plt_models_df[, var])), NA, 1.0 / tmp)
        1.0 / plt_models_df[, var]
    plt_models_df <- plt_models_df[ , -grep(var, names(plt_models_df))]
}
print(plt_models_df)
```

```
##                     model_id     model_method
## 1          MFO.myMFO_classfr    myMFO_classfr
## 2    Random.myrandom_classfr myrandom_classfr
## 3       Max.cor.Y.cv.0.rpart            rpart
## 4  Max.cor.Y.cv.0.cp.0.rpart            rpart
## 5            Max.cor.Y.rpart            rpart
## 6              Max.cor.Y.glm              glm
## 7              Low.cor.X.glm              glm
## 8                  All.X.glm              glm
## 9                All.X.rpart            rpart
## 10          All.X.cp.0.rpart            rpart
## 11                  All.X.rf               rf
## 12         All.X.no.rnorm.rf               rf
##                                                                                                                                                                                   feats
## 1                                                                                                                                                                                .rnorm
## 2                                                                                                                                                                                .rnorm
## 3                                                                                                                                                                                   age
## 4                                                                                                                                                                                   age
## 5                                                                                                                                                                                   age
## 6                                                                                                                                                                                   age
## 7  age, capitalgain, hoursperweek, capitalloss, workclass.fctr, maritalstatus.fctr, race.fctr, .rnorm, education.fctr, nativecountry.fctr, occupation.fctr, relationship.fctr, sex.fctr
## 8  age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr, .rnorm
## 9          age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr
## 10         age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr
## 11 age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr, .rnorm
## 12         age, capitalgain, capitalloss, hoursperweek, workclass.fctr, education.fctr, maritalstatus.fctr, occupation.fctr, relationship.fctr, race.fctr, sex.fctr, nativecountry.fctr
##    max.nTuningRuns max.auc.fit opt.prob.threshold.fit max.f.score.fit
## 1                0   0.5000000                    0.5       0.0000000
## 2                0   0.5010961                    0.2       0.3883965
## 3                0   0.5000000                    0.5       0.0000000
## 4                0   0.7183097                    0.2       0.4791878
## 5                3   0.5000000                    0.5       0.0000000
## 6                1   0.6880368                    0.2       0.4728192
## 7                1   0.9325658                    0.4       0.7360000
## 8                1   0.9325658                    0.4       0.7360000
## 9                3   0.8336586                    0.3       0.6172466
## 10               0   0.8936906                    0.4       0.7326316
## 11               3   1.0000000                    0.6       1.0000000
## 12               3   0.9999891                    0.4       0.9958506
##    max.Accuracy.fit max.Kappa.fit max.auc.OOB opt.prob.threshold.OOB
## 1         0.7590000    0.00000000   0.5000000                    0.5
## 2         0.2410000    0.00000000   0.4879209                    0.2
## 3         0.7590000    0.00000000   0.5000000                    0.5
## 4         0.4870000    0.18102402   0.6837507                    0.2
## 5         0.7459975    0.05249700   0.5000000                    0.5
## 6         0.7550095    0.02934095   0.6937196                    0.2
## 7         0.8299947    0.52918164   0.8353370                    0.4
## 8         0.8299947    0.52918164   0.8353370                    0.4
## 9         0.8020056    0.34051565   0.8063651                    0.3
## 10        0.8730000    0.64937661   0.8506005                    0.3
## 11        0.8290000    0.49925621   0.8749638                    0.4
## 12        0.8280000    0.51473020   0.8805182                    0.3
##    max.f.score.OOB max.Accuracy.OOB max.Kappa.OOB
## 1        0.0000000            0.759     0.0000000
## 2        0.3883965            0.241     0.0000000
## 3        0.0000000            0.759     0.0000000
## 4        0.4763903            0.501     0.1817204
## 5        0.0000000            0.759     0.0000000
## 6        0.4916773            0.603     0.2375941
## 7        0.6404959            0.826     0.5257227
## 8        0.6404959            0.826     0.5257227
## 9        0.5987842            0.736     0.4223271
## 10       0.6396761            0.822     0.5215748
## 11       0.6482213            0.822     0.5294366
## 12       0.6643234            0.809     0.5351712
##    inv.elapsedtime.everything inv.elapsedtime.final  inv.aic.fit
## 1                  2.68817204           500.0000000           NA
## 2                  4.48430493          1000.0000000           NA
## 3                  1.28369705            50.0000000           NA
## 4                  1.49700599            55.5555556           NA
## 5                  1.02249489            47.6190476           NA
## 6                  0.93109870            52.6315789 0.0009541247
## 7                  0.35335689             2.0040080 0.0013905079
## 8                  0.32061558             2.0161290 0.0013905079
## 9                  0.41806020             4.0485830           NA
## 10                 1.39082058             4.1322314           NA
## 11                 0.04274418             0.1487874           NA
## 12                 0.04902201             0.1980590           NA
```

```r
print(myplot_radar(radar_inp_df=plt_models_df))
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 12. Consider specifying shapes manually. if you must have them.
```

```
## Warning: Removed 6 rows containing missing values (geom_path).
```

```
## Warning: Removed 89 rows containing missing values (geom_point).
```

```
## Warning: Removed 9 rows containing missing values (geom_text).
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 12. Consider specifying shapes manually. if you must have them.
```

![](USCensus_Earnings_files/figure-html/fit.models_1-1.png) 

```r
# print(myplot_radar(radar_inp_df=subset(plt_models_df, 
#         !(model_id %in% grep("random|MFO", plt_models_df$model_id, value=TRUE)))))

# Compute CI for <metric>SD
glb_models_df <- mutate(glb_models_df, 
                max.df = ifelse(max.nTuningRuns > 1, max.nTuningRuns - 1, NA),
                min.sd2ci.scaler = ifelse(is.na(max.df), NA, qt(0.975, max.df)))
for (var in grep("SD", names(glb_models_df), value=TRUE)) {
    # Does CI alredy exist ?
    var_components <- unlist(strsplit(var, "SD"))
    varActul <- paste0(var_components[1],          var_components[2])
    varUpper <- paste0(var_components[1], "Upper", var_components[2])
    varLower <- paste0(var_components[1], "Lower", var_components[2])
    if (varUpper %in% names(glb_models_df)) {
        warning(varUpper, " already exists in glb_models_df")
        # Assuming Lower also exists
        next
    }    
    print(sprintf("var:%s", var))
    # CI is dependent on sample size in t distribution; df=n-1
    glb_models_df[, varUpper] <- glb_models_df[, varActul] + 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
    glb_models_df[, varLower] <- glb_models_df[, varActul] - 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
}
```

```
## Warning: max.AccuracyUpper.fit already exists in glb_models_df
```

```
## [1] "var:max.KappaSD.fit"
```

```r
# Plot metrics with CI
plt_models_df <- glb_models_df[, "model_id", FALSE]
pltCI_models_df <- glb_models_df[, "model_id", FALSE]
for (var in grep("Upper", names(glb_models_df), value=TRUE)) {
    var_components <- unlist(strsplit(var, "Upper"))
    col_name <- unlist(paste(var_components, collapse=""))
    plt_models_df[, col_name] <- glb_models_df[, col_name]
    for (name in paste0(var_components[1], c("Upper", "Lower"), var_components[2]))
        pltCI_models_df[, name] <- glb_models_df[, name]
}

build_statsCI_data <- function(plt_models_df) {
    mltd_models_df <- melt(plt_models_df, id.vars="model_id")
    mltd_models_df$data <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) tail(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), "[.]")), 1))
    mltd_models_df$label <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) head(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), paste0(".", mltd_models_df[row_ix, "data"]))), 1))
    #print(mltd_models_df)
    
    return(mltd_models_df)
}
mltd_models_df <- build_statsCI_data(plt_models_df)

mltdCI_models_df <- melt(pltCI_models_df, id.vars="model_id")
for (row_ix in 1:nrow(mltdCI_models_df)) {
    for (type in c("Upper", "Lower")) {
        if (length(var_components <- unlist(strsplit(
                as.character(mltdCI_models_df[row_ix, "variable"]), type))) > 1) {
            #print(sprintf("row_ix:%d; type:%s; ", row_ix, type))
            mltdCI_models_df[row_ix, "label"] <- var_components[1]
            mltdCI_models_df[row_ix, "data"] <- unlist(strsplit(var_components[2], "[.]"))[2]
            mltdCI_models_df[row_ix, "type"] <- type
            break
        }
    }    
}
#print(mltdCI_models_df)
# castCI_models_df <- dcast(mltdCI_models_df, value ~ type, fun.aggregate=sum)
# print(castCI_models_df)
wideCI_models_df <- reshape(subset(mltdCI_models_df, select=-variable), 
                            timevar="type", 
        idvar=setdiff(names(mltdCI_models_df), c("type", "value", "variable")), 
                            direction="wide")
#print(wideCI_models_df)
mrgdCI_models_df <- merge(wideCI_models_df, mltd_models_df, all.x=TRUE)
#print(mrgdCI_models_df)

# Merge stats back in if CIs don't exist
goback_vars <- c()
for (var in unique(mltd_models_df$label)) {
    for (type in unique(mltd_models_df$data)) {
        var_type <- paste0(var, ".", type)
        # if this data is already present, next
        if (var_type %in% unique(paste(mltd_models_df$label, mltd_models_df$data, sep=".")))
            next
        #print(sprintf("var_type:%s", var_type))
        goback_vars <- c(goback_vars, var_type)
    }
}

if (length(goback_vars) > 0) {
    mltd_goback_df <- build_statsCI_data(glb_models_df[, c("model_id", goback_vars)])
    mltd_models_df <- rbind(mltd_models_df, mltd_goback_df)
}

mltd_models_df <- merge(mltd_models_df, glb_models_df[, c("model_id", "model_method")], all.x=TRUE)

# print(myplot_bar(mltd_models_df, "model_id", "value", colorcol_name="data") + 
#         geom_errorbar(data=mrgdCI_models_df, 
#             mapping=aes(x=model_id, ymax=value.Upper, ymin=value.Lower), width=0.5) + 
#           facet_grid(label ~ data, scales="free") + 
#           theme(axis.text.x = element_text(angle = 45,vjust = 1)))
# mltd_models_df <- orderBy(~ value +variable +data +label + model_method + model_id, 
#                           mltd_models_df)
print(myplot_bar(mltd_models_df, "model_id", "value", colorcol_name="model_method") + 
        geom_errorbar(data=mrgdCI_models_df, 
            mapping=aes(x=model_id, ymax=value.Upper, ymin=value.Lower), width=0.5) + 
          facet_grid(label ~ data, scales="free") + 
          theme(axis.text.x = element_text(angle = 90,vjust = 0.5)))
```

![](USCensus_Earnings_files/figure-html/fit.models_1-2.png) 

```r
model_evl_terms <- c(NULL)
for (metric in glb_model_evl_criteria)
    model_evl_terms <- c(model_evl_terms, 
                    ifelse(length(grep("max", metric)) > 0, "-", "+"), metric)
model_sel_frmla <- as.formula(paste(c("~ ", model_evl_terms), collapse=" "))
print(tmp_models_df <- orderBy(model_sel_frmla, glb_models_df)[, c("model_id", glb_model_evl_criteria)])
```

```
##                     model_id max.Accuracy.OOB max.Kappa.OOB min.aic.fit
## 7              Low.cor.X.glm            0.826     0.5257227    719.1617
## 8                  All.X.glm            0.826     0.5257227    719.1617
## 11                  All.X.rf            0.822     0.5294366          NA
## 10          All.X.cp.0.rpart            0.822     0.5215748          NA
## 12         All.X.no.rnorm.rf            0.809     0.5351712          NA
## 1          MFO.myMFO_classfr            0.759     0.0000000          NA
## 3       Max.cor.Y.cv.0.rpart            0.759     0.0000000          NA
## 5            Max.cor.Y.rpart            0.759     0.0000000          NA
## 9                All.X.rpart            0.736     0.4223271          NA
## 6              Max.cor.Y.glm            0.603     0.2375941   1048.0810
## 4  Max.cor.Y.cv.0.cp.0.rpart            0.501     0.1817204          NA
## 2    Random.myrandom_classfr            0.241     0.0000000          NA
```

```r
print(myplot_radar(radar_inp_df=tmp_models_df))
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 12. Consider specifying shapes manually. if you must have them.
```

```
## Warning: Removed 6 rows containing missing values (geom_path).
```

```
## Warning: Removed 23 rows containing missing values (geom_point).
```

```
## Warning: Removed 9 rows containing missing values (geom_text).
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 12. Consider specifying shapes manually. if you must have them.
```

![](USCensus_Earnings_files/figure-html/fit.models_1-3.png) 

```r
print("Metrics used for model selection:"); print(model_sel_frmla)
```

```
## [1] "Metrics used for model selection:"
```

```
## ~-max.Accuracy.OOB - max.Kappa.OOB + min.aic.fit
```

```r
print(sprintf("Best model id: %s", tmp_models_df[1, "model_id"]))
```

```
## [1] "Best model id: Low.cor.X.glm"
```

```r
if (is.null(glb_sel_mdl_id)) 
    { glb_sel_mdl_id <- tmp_models_df[1, "model_id"] } else 
        print(sprintf("User specified selection: %s", glb_sel_mdl_id))   
    
myprint_mdl(glb_sel_mdl <- glb_models_lst[[glb_sel_mdl_id]])
```

```
## Warning: not plotting observations with leverage one:
##   118, 160, 182, 194, 371, 422, 477, 518, 535, 638, 739, 798, 805, 808, 811, 921
```

![](USCensus_Earnings_files/figure-html/fit.models_1-4.png) ![](USCensus_Earnings_files/figure-html/fit.models_1-5.png) 

```
## Warning: not plotting observations with leverage one:
##   118, 160, 182, 194, 371, 422, 477, 518, 535, 638, 739, 798, 805, 808, 811, 921
```

![](USCensus_Earnings_files/figure-html/fit.models_1-6.png) 

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

![](USCensus_Earnings_files/figure-html/fit.models_1-7.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##      Min        1Q    Median        3Q       Max  
## -2.24059  -0.43840  -0.08362   0.00000   2.62797  
## 
## Coefficients: (12 not defined because of singularities)
##                                                   Estimate Std. Error
## (Intercept)                                     -6.821e+00  1.159e+00
## age                                              3.818e-02  1.089e-02
## capitalgain                                      4.161e-04  7.105e-05
## hoursperweek                                     2.970e-02  1.051e-02
## capitalloss                                      2.194e-04  2.357e-04
## `workclass.fctr Self-emp-not-inc`                1.610e-01  7.478e-01
## `workclass.fctr Private`                         1.125e+00  6.789e-01
## `workclass.fctr Federal-gov`                     2.904e+00  9.072e-01
## `workclass.fctr Local-gov`                       3.428e-01  7.691e-01
## `workclass.fctr ?`                               6.723e-01  9.573e-01
## `workclass.fctr Self-emp-inc`                    1.042e+00  8.233e-01
## `workclass.fctr Without-pay`                    -1.725e+01  1.075e+04
## `workclass.fctr Never-worked`                           NA         NA
## `maritalstatus.fctr Married-civ-spouse`         -1.878e+00  7.345e+03
## `maritalstatus.fctr Divorced`                    4.708e-01  5.824e-01
## `maritalstatus.fctr Married-spouse-absent`      -1.558e+01  2.722e+03
## `maritalstatus.fctr Separated`                   4.013e-01  8.565e-01
## `maritalstatus.fctr Married-AF-spouse`          -1.751e+01  1.302e+04
## `maritalstatus.fctr Widowed`                     1.088e+00  8.744e-01
## `race.fctr Black`                                4.090e-01  4.460e-01
## `race.fctr Asian-Pac-Islander`                   1.380e+00  1.491e+00
## `race.fctr Amer-Indian-Eskimo`                   5.347e-01  1.206e+00
## `race.fctr Other`                                1.267e+00  1.487e+00
## .rnorm                                          -3.095e-02  1.171e-01
## `education.fctr HS-grad`                        -1.559e+00  3.436e-01
## `education.fctr 11th`                           -1.815e+01  2.222e+03
## `education.fctr Masters`                        -2.940e-02  4.506e-01
## `education.fctr 9th`                            -1.666e+00  1.406e+00
## `education.fctr Some-college`                   -8.444e-01  3.563e-01
## `education.fctr Assoc-acdm`                     -6.287e-01  7.288e-01
## `education.fctr 7th-8th`                        -2.846e+00  1.162e+00
## `education.fctr Doctorate`                       1.534e+00  9.430e-01
## `education.fctr Assoc-voc`                      -8.237e-01  5.183e-01
## `education.fctr Prof-school`                     1.781e+00  9.115e-01
## `education.fctr 5th-6th`                        -2.305e+00  1.545e+00
## `education.fctr 10th`                           -1.951e+00  7.429e-01
## `education.fctr 1st-4th`                        -2.408e+00  5.929e+03
## `education.fctr Preschool`                      -2.289e+00  1.097e+04
## `education.fctr 12th`                           -1.821e+01  2.665e+03
## `nativecountry.fctr Cuba`                        2.854e+00  1.496e+00
## `nativecountry.fctr Jamaica`                     1.769e-01  1.363e+00
## `nativecountry.fctr India`                      -1.501e+00  1.781e+00
## `nativecountry.fctr Mexico`                     -1.704e+01  2.148e+03
## `nativecountry.fctr South`                       3.292e-01  1.099e+04
## `nativecountry.fctr Puerto-Rico`                -1.631e+01  7.046e+03
## `nativecountry.fctr Honduras`                           NA         NA
## `nativecountry.fctr England`                     9.965e-01  1.034e+00
## `nativecountry.fctr Canada`                      2.268e+00  2.148e+00
## `nativecountry.fctr Germany`                    -1.790e+01  4.090e+03
## `nativecountry.fctr Iran`                               NA         NA
## `nativecountry.fctr Philippines`                -1.854e+00  2.302e+00
## `nativecountry.fctr Italy`                      -1.582e+01  1.075e+04
## `nativecountry.fctr Poland`                     -1.698e+01  1.075e+04
## `nativecountry.fctr Columbia`                   -1.694e+01  4.154e+03
## `nativecountry.fctr Cambodia`                           NA         NA
## `nativecountry.fctr Thailand`                           NA         NA
## `nativecountry.fctr Ecuador`                    -1.120e+01  1.075e+04
## `nativecountry.fctr Laos`                        5.683e-01  2.207e+00
## `nativecountry.fctr Taiwan`                      1.847e+01  7.390e+03
## `nativecountry.fctr Haiti`                      -1.274e+01  1.075e+04
## `nativecountry.fctr Portugal`                   -1.658e+01  1.075e+04
## `nativecountry.fctr Dominican-Republic`                 NA         NA
## `nativecountry.fctr El-Salvador`                -1.699e+01  3.705e+03
## `nativecountry.fctr France`                     -2.093e+01  6.381e+03
## `nativecountry.fctr Guatemala`                   1.553e+01  1.175e+04
## `nativecountry.fctr China`                      -2.950e+00  2.128e+00
## `nativecountry.fctr Japan`                      -1.679e+01  1.075e+04
## `nativecountry.fctr Yugoslavia`                         NA         NA
## `nativecountry.fctr Peru`                       -1.283e+01  1.075e+04
## `nativecountry.fctr Outlying-US(Guam-USVI-etc)`         NA         NA
## `nativecountry.fctr Scotland`                   -1.651e+01  1.075e+04
## `nativecountry.fctr Trinadad&Tobago`            -1.849e+01  1.075e+04
## `nativecountry.fctr Greece`                     -1.473e+01  1.075e+04
## `nativecountry.fctr Nicaragua`                  -1.745e+01  1.075e+04
## `nativecountry.fctr Vietnam`                    -1.078e+00  7.665e+03
## `nativecountry.fctr Hong`                       -1.907e+01  1.075e+04
## `nativecountry.fctr Ireland`                    -1.925e+01  1.075e+04
## `nativecountry.fctr Hungary`                            NA         NA
## `nativecountry.fctr Holand-Netherlands`                 NA         NA
## `occupation.fctr Exec-managerial`                1.304e+00  4.950e-01
## `occupation.fctr Handlers-cleaners`              2.694e-02  8.208e-01
## `occupation.fctr Prof-specialty`                 1.153e+00  5.194e-01
## `occupation.fctr Other-service`                 -2.126e+00  1.190e+00
## `occupation.fctr Sales`                          1.069e+00  5.227e-01
## `occupation.fctr Transport-moving`               1.505e-01  6.237e-01
## `occupation.fctr Farming-fishing`               -1.718e+01  1.838e+03
## `occupation.fctr Machine-op-inspct`              3.740e-01  6.315e-01
## `occupation.fctr Tech-support`                   1.914e+00  7.198e-01
## `occupation.fctr Craft-repair`                   4.127e-01  5.182e-01
## `occupation.fctr ?`                                     NA         NA
## `occupation.fctr Protective-serv`               -1.078e+00  1.275e+00
## `occupation.fctr Armed-Forces`                          NA         NA
## `occupation.fctr Priv-house-serv`               -1.410e+01  4.167e+03
## `relationship.fctr Husband`                      4.541e+00  7.345e+03
## `relationship.fctr Wife`                         6.133e+00  7.345e+03
## `relationship.fctr Own-child`                   -9.141e-01  1.018e+00
## `relationship.fctr Unmarried`                    1.153e+00  5.278e-01
## `relationship.fctr Other-relative`              -1.367e+01  2.263e+03
## `sex.fctr Female`                               -1.083e+00  4.967e-01
##                                                 z value Pr(>|z|)    
## (Intercept)                                      -5.886 3.95e-09 ***
## age                                               3.507 0.000454 ***
## capitalgain                                       5.857 4.71e-09 ***
## hoursperweek                                      2.828 0.004691 ** 
## capitalloss                                       0.931 0.351975    
## `workclass.fctr Self-emp-not-inc`                 0.215 0.829567    
## `workclass.fctr Private`                          1.657 0.097516 .  
## `workclass.fctr Federal-gov`                      3.201 0.001370 ** 
## `workclass.fctr Local-gov`                        0.446 0.655834    
## `workclass.fctr ?`                                0.702 0.482461    
## `workclass.fctr Self-emp-inc`                     1.265 0.205836    
## `workclass.fctr Without-pay`                     -0.002 0.998720    
## `workclass.fctr Never-worked`                        NA       NA    
## `maritalstatus.fctr Married-civ-spouse`           0.000 0.999796    
## `maritalstatus.fctr Divorced`                     0.808 0.418880    
## `maritalstatus.fctr Married-spouse-absent`       -0.006 0.995434    
## `maritalstatus.fctr Separated`                    0.469 0.639391    
## `maritalstatus.fctr Married-AF-spouse`           -0.001 0.998927    
## `maritalstatus.fctr Widowed`                      1.244 0.213532    
## `race.fctr Black`                                 0.917 0.359078    
## `race.fctr Asian-Pac-Islander`                    0.925 0.354892    
## `race.fctr Amer-Indian-Eskimo`                    0.443 0.657450    
## `race.fctr Other`                                 0.852 0.394420    
## .rnorm                                           -0.264 0.791562    
## `education.fctr HS-grad`                         -4.537 5.71e-06 ***
## `education.fctr 11th`                            -0.008 0.993482    
## `education.fctr Masters`                         -0.065 0.947977    
## `education.fctr 9th`                             -1.185 0.236034    
## `education.fctr Some-college`                    -2.370 0.017790 *  
## `education.fctr Assoc-acdm`                      -0.863 0.388343    
## `education.fctr 7th-8th`                         -2.450 0.014284 *  
## `education.fctr Doctorate`                        1.627 0.103707    
## `education.fctr Assoc-voc`                       -1.589 0.112020    
## `education.fctr Prof-school`                      1.954 0.050691 .  
## `education.fctr 5th-6th`                         -1.492 0.135688    
## `education.fctr 10th`                            -2.626 0.008635 ** 
## `education.fctr 1st-4th`                          0.000 0.999676    
## `education.fctr Preschool`                        0.000 0.999833    
## `education.fctr 12th`                            -0.007 0.994550    
## `nativecountry.fctr Cuba`                         1.908 0.056379 .  
## `nativecountry.fctr Jamaica`                      0.130 0.896757    
## `nativecountry.fctr India`                       -0.843 0.399391    
## `nativecountry.fctr Mexico`                      -0.008 0.993670    
## `nativecountry.fctr South`                        0.000 0.999976    
## `nativecountry.fctr Puerto-Rico`                 -0.002 0.998153    
## `nativecountry.fctr Honduras`                        NA       NA    
## `nativecountry.fctr England`                      0.964 0.334943    
## `nativecountry.fctr Canada`                       1.056 0.291095    
## `nativecountry.fctr Germany`                     -0.004 0.996508    
## `nativecountry.fctr Iran`                            NA       NA    
## `nativecountry.fctr Philippines`                 -0.805 0.420791    
## `nativecountry.fctr Italy`                       -0.001 0.998826    
## `nativecountry.fctr Poland`                      -0.002 0.998740    
## `nativecountry.fctr Columbia`                    -0.004 0.996746    
## `nativecountry.fctr Cambodia`                        NA       NA    
## `nativecountry.fctr Thailand`                        NA       NA    
## `nativecountry.fctr Ecuador`                     -0.001 0.999169    
## `nativecountry.fctr Laos`                         0.258 0.796765    
## `nativecountry.fctr Taiwan`                       0.002 0.998006    
## `nativecountry.fctr Haiti`                       -0.001 0.999054    
## `nativecountry.fctr Portugal`                    -0.002 0.998770    
## `nativecountry.fctr Dominican-Republic`              NA       NA    
## `nativecountry.fctr El-Salvador`                 -0.005 0.996341    
## `nativecountry.fctr France`                      -0.003 0.997383    
## `nativecountry.fctr Guatemala`                    0.001 0.998945    
## `nativecountry.fctr China`                       -1.386 0.165625    
## `nativecountry.fctr Japan`                       -0.002 0.998754    
## `nativecountry.fctr Yugoslavia`                      NA       NA    
## `nativecountry.fctr Peru`                        -0.001 0.999048    
## `nativecountry.fctr Outlying-US(Guam-USVI-etc)`      NA       NA    
## `nativecountry.fctr Scotland`                    -0.002 0.998775    
## `nativecountry.fctr Trinadad&Tobago`             -0.002 0.998628    
## `nativecountry.fctr Greece`                      -0.001 0.998907    
## `nativecountry.fctr Nicaragua`                   -0.002 0.998705    
## `nativecountry.fctr Vietnam`                      0.000 0.999888    
## `nativecountry.fctr Hong`                        -0.002 0.998585    
## `nativecountry.fctr Ireland`                     -0.002 0.998572    
## `nativecountry.fctr Hungary`                         NA       NA    
## `nativecountry.fctr Holand-Netherlands`              NA       NA    
## `occupation.fctr Exec-managerial`                 2.634 0.008448 ** 
## `occupation.fctr Handlers-cleaners`               0.033 0.973816    
## `occupation.fctr Prof-specialty`                  2.220 0.026412 *  
## `occupation.fctr Other-service`                  -1.786 0.074026 .  
## `occupation.fctr Sales`                           2.045 0.040865 *  
## `occupation.fctr Transport-moving`                0.241 0.809342    
## `occupation.fctr Farming-fishing`                -0.009 0.992542    
## `occupation.fctr Machine-op-inspct`               0.592 0.553673    
## `occupation.fctr Tech-support`                    2.658 0.007851 ** 
## `occupation.fctr Craft-repair`                    0.796 0.425807    
## `occupation.fctr ?`                                  NA       NA    
## `occupation.fctr Protective-serv`                -0.846 0.397550    
## `occupation.fctr Armed-Forces`                       NA       NA    
## `occupation.fctr Priv-house-serv`                -0.003 0.997300    
## `relationship.fctr Husband`                       0.001 0.999507    
## `relationship.fctr Wife`                          0.001 0.999334    
## `relationship.fctr Own-child`                    -0.898 0.369177    
## `relationship.fctr Unmarried`                     2.183 0.029005 *  
## `relationship.fctr Other-relative`               -0.006 0.995180    
## `sex.fctr Female`                                -2.180 0.029227 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1104.46  on 999  degrees of freedom
## Residual deviance:  545.16  on 913  degrees of freedom
## AIC: 719.16
## 
## Number of Fisher Scoring iterations: 18
```

```
## [1] TRUE
```

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "model.selected")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0
```

![](USCensus_Earnings_files/figure-html/fit.models_1-8.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.data.training.all", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed9             fit.models                5                1  98.841
## elapsed10 fit.data.training.all                6                0 110.406
```

## Step `6`: fit.data.training.all

```r
if (!is.null(glb_fin_mdl_id) && (glb_fin_mdl_id %in% names(glb_models_lst))) {
    warning("Final model same as user selected model")
    glb_fin_mdl <- glb_sel_mdl
} else {    
    print(mdl_feats_df <- myextract_mdl_feats(sel_mdl=glb_sel_mdl, entity_df=glb_trnent_df))
    
    if ((model_method <- glb_sel_mdl$method) == "custom")
        # get actual method from the model_id
        model_method <- tail(unlist(strsplit(glb_sel_mdl_id, "[.]")), 1)
        
    # Sync with parameters in mydsutils.R
    ret_lst <- myfit_mdl(model_id="Final", model_method=model_method,
                            indep_vars_vctr=mdl_feats_df$id, model_type=glb_model_type,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out, 
                            fit_df=glb_trnent_df, OOB_df=NULL,
                         # Automate from here
                         #  Issues if glb_sel_mdl$method == "rf" b/c trainControl is "oob"; not "cv"
                         n_cv_folds=glb_n_cv_folds, tune_models_df=NULL,
                            model_loss_mtrx=glb_model_metric_terms,
                            model_summaryFunction=glb_sel_mdl$control$summaryFunction,
                            model_metric=glb_sel_mdl$metric,
                            model_metric_maximize=glb_sel_mdl$maximize)
    glb_fin_mdl <- glb_models_lst[[length(glb_models_lst)]] 
    glb_fin_mdl_id <- glb_models_df[length(glb_models_lst), "model_id"]
}
```

```
##                                    id importance
## capitalgain               capitalgain 100.000000
## education.fctr         education.fctr  77.459886
## age                               age  59.871191
## workclass.fctr              race.fctr  54.648029
## hoursperweek             hoursperweek  48.274918
## occupation.fctr     relationship.fctr  45.387086
## relationship.fctr  nativecountry.fctr  37.277684
## sex.fctr           maritalstatus.fctr  37.226377
## nativecountry.fctr    occupation.fctr  32.576969
## maritalstatus.fctr     workclass.fctr  21.237208
## capitalloss               capitalloss  15.890756
## race.fctr                    sex.fctr  15.794747
## .rnorm                         .rnorm   4.511675
## [1] "fitting model: Final.glm"
## [1] "    indep_vars: capitalgain, education.fctr, age, race.fctr, hoursperweek, relationship.fctr, nativecountry.fctr, maritalstatus.fctr, occupation.fctr, workclass.fctr, capitalloss, sex.fctr, .rnorm"
## + Fold1: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold1: parameter=none 
## + Fold2: parameter=none
```

```
## Warning: glm.fit: algorithm did not converge
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold2: parameter=none 
## + Fold3: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning: not plotting observations with leverage one:
##   118, 160, 182, 194, 239, 293, 371, 422, 477, 518, 535, 638, 739, 798, 805, 808, 811, 921
```

![](USCensus_Earnings_files/figure-html/fit.data.training.all_0-1.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_0-2.png) 

```
## Warning: not plotting observations with leverage one:
##   118, 160, 182, 194, 239, 293, 371, 422, 477, 518, 535, 638, 739, 798, 805, 808, 811, 921
```

![](USCensus_Earnings_files/figure-html/fit.data.training.all_0-3.png) 

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##      Min        1Q    Median        3Q       Max  
## -2.24059  -0.43840  -0.08362   0.00000   2.62797  
## 
## Coefficients: (12 not defined because of singularities)
##                                                   Estimate Std. Error
## (Intercept)                                     -6.821e+00  1.159e+00
## capitalgain                                      4.161e-04  7.105e-05
## `education.fctr HS-grad`                        -1.559e+00  3.436e-01
## `education.fctr 11th`                           -1.815e+01  2.222e+03
## `education.fctr Masters`                        -2.940e-02  4.506e-01
## `education.fctr 9th`                            -1.666e+00  1.406e+00
## `education.fctr Some-college`                   -8.444e-01  3.563e-01
## `education.fctr Assoc-acdm`                     -6.287e-01  7.288e-01
## `education.fctr 7th-8th`                        -2.846e+00  1.162e+00
## `education.fctr Doctorate`                       1.534e+00  9.430e-01
## `education.fctr Assoc-voc`                      -8.237e-01  5.183e-01
## `education.fctr Prof-school`                     1.781e+00  9.115e-01
## `education.fctr 5th-6th`                        -2.305e+00  1.545e+00
## `education.fctr 10th`                           -1.951e+00  7.429e-01
## `education.fctr 1st-4th`                        -2.408e+00  5.929e+03
## `education.fctr Preschool`                      -2.289e+00  1.097e+04
## `education.fctr 12th`                           -1.821e+01  2.665e+03
## age                                              3.818e-02  1.089e-02
## `race.fctr Black`                                4.090e-01  4.460e-01
## `race.fctr Asian-Pac-Islander`                   1.380e+00  1.491e+00
## `race.fctr Amer-Indian-Eskimo`                   5.347e-01  1.206e+00
## `race.fctr Other`                                1.267e+00  1.487e+00
## hoursperweek                                     2.970e-02  1.051e-02
## `relationship.fctr Husband`                      4.541e+00  7.345e+03
## `relationship.fctr Wife`                         6.133e+00  7.345e+03
## `relationship.fctr Own-child`                   -9.141e-01  1.018e+00
## `relationship.fctr Unmarried`                    1.153e+00  5.278e-01
## `relationship.fctr Other-relative`              -1.367e+01  2.263e+03
## `nativecountry.fctr Cuba`                        2.854e+00  1.496e+00
## `nativecountry.fctr Jamaica`                     1.769e-01  1.363e+00
## `nativecountry.fctr India`                      -1.501e+00  1.781e+00
## `nativecountry.fctr Mexico`                     -1.704e+01  2.148e+03
## `nativecountry.fctr South`                       3.292e-01  1.099e+04
## `nativecountry.fctr Puerto-Rico`                -1.631e+01  7.046e+03
## `nativecountry.fctr Honduras`                           NA         NA
## `nativecountry.fctr England`                     9.965e-01  1.034e+00
## `nativecountry.fctr Canada`                      2.268e+00  2.148e+00
## `nativecountry.fctr Germany`                    -1.790e+01  4.090e+03
## `nativecountry.fctr Iran`                               NA         NA
## `nativecountry.fctr Philippines`                -1.854e+00  2.302e+00
## `nativecountry.fctr Italy`                      -1.582e+01  1.075e+04
## `nativecountry.fctr Poland`                     -1.698e+01  1.075e+04
## `nativecountry.fctr Columbia`                   -1.694e+01  4.154e+03
## `nativecountry.fctr Cambodia`                           NA         NA
## `nativecountry.fctr Thailand`                           NA         NA
## `nativecountry.fctr Ecuador`                    -1.120e+01  1.075e+04
## `nativecountry.fctr Laos`                        5.683e-01  2.207e+00
## `nativecountry.fctr Taiwan`                      1.847e+01  7.390e+03
## `nativecountry.fctr Haiti`                      -1.274e+01  1.075e+04
## `nativecountry.fctr Portugal`                   -1.658e+01  1.075e+04
## `nativecountry.fctr Dominican-Republic`                 NA         NA
## `nativecountry.fctr El-Salvador`                -1.699e+01  3.705e+03
## `nativecountry.fctr France`                     -2.093e+01  6.381e+03
## `nativecountry.fctr Guatemala`                   1.553e+01  1.175e+04
## `nativecountry.fctr China`                      -2.950e+00  2.128e+00
## `nativecountry.fctr Japan`                      -1.679e+01  1.075e+04
## `nativecountry.fctr Yugoslavia`                         NA         NA
## `nativecountry.fctr Peru`                       -1.283e+01  1.075e+04
## `nativecountry.fctr Outlying-US(Guam-USVI-etc)`         NA         NA
## `nativecountry.fctr Scotland`                   -1.651e+01  1.075e+04
## `nativecountry.fctr Trinadad&Tobago`            -1.849e+01  1.075e+04
## `nativecountry.fctr Greece`                     -1.473e+01  1.075e+04
## `nativecountry.fctr Nicaragua`                  -1.745e+01  1.075e+04
## `nativecountry.fctr Vietnam`                    -1.078e+00  7.665e+03
## `nativecountry.fctr Hong`                       -1.907e+01  1.075e+04
## `nativecountry.fctr Ireland`                    -1.925e+01  1.075e+04
## `nativecountry.fctr Hungary`                            NA         NA
## `nativecountry.fctr Holand-Netherlands`                 NA         NA
## `maritalstatus.fctr Married-civ-spouse`         -1.878e+00  7.345e+03
## `maritalstatus.fctr Divorced`                    4.708e-01  5.824e-01
## `maritalstatus.fctr Married-spouse-absent`      -1.558e+01  2.722e+03
## `maritalstatus.fctr Separated`                   4.013e-01  8.565e-01
## `maritalstatus.fctr Married-AF-spouse`          -1.751e+01  1.302e+04
## `maritalstatus.fctr Widowed`                     1.088e+00  8.744e-01
## `occupation.fctr Exec-managerial`                1.304e+00  4.950e-01
## `occupation.fctr Handlers-cleaners`              2.694e-02  8.208e-01
## `occupation.fctr Prof-specialty`                 1.153e+00  5.194e-01
## `occupation.fctr Other-service`                 -2.126e+00  1.190e+00
## `occupation.fctr Sales`                          1.069e+00  5.227e-01
## `occupation.fctr Transport-moving`               1.505e-01  6.237e-01
## `occupation.fctr Farming-fishing`               -1.718e+01  1.838e+03
## `occupation.fctr Machine-op-inspct`              3.740e-01  6.315e-01
## `occupation.fctr Tech-support`                   1.914e+00  7.198e-01
## `occupation.fctr Craft-repair`                   4.127e-01  5.182e-01
## `occupation.fctr ?`                              6.723e-01  9.573e-01
## `occupation.fctr Protective-serv`               -1.078e+00  1.275e+00
## `occupation.fctr Armed-Forces`                          NA         NA
## `occupation.fctr Priv-house-serv`               -1.410e+01  4.167e+03
## `workclass.fctr Self-emp-not-inc`                1.610e-01  7.478e-01
## `workclass.fctr Private`                         1.125e+00  6.789e-01
## `workclass.fctr Federal-gov`                     2.904e+00  9.072e-01
## `workclass.fctr Local-gov`                       3.428e-01  7.691e-01
## `workclass.fctr ?`                                      NA         NA
## `workclass.fctr Self-emp-inc`                    1.042e+00  8.233e-01
## `workclass.fctr Without-pay`                    -1.725e+01  1.075e+04
## `workclass.fctr Never-worked`                           NA         NA
## capitalloss                                      2.194e-04  2.357e-04
## `sex.fctr Female`                               -1.083e+00  4.967e-01
## .rnorm                                          -3.095e-02  1.171e-01
##                                                 z value Pr(>|z|)    
## (Intercept)                                      -5.886 3.95e-09 ***
## capitalgain                                       5.857 4.71e-09 ***
## `education.fctr HS-grad`                         -4.537 5.71e-06 ***
## `education.fctr 11th`                            -0.008 0.993482    
## `education.fctr Masters`                         -0.065 0.947977    
## `education.fctr 9th`                             -1.185 0.236034    
## `education.fctr Some-college`                    -2.370 0.017790 *  
## `education.fctr Assoc-acdm`                      -0.863 0.388343    
## `education.fctr 7th-8th`                         -2.450 0.014284 *  
## `education.fctr Doctorate`                        1.627 0.103707    
## `education.fctr Assoc-voc`                       -1.589 0.112020    
## `education.fctr Prof-school`                      1.954 0.050691 .  
## `education.fctr 5th-6th`                         -1.492 0.135688    
## `education.fctr 10th`                            -2.626 0.008635 ** 
## `education.fctr 1st-4th`                          0.000 0.999676    
## `education.fctr Preschool`                        0.000 0.999833    
## `education.fctr 12th`                            -0.007 0.994550    
## age                                               3.507 0.000454 ***
## `race.fctr Black`                                 0.917 0.359078    
## `race.fctr Asian-Pac-Islander`                    0.925 0.354892    
## `race.fctr Amer-Indian-Eskimo`                    0.443 0.657450    
## `race.fctr Other`                                 0.852 0.394420    
## hoursperweek                                      2.828 0.004691 ** 
## `relationship.fctr Husband`                       0.001 0.999507    
## `relationship.fctr Wife`                          0.001 0.999334    
## `relationship.fctr Own-child`                    -0.898 0.369177    
## `relationship.fctr Unmarried`                     2.183 0.029005 *  
## `relationship.fctr Other-relative`               -0.006 0.995180    
## `nativecountry.fctr Cuba`                         1.908 0.056379 .  
## `nativecountry.fctr Jamaica`                      0.130 0.896757    
## `nativecountry.fctr India`                       -0.843 0.399391    
## `nativecountry.fctr Mexico`                      -0.008 0.993670    
## `nativecountry.fctr South`                        0.000 0.999976    
## `nativecountry.fctr Puerto-Rico`                 -0.002 0.998153    
## `nativecountry.fctr Honduras`                        NA       NA    
## `nativecountry.fctr England`                      0.964 0.334943    
## `nativecountry.fctr Canada`                       1.056 0.291095    
## `nativecountry.fctr Germany`                     -0.004 0.996508    
## `nativecountry.fctr Iran`                            NA       NA    
## `nativecountry.fctr Philippines`                 -0.805 0.420791    
## `nativecountry.fctr Italy`                       -0.001 0.998826    
## `nativecountry.fctr Poland`                      -0.002 0.998740    
## `nativecountry.fctr Columbia`                    -0.004 0.996746    
## `nativecountry.fctr Cambodia`                        NA       NA    
## `nativecountry.fctr Thailand`                        NA       NA    
## `nativecountry.fctr Ecuador`                     -0.001 0.999169    
## `nativecountry.fctr Laos`                         0.258 0.796765    
## `nativecountry.fctr Taiwan`                       0.002 0.998006    
## `nativecountry.fctr Haiti`                       -0.001 0.999054    
## `nativecountry.fctr Portugal`                    -0.002 0.998770    
## `nativecountry.fctr Dominican-Republic`              NA       NA    
## `nativecountry.fctr El-Salvador`                 -0.005 0.996341    
## `nativecountry.fctr France`                      -0.003 0.997383    
## `nativecountry.fctr Guatemala`                    0.001 0.998945    
## `nativecountry.fctr China`                       -1.386 0.165625    
## `nativecountry.fctr Japan`                       -0.002 0.998754    
## `nativecountry.fctr Yugoslavia`                      NA       NA    
## `nativecountry.fctr Peru`                        -0.001 0.999048    
## `nativecountry.fctr Outlying-US(Guam-USVI-etc)`      NA       NA    
## `nativecountry.fctr Scotland`                    -0.002 0.998775    
## `nativecountry.fctr Trinadad&Tobago`             -0.002 0.998628    
## `nativecountry.fctr Greece`                      -0.001 0.998907    
## `nativecountry.fctr Nicaragua`                   -0.002 0.998705    
## `nativecountry.fctr Vietnam`                      0.000 0.999888    
## `nativecountry.fctr Hong`                        -0.002 0.998585    
## `nativecountry.fctr Ireland`                     -0.002 0.998572    
## `nativecountry.fctr Hungary`                         NA       NA    
## `nativecountry.fctr Holand-Netherlands`              NA       NA    
## `maritalstatus.fctr Married-civ-spouse`           0.000 0.999796    
## `maritalstatus.fctr Divorced`                     0.808 0.418880    
## `maritalstatus.fctr Married-spouse-absent`       -0.006 0.995434    
## `maritalstatus.fctr Separated`                    0.469 0.639391    
## `maritalstatus.fctr Married-AF-spouse`           -0.001 0.998927    
## `maritalstatus.fctr Widowed`                      1.244 0.213532    
## `occupation.fctr Exec-managerial`                 2.634 0.008448 ** 
## `occupation.fctr Handlers-cleaners`               0.033 0.973816    
## `occupation.fctr Prof-specialty`                  2.220 0.026412 *  
## `occupation.fctr Other-service`                  -1.786 0.074026 .  
## `occupation.fctr Sales`                           2.045 0.040865 *  
## `occupation.fctr Transport-moving`                0.241 0.809342    
## `occupation.fctr Farming-fishing`                -0.009 0.992542    
## `occupation.fctr Machine-op-inspct`               0.592 0.553673    
## `occupation.fctr Tech-support`                    2.658 0.007851 ** 
## `occupation.fctr Craft-repair`                    0.796 0.425807    
## `occupation.fctr ?`                               0.702 0.482461    
## `occupation.fctr Protective-serv`                -0.846 0.397550    
## `occupation.fctr Armed-Forces`                       NA       NA    
## `occupation.fctr Priv-house-serv`                -0.003 0.997300    
## `workclass.fctr Self-emp-not-inc`                 0.215 0.829567    
## `workclass.fctr Private`                          1.657 0.097516 .  
## `workclass.fctr Federal-gov`                      3.201 0.001370 ** 
## `workclass.fctr Local-gov`                        0.446 0.655834    
## `workclass.fctr ?`                                   NA       NA    
## `workclass.fctr Self-emp-inc`                     1.265 0.205836    
## `workclass.fctr Without-pay`                     -0.002 0.998720    
## `workclass.fctr Never-worked`                        NA       NA    
## capitalloss                                       0.931 0.351975    
## `sex.fctr Female`                                -2.180 0.029227 *  
## .rnorm                                           -0.264 0.791562    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1104.46  on 999  degrees of freedom
## Residual deviance:  545.16  on 913  degrees of freedom
## AIC: 719.16
## 
## Number of Fisher Scoring iterations: 18
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](USCensus_Earnings_files/figure-html/fit.data.training.all_0-4.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_0-5.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 759 241
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                                0
## 2            Y                                0
##   over50k.fctr.predict.Final.glm.Y
## 1                              759
## 2                              241
##           Reference
## Prediction   N   Y
##          N 523   9
##          Y 236 232
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              523
## 2            Y                                9
##   over50k.fctr.predict.Final.glm.Y
## 1                              236
## 2                              232
##           Reference
## Prediction   N   Y
##          N 599  25
##          Y 160 216
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              599
## 2            Y                               25
##   over50k.fctr.predict.Final.glm.Y
## 1                              160
## 2                              216
##           Reference
## Prediction   N   Y
##          N 656  41
##          Y 103 200
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              656
## 2            Y                               41
##   over50k.fctr.predict.Final.glm.Y
## 1                              103
## 2                              200
##           Reference
## Prediction   N   Y
##          N 684  57
##          Y  75 184
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              684
## 2            Y                               57
##   over50k.fctr.predict.Final.glm.Y
## 1                               75
## 2                              184
##           Reference
## Prediction   N   Y
##          N 701  82
##          Y  58 159
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              701
## 2            Y                               82
##   over50k.fctr.predict.Final.glm.Y
## 1                               58
## 2                              159
##           Reference
## Prediction   N   Y
##          N 728  96
##          Y  31 145
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              728
## 2            Y                               96
##   over50k.fctr.predict.Final.glm.Y
## 1                               31
## 2                              145
##           Reference
## Prediction   N   Y
##          N 741 123
##          Y  18 118
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              741
## 2            Y                              123
##   over50k.fctr.predict.Final.glm.Y
## 1                               18
## 2                              118
##           Reference
## Prediction   N   Y
##          N 753 158
##          Y   6  83
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              753
## 2            Y                              158
##   over50k.fctr.predict.Final.glm.Y
## 1                                6
## 2                               83
##           Reference
## Prediction   N   Y
##          N 758 183
##          Y   1  58
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              758
## 2            Y                              183
##   over50k.fctr.predict.Final.glm.Y
## 1                                1
## 2                               58
##           Reference
## Prediction   N   Y
##          N 759 241
##          Y   0   0
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              759
## 2            Y                              241
##   over50k.fctr.predict.Final.glm.Y
## 1                                0
## 2                                0
##    threshold   f.score
## 1        0.0 0.3883965
## 2        0.1 0.6544429
## 3        0.2 0.7001621
## 4        0.3 0.7352941
## 5        0.4 0.7360000
## 6        0.5 0.6943231
## 7        0.6 0.6954436
## 8        0.7 0.6259947
## 9        0.8 0.5030303
## 10       0.9 0.3866667
## 11       1.0 0.0000000
```

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.fit"
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              684
## 2            Y                               57
##   over50k.fctr.predict.Final.glm.Y
## 1                               75
## 2                              184
##           Reference
## Prediction   N   Y
##          N 684  57
##          Y  75 184
##   over50k.fctr over50k.fctr.predict.Final.glm.N
## 1            N                              684
## 2            Y                               57
##   over50k.fctr.predict.Final.glm.Y
## 1                               75
## 2                              184
##          Prediction
## Reference   N   Y
##         N 684  75
##         Y  57 184
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.680000e-01   6.481520e-01   8.454346e-01   8.883720e-01   7.590000e-01 
## AccuracyPValue  McnemarPValue 
##   5.968845e-18   1.389640e-01
```

```
## Warning in mypredict_mdl(mdl, df = fit_df, rsp_var, rsp_var_out,
## model_id_method, : Expecting 1 metric: Accuracy; recd: Accuracy, Kappa;
## retaining Accuracy only
```

![](USCensus_Earnings_files/figure-html/fit.data.training.all_0-6.png) 

```
##    model_id model_method
## 1 Final.glm          glm
##                                                                                                                                                                                  feats
## 1 capitalgain, education.fctr, age, race.fctr, hoursperweek, relationship.fctr, nativecountry.fctr, maritalstatus.fctr, occupation.fctr, workclass.fctr, capitalloss, sex.fctr, .rnorm
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                      2.535                 0.502
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.9325658                    0.4           0.736        0.8120306
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit min.aic.fit
## 1             0.8454346              0.888372     0.5100264    719.1617
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.05885356       0.1288817
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.data.training.all", 
    chunk_step_major=glb_script_df[nrow(glb_script_df), "chunk_step_major"], 
    chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed10 fit.data.training.all                6                0 110.406
## elapsed11 fit.data.training.all                6                1 117.134
```


```r
glb_rsp_var_out <- paste0(glb_rsp_var_out, tail(names(glb_models_lst), 1))

# Used again in predict.data.new chunk
glb_get_predictions <- function(df) {
    if (glb_is_regression) {
        df[, glb_rsp_var_out] <- predict(glb_fin_mdl, newdata=df, type="raw")
        print(myplot_scatter(df, glb_rsp_var, glb_rsp_var_out, 
                             smooth=TRUE))
        df[, paste0(glb_rsp_var_out, ".err")] <- 
            abs(df[, glb_rsp_var_out] - df[, glb_rsp_var])
        print(head(orderBy(reformulate(c("-", paste0(glb_rsp_var_out, ".err"))), 
                           df)))                             
    }

    if (glb_is_classification && glb_is_binomial) {
        # incorporate glb_clf_proba_threshold
        #   shd it only be for glb_fin_mdl or for earlier models ?
        if ((prob_threshold <- 
                glb_models_df[glb_models_df$model_id == glb_fin_mdl_id, "opt.prob.threshold.fit"]) != 
            glb_models_df[glb_models_df$model_id == glb_sel_mdl_id, "opt.prob.threshold.fit"])
            stop("user specification for probability threshold required")
        
        df[, paste0(glb_rsp_var_out, ".prob")] <- 
            predict(glb_fin_mdl, newdata=df, type="prob")[, 2]
        df[, glb_rsp_var_out] <- 
    			factor(levels(df[, glb_rsp_var])[
    				(df[, paste0(glb_rsp_var_out, ".prob")] >=
    					prob_threshold) * 1 + 1], levels(df[, glb_rsp_var]))
    
        # prediction stats already reported by myfit_mdl ???
    }    
    
    if (glb_is_classification && !glb_is_binomial) {
        df[, glb_rsp_var_out] <- predict(glb_fin_mdl, newdata=df, type="raw")
    }

    return(df)
}    
glb_trnent_df <- glb_get_predictions(glb_trnent_df)
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```r
print(glb_feats_df <- mymerge_feats_importance(feats_df=glb_feats_df, sel_mdl=glb_fin_mdl, 
                                               entity_df=glb_trnent_df))
```

```
##                    id       cor.y exclude.as.feat  cor.y.abs cor.low
## 3         capitalgain  0.23222327               0 0.23222327       1
## 5      education.fctr -0.06153968               0 0.06153968       1
## 2                 age  0.24780278               0 0.24780278       1
## 10          race.fctr -0.01730292               0 0.01730292       1
## 6        hoursperweek  0.22402106               0 0.22402106       1
## 11  relationship.fctr -0.12845800               0 0.12845800       1
## 8  nativecountry.fctr -0.06238053               0 0.06238053       1
## 7  maritalstatus.fctr  0.01310876               0 0.01310876       1
## 9     occupation.fctr -0.12686813               0 0.12686813       1
## 13     workclass.fctr  0.07835390               0 0.07835390       1
## 4         capitalloss  0.10182189               0 0.10182189       1
## 12           sex.fctr -0.20283667               0 0.20283667       1
## 1              .rnorm -0.01983209               0 0.01983209       1
##    importance
## 3  100.000000
## 5   77.459886
## 2   59.871191
## 10  54.648029
## 6   48.274918
## 11  45.387086
## 8   37.277684
## 7   37.226377
## 9   32.576969
## 13  21.237208
## 4   15.890756
## 12  15.794747
## 1    4.511675
```

```r
# Used again in predict.data.new chunk
glb_analytics_diag_plots <- function(obs_df) {
    for (var in subset(glb_feats_df, !is.na(importance))$id) {
        plot_df <- melt(obs_df, id.vars=var, 
                        measure.vars=c(glb_rsp_var, glb_rsp_var_out))
#         if (var == "<feat_name>") print(myplot_scatter(plot_df, var, "value", 
#                                              facet_colcol_name="variable") + 
#                       geom_vline(xintercept=<divider_val>, linetype="dotted")) else     
            print(myplot_scatter(plot_df, var, "value", colorcol_name="variable",
                                 facet_colcol_name="variable", jitter=TRUE) + 
                      guides(color=FALSE))
    }
    
    if (glb_is_regression) {
        plot_vars_df <- subset(glb_feats_df, importance > glb_feats_df[glb_feats_df$id == ".rnorm", "importance"])
        print(myplot_prediction_regression(df=obs_df, 
                    feat_x=ifelse(nrow(plot_vars_df) > 1, plot_vars_df$id[2], ".rownames"), 
                                           feat_y=plot_vars_df$id[1],
                    rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                    id_vars=glb_id_vars)
#               + facet_wrap(reformulate(plot_vars_df$id[2])) # if [1 or 2] is a factor                                                         
#               + geom_point(aes_string(color="<col_name>.fctr")) #  to color the plot
              )
    }    
    
    if (glb_is_classification) {
        if (nrow(plot_vars_df <- subset(glb_feats_df, !is.na(importance))) == 0)
            warning("No features in selected model are statistically important")
        else print(myplot_prediction_classification(df=obs_df, 
                feat_x=ifelse(nrow(plot_vars_df) > 1, plot_vars_df$id[2], 
                              ".rownames"),
                                               feat_y=plot_vars_df$id[1],
                     rsp_var=glb_rsp_var, 
                     rsp_var_out=glb_rsp_var_out, 
                     id_vars=glb_id_vars)
#               + geom_hline(yintercept=<divider_val>, linetype = "dotted")
                )
    }    
}
glb_analytics_diag_plots(obs_df=glb_trnent_df)
```

![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-1.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-2.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-3.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-4.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-5.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-6.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-7.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-8.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-9.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-10.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-11.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-12.png) ![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-13.png) 

```
##      age         workclass    education       maritalstatus
## 6     37           Private      Masters  Married-civ-spouse
## 355   43           Private    Bachelors  Married-civ-spouse
## 418   36       Federal-gov      Masters       Never-married
## 1250  51         Local-gov    Bachelors  Married-civ-spouse
## 1806  38           Private    Bachelors  Married-civ-spouse
## 2388  61  Self-emp-not-inc      HS-grad  Married-civ-spouse
## 5367  44           Private  Prof-school  Married-civ-spouse
##             occupation   relationship   race     sex capitalgain
## 6      Exec-managerial           Wife  White  Female           0
## 355   Transport-moving        Husband  White    Male           0
## 418     Prof-specialty  Not-in-family  White    Male           0
## 1250      Adm-clerical        Husband  White    Male           0
## 1806      Craft-repair           Wife  White  Female           0
## 2388      Craft-repair        Husband  White    Male           0
## 5367    Prof-specialty        Husband  White    Male       99999
##      capitalloss hoursperweek  nativecountry over50k    workclass.fctr
## 6              0           40  United-States   <=50K           Private
## 355            0           45  United-States   <=50K           Private
## 418         1408           40  United-States   <=50K       Federal-gov
## 1250           0           40  United-States    >50K         Local-gov
## 1806           0           40  United-States   <=50K           Private
## 2388           0           52  United-States    >50K  Self-emp-not-inc
## 5367           0           65  United-States    >50K           Private
##      education.fctr  maritalstatus.fctr   occupation.fctr
## 6           Masters  Married-civ-spouse   Exec-managerial
## 355       Bachelors  Married-civ-spouse  Transport-moving
## 418         Masters       Never-married    Prof-specialty
## 1250      Bachelors  Married-civ-spouse      Adm-clerical
## 1806      Bachelors  Married-civ-spouse      Craft-repair
## 2388        HS-grad  Married-civ-spouse      Craft-repair
## 5367    Prof-school  Married-civ-spouse    Prof-specialty
##      relationship.fctr race.fctr sex.fctr nativecountry.fctr     .rnorm
## 6                 Wife     White   Female      United-States  1.5640010
## 355            Husband     White     Male      United-States -1.5273574
## 418      Not-in-family     White     Male      United-States -0.2095156
## 1250           Husband     White     Male      United-States -0.5379833
## 1806              Wife     White   Female      United-States  1.1681171
## 2388           Husband     White     Male      United-States -1.5651562
## 5367           Husband     White     Male      United-States -0.8259619
##      over50k.fctr over50k.fctr.predict.Final.glm.prob
## 6               N                           0.7861746
## 355             N                           0.5357181
## 418             N                           0.5211014
## 1250            Y                           0.3400039
## 1806            N                           0.6203245
## 2388            Y                           0.2277406
## 5367            Y                           1.0000000
##      over50k.fctr.predict.Final.glm
## 6                                 Y
## 355                               Y
## 418                               Y
## 1250                              N
## 1806                              Y
## 2388                              N
## 5367                              Y
##      over50k.fctr.predict.Final.glm.accurate .label
## 6                                      FALSE     .6
## 355                                    FALSE   .355
## 418                                    FALSE   .418
## 1250                                   FALSE  .1250
## 1806                                   FALSE  .1806
## 2388                                   FALSE  .2388
## 5367                                    TRUE  .5367
```

![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-14.png) 

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all.prediction","model.final")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1
```

![](USCensus_Earnings_files/figure-html/fit.data.training.all_1-15.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="predict.data.new", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed11 fit.data.training.all                6                1 117.134
## elapsed12      predict.data.new                7                0 127.038
```

## Step `7`: predict data.new

```r
# Compute final model predictions
glb_newent_df <- glb_get_predictions(glb_newent_df)
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```r
glb_analytics_diag_plots(obs_df=glb_newent_df)
```

![](USCensus_Earnings_files/figure-html/predict.data.new-1.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-2.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-3.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-4.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-5.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-6.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-7.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-8.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-9.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-10.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-11.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-12.png) ![](USCensus_Earnings_files/figure-html/predict.data.new-13.png) 

```
##       age         workclass     education       maritalstatus
## 31     20           Private  Some-college       Never-married
## 117    40           Private     Bachelors  Married-civ-spouse
## 608    52      Self-emp-inc     Bachelors  Married-civ-spouse
## 810    57           Private       HS-grad             Widowed
## 1311   40           Private       HS-grad  Married-civ-spouse
## 1462   32           Private  Some-college  Married-civ-spouse
## 1468   32           Private          11th            Divorced
## 23048  42  Self-emp-not-inc       Masters  Married-civ-spouse
##              occupation   relationship   race   sex capitalgain
## 31                Sales      Own-child  Black  Male           0
## 117     Exec-managerial        Husband  White  Male           0
## 608               Sales        Husband  White  Male           0
## 810    Transport-moving      Unmarried  White  Male           0
## 1311    Exec-managerial        Husband  White  Male           0
## 1462              Sales        Husband  White  Male           0
## 1468              Sales  Not-in-family  White  Male           0
## 23048             Sales        Husband  White  Male       99999
##       capitalloss hoursperweek  nativecountry over50k    workclass.fctr
## 31              0           44  United-States   <=50K           Private
## 117             0           60  United-States   <=50K           Private
## 608             0           55  United-States   <=50K      Self-emp-inc
## 810           653           42  United-States    >50K           Private
## 1311            0           40  United-States    >50K           Private
## 1462            0           48  United-States   <=50K           Private
## 1468            0           43  United-States    >50K           Private
## 23048           0           80  United-States    >50K  Self-emp-not-inc
##       education.fctr  maritalstatus.fctr   occupation.fctr
## 31      Some-college       Never-married             Sales
## 117        Bachelors  Married-civ-spouse   Exec-managerial
## 608        Bachelors  Married-civ-spouse             Sales
## 810          HS-grad             Widowed  Transport-moving
## 1311         HS-grad  Married-civ-spouse   Exec-managerial
## 1462    Some-college  Married-civ-spouse             Sales
## 1468            11th            Divorced             Sales
## 23048        Masters  Married-civ-spouse             Sales
##       relationship.fctr race.fctr sex.fctr nativecountry.fctr      .rnorm
## 31            Own-child     Black     Male      United-States -0.08837181
## 117             Husband     White     Male      United-States -3.11383046
## 608             Husband     White     Male      United-States  0.62036514
## 810           Unmarried     White     Male      United-States -1.73102184
## 1311            Husband     White     Male      United-States  1.82063036
## 1462            Husband     White     Male      United-States  0.44603769
## 1468      Not-in-family     White     Male      United-States  0.13299383
## 23048           Husband     White     Male      United-States -0.78729784
##       over50k.fctr over50k.fctr.predict.Final.glm.prob
## 31               N                        1.977742e-02
## 117              N                        8.424240e-01
## 608              N                        8.252479e-01
## 810              Y                        2.238346e-01
## 1311             Y                        3.477002e-01
## 1462             N                        4.564036e-01
## 1468             Y                        2.486192e-09
## 23048            Y                        1.000000e+00
##       over50k.fctr.predict.Final.glm
## 31                                 N
## 117                                Y
## 608                                Y
## 810                                N
## 1311                               N
## 1462                               Y
## 1468                               N
## 23048                              Y
##       over50k.fctr.predict.Final.glm.accurate .label
## 31                                       TRUE    .31
## 117                                     FALSE   .117
## 608                                     FALSE   .608
## 810                                     FALSE   .810
## 1311                                    FALSE  .1311
## 1462                                    FALSE  .1462
## 1468                                    FALSE  .1468
## 23048                                    TRUE .23048
```

![](USCensus_Earnings_files/figure-html/predict.data.new-14.png) 

```r
tmp_replay_lst <- replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.new.prediction")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1 
## 6.0000 	 6 	 0 0 1 2
```

![](USCensus_Earnings_files/figure-html/predict.data.new-15.png) 

```r
print(ggplot.petrinet(tmp_replay_lst[["pn"]]) + coord_flip())
```

![](USCensus_Earnings_files/figure-html/predict.data.new-16.png) 

Null Hypothesis ($\sf{H_{0}}$): mpg is not impacted by am_fctr.  
The variance by am_fctr appears to be independent. 
#```{r q1, cache=FALSE}
# print(t.test(subset(cars_df, am_fctr == "automatic")$mpg, 
#              subset(cars_df, am_fctr == "manual")$mpg, 
#              var.equal=FALSE)$conf)
#```
We reject the null hypothesis i.e. we have evidence to conclude that am_fctr impacts mpg (95% confidence). Manual transmission is better for miles per gallon versus automatic transmission.


```
##                   chunk_label chunk_step_major chunk_step_minor elapsed
## 10                 fit.models                5                1  98.841
## 11      fit.data.training.all                6                0 110.406
## 13           predict.data.new                7                0 127.038
## 12      fit.data.training.all                6                1 117.134
## 6            extract_features                3                0   6.361
## 2                cleanse_data                2                0   1.654
## 7             select_features                4                0   7.735
## 4         manage_missing_data                2                2   2.860
## 5          encode_retype_data                2                3   3.141
## 8  remove_correlated_features                4                1   7.936
## 9                  fit.models                5                0   7.989
## 3       inspectORexplore.data                2                1   1.692
## 1                 import_data                1                0   0.002
##    elapsed_diff
## 10       90.852
## 11       11.565
## 13        9.904
## 12        6.728
## 6         3.220
## 2         1.652
## 7         1.374
## 4         1.168
## 5         0.281
## 8         0.201
## 9         0.053
## 3         0.038
## 1         0.000
```

```
## [1] "Total Elapsed Time: 127.038 secs"
```

![](USCensus_Earnings_files/figure-html/print_sessionInfo-1.png) 

```
## R version 3.1.3 (2015-03-09)
## Platform: x86_64-apple-darwin13.4.0 (64-bit)
## Running under: OS X 10.10.3 (Yosemite)
## 
## locale:
## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
## 
## attached base packages:
## [1] tcltk     grid      stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] randomForest_4.6-10 rpart.plot_1.5.2    rpart_4.1-9        
##  [4] ROCR_1.0-7          gplots_2.16.0       caret_6.0-41       
##  [7] lattice_0.20-31     sqldf_0.4-10        RSQLite_1.0.0      
## [10] DBI_0.3.1           gsubfn_0.6-6        proto_0.3-10       
## [13] reshape2_1.4.1      plyr_1.8.1          caTools_1.17.1     
## [16] doBy_4.5-13         survival_2.38-1     ggplot2_1.0.1      
## 
## loaded via a namespace (and not attached):
##  [1] bitops_1.0-6        BradleyTerry2_1.0-6 brglm_0.5-9        
##  [4] car_2.0-25          chron_2.3-45        class_7.3-12       
##  [7] codetools_0.2-11    colorspace_1.2-6    compiler_3.1.3     
## [10] digest_0.6.8        e1071_1.6-4         evaluate_0.5.5     
## [13] foreach_1.4.2       formatR_1.1         gdata_2.13.3       
## [16] gtable_0.1.2        gtools_3.4.1        htmltools_0.2.6    
## [19] iterators_1.0.7     KernSmooth_2.23-14  knitr_1.9          
## [22] labeling_0.3        lme4_1.1-7          MASS_7.3-40        
## [25] Matrix_1.2-0        mgcv_1.8-6          minqa_1.2.4        
## [28] munsell_0.4.2       nlme_3.1-120        nloptr_1.0.4       
## [31] nnet_7.3-9          parallel_3.1.3      pbkrtest_0.4-2     
## [34] quantreg_5.11       RColorBrewer_1.1-2  Rcpp_0.11.5        
## [37] rmarkdown_0.5.1     scales_0.2.4        SparseM_1.6        
## [40] splines_3.1.3       stringr_0.6.2       tools_3.1.3        
## [43] yaml_2.1.13
```
