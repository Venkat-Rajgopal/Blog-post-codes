rm(list = ls())
library(caret)
# read data
df <- read.table("Data/breast-cancer-wisconsin.txt", header = FALSE,  sep = ",")

# update column names
colnames(df) <- c("sample_no", "clump_thickness", "cell_size", "cell_shape", 
                "marginal_adhesion","single_epithelial_cell_size", 
                "bare_nuclei", "bland_chromatin", "normal_nucleoli", 
                "mitosis", "classes")

# update the classes
df$classes <- ifelse(df$classes == "2", "benign", ifelse(df$classes == "4", "malignant", NA))
df$classes <- as.factor(df$classes)

ggplot(data = df, aes(x=df$cell_size, y=df$cell_shape, color=classes))+
    geom_point(size=2)+
    labs(x='Cell Size', y='Cell Shape')+
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())


# split data 
dfindex <- createDataPartition(df$classes, p = 0.7, list = FALSE)
train_data <- df[dfindex, ]
test_data  <- df[-dfindex, ]

table(train_data$classes)

# trnctrl <- trainControl(method = "repeatedcv", number = 10, 
#                      repeats = 10, verboseIter = FALSE,
#                      sampling = "smote")
# model.rf <- caret::train(classes ~ ., data = train_data, method = "rf", trControl = trnctrl)

library(DMwR)
dat <-  SMOTE(classes~., data = train_data[c(2:11)], perc.over = 100, k = 3)
table(train_data$classes)
table(dat$classes)

# Error in knearest(P_set, P_set, K) : object 'knD' not found
# install.packages("FNN")


# plot data
library(reshape2)
