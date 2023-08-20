library(dplyr)
library(arules)
library(arulesViz)
library(ggplot2)
library(cluster)
library(ggplot2)
library(corrplot)
library(xgboost)
library(e1071)
library(randomForest)
library(class)
library(rpart)
library(readr)


# Load the dataset
data <- read_csv("data.csv")

# Drop unnecessary columns
data <- data %>% select(-c(Patient_Number, Pregnancy))

# Handle missing values
df <- na.omit(data)

# Handle missing values in 'alcohol_consumption_per_day' column
df$alcohol_consumption_per_day <- as.numeric(df$alcohol_consumption_per_day)

# Convert non-integer columns to integers
df$Physical_activity <- as.integer(df$Physical_activity)
df$salt_content_in_the_diet <- as.integer(df$salt_content_in_the_diet)

write.csv(df, "processed_data.csv", row.names = FALSE)

# Create a histogram for Age
ggplot(data, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black") +
  labs(title = "Distribution of Age", x = "Age", y = "Frequency")

# Written by Taranjith
ggplot(data, aes(x = Blood_Pressure_Abnormality)) +
  geom_histogram(binwidth = 1, fill = "green", color = "black") +
  labs(title = "Blood pressure abnormality", x = "Age", y = "Frequency")


#written by Taranjith
# Create a box plot for Level of Hemoglobin by Gender
ggplot(data, aes(x = Sex, y = Level_of_Hemoglobin, fill = Sex, group = Sex)) +
  geom_boxplot() +
  labs(title = "Level of Hemoglobin by Gender", x = "Gender", y = "Level of Hemoglobin")

#written by Sree Nandini
# Create a bigger scatter plot with trend lines for Level of Hemoglobin vs Age
ggplot(data, aes(x = Age, y = Level_of_Hemoglobin)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(title = "Scatter Plot: Level of Hemoglobin vs Age", x = "Age", y = "Level of Hemoglobin") +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12)) +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"),
        panel.grid.major = element_line(colour = "gray"),
        panel.grid.minor = element_blank())

#written by Sree Nandini
# Create a pie chart for Smoking distribution

smoking_data <- data %>%
  group_by(Smoking) %>%
  summarise(count = n())

ggplot(smoking_data, aes(x = "", y = count, fill = Smoking)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  labs(title = "Smoking Distribution", fill = "Smoking")

#written by Sree Nandini
# ASSOCIATION MINING
# Convert variables to appropriate types (e.g., binary or categorical)
df$Chronic_kidney_disease <- as.factor(df$Chronic_kidney_disease)

# Convert dataset to transactions
df_encoded <- as(df, "transactions")

# Specify the correct item label for the appearance parameter
frequent_itemsets <- apriori(df_encoded, parameter = list(support = 0.1), 
                             appearance = list(lhs = "Chronic_kidney_disease=1", default = "lhs"))

# Print the generated rules
print(frequent_itemsets)


#written by Sree Nandini
# CLUSTERING
numeric_columns <- c('Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'BMI', 'Physical_activity',
                     'salt_content_in_the_diet', 'alcohol_consumption_per_day', 'Level_of_Stress')

# Select only numeric columns and exclude missing values
scaled_data <- df[, numeric_columns]
scaled_data <- na.omit(scaled_data)

# Scale the numeric columns
scaled_data <- scale(scaled_data)

# Perform clustering on scaled data
kmeans_result <- kmeans(scaled_data, centers = 4, nstart = 25, iter.max = 10, algorithm = "Hartigan-Wong")

df$Cluster_Labels <- as.factor(kmeans_result$cluster)

# Plot the clusters
ggplot(data=df, aes(x=Age, y=Level_of_Hemoglobin, color=Cluster_Labels)) +
  geom_point() +
  labs(x = "Age", y = "Level of Hemoglobin", title = "K-means Clustering") +
  theme_minimal()


#written by Sree Nandini
# CLASSIFICATION

df <- df %>% select(-c(Level_of_Hemoglobin, Genetic_Pedigree_Coefficient, Age))

# Separate features and target variable
X <- df %>% select(-Chronic_kidney_disease)
y <- df$Chronic_kidney_disease

# Split the data into training and testing sets
set.seed(42)
train_indices <- sample(1:nrow(df), nrow(df)*0.8)
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

#written by Sree Nandini
# Logistic Regression Model
logreg <- glm(Chronic_kidney_disease ~ ., data = df, family = binomial(link = "logit"))
y_pred <- predict(logreg, newdata = X_test, type = "response")
lr_accuracy <- mean(ifelse(y_pred >= 0.5, 1, 0) == y_test)
print(paste("Logistic Regression Accuracy:", lr_accuracy))

#recall_lr <- true_positives / (true_positives + false_negatives)
#precision_lr <- true_positives / (true_positives + false_positives)
#print(paste("Logistic Regression Recall:", recall_lr))
#print(paste("Logistic Regression Precision:", precision_lr))




y_pred_train_lr <- predict(logreg, newdata = X_train, type = "response")
y_pred_test_lr <- predict(logreg, newdata = X_test, type = "response")

# Create scatter plots for Logistic Regression predictions
plot_predictions <- function(predictions, actual, title) {
  ggplot(data.frame(Predicted = predictions, Actual = actual), aes(x = Predicted, y = Actual)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = title, x = "Predicted Probability", y = "Actual Value")
}


# Random Forest Classifier Model
rf <- randomForest(Chronic_kidney_disease ~ ., data = df, ntree = 100)
y_pred <- predict(rf, newdata = X_test, type = "response")
rfc_accuracy <- mean(ifelse(y_pred == 0.5, 1, 0) == y_test)
print(paste("Random Forest Classifier Accuracy:", rfc_accuracy))

conf_matrix_rf <- table(y_pred, y_test)
recall_rf <- conf_matrix_rf[2, 2] / sum(conf_matrix_rf[2, ])
precision_rf <- conf_matrix_rf[2, 2] / sum(conf_matrix_rf[, 2])
print(paste("Random Forest Recall:", recall_rf))
print(paste("Random Forest Precision:", precision_rf))

y_pred_train_rf <- predict(rf, newdata = X_train, type = "response")
y_pred_test_rf <- predict(rf, newdata = X_test, type = "response")
# Calculate accuracy for training and testing sets
rfc_accuracy_train <- mean(ifelse(y_pred_train_rf == 0.5, 1, 0) == y_train)
rfc_accuracy_test <- mean(ifelse(y_pred_test_rf == 0.5, 1, 0) == y_test)
plot_rfc_training <- plot_predictions(y_pred_train_rf, y_train, "Random Forest Training Set Predictions")
plot_rfc_testing <- plot_predictions(y_pred_test_rf, y_test, "Random Forest Testing Set Predictions")

# Support Vector Machine model

svm <- svm(Chronic_kidney_disease ~ ., data = df, probability = TRUE)
y_pred <- predict(svm, newdata = X_test, probability = TRUE)
svm_accuracy <- mean(ifelse(y_pred == 0.5, 1, 0) == y_test)
print(paste("SVM Accuracy:", svm_accuracy))


conf_matrix_svm <- table(y_pred, y_test)
recall_svm <- conf_matrix_svm[2, 2] / sum(conf_matrix_svm[2, ])
precision_svm <- conf_matrix_svm[2, 2] / sum(conf_matrix_svm[, 2])
print(paste("SVM Recall:", recall_svm))
print(paste("SVM Precision:", precision_svm))


y_pred_train_svm <- predict(svm, newdata = X_train, probability = TRUE)
y_pred_test_svm <- predict(svm, newdata = X_test, probability = TRUE)
svm_accuracy_train <- mean(ifelse(y_pred_train_svm == 0.5, 1, 0) == y_train)
svm_accuracy_test <- mean(ifelse(y_pred_test_svm == 0.5, 1, 0) == y_test)

plot_svm_training <- plot_predictions(y_pred_train_svm, y_train, "SVM Training Set Predictions")
plot_svm_testing <- plot_predictions(y_pred_test_svm, y_test, "SVM Testing Set Predictions")

# Decision Tree Classifier Model
dt <- rpart(Chronic_kidney_disease ~ ., data = df, method = "class")
y_pred <- predict(dt, newdata = X_test, type = "class")
dtc_accuracy <- mean(y_pred == y_test)
print(paste("Accuracy:", dtc_accuracy))


conf_matrix_dt <- table(y_pred, y_test)
recall_dt <- conf_matrix_dt[2, 2] / sum(conf_matrix_dt[2, ])
precision_dt <- conf_matrix_dt[2, 2] / sum(conf_matrix_dt[, 2])
print(paste("Decision Tree Recall:", recall_dt))
print(paste("Decision Tree Precision:", precision_dt))


y_pred_train_dt <- predict(dt, newdata = X_train, type = "class")
y_pred_test_dt <- predict(dt, newdata = X_test, type = "class")
# Calculate accuracy for training and testing sets
dtc_accuracy_train <- mean(y_pred_train_dt == y_train)
dtc_accuracy_test <- mean(y_pred_test_dt == y_test)
plot_dt_training <- plot_predictions(as.numeric(y_pred_train_dt), as.numeric(y_train), "Decision Tree Training Set Predictions")
plot_dt_testing <- plot_predictions(as.numeric(y_pred_test_dt), as.numeric(y_test), "Decision Tree Testing Set Predictions")

# Naive Bayes Classifier Model
library(naivebayes)
yy_train <- as.factor(y_train)
nb <- naive_bayes(X_train, yy_train)
y_pred <- predict(nb, newdata = X_test)
nbc_accuracy <- mean(y_pred == y_test)
print(paste("Accuracy:", nbc_accuracy))

conf_matrix_nb <- table(y_pred, y_test)
recall_nb <- conf_matrix_nb[2, 2] / sum(conf_matrix_nb[2, ])
precision_nb <- conf_matrix_nb[2, 2] / sum(conf_matrix_nb[, 2])
print(paste("Naive Bayes Recall:", recall_nb))
print(paste("Naive Bayes Precision:", precision_nb))



y_pred_train_nb <- predict(nb, newdata = X_train)
y_pred_test_nb <- predict(nb, newdata = X_test)
nbc_accuracy_train <- mean(y_pred_train_nb == y_train)
nbc_accuracy_test <- mean(y_pred_test_nb == y_test)
plot_nb_training <- plot_predictions(as.numeric(y_pred_train_nb), as.numeric(y_train), "Naive Bayes Training Set Predictions")
plot_nb_testing <- plot_predictions(as.numeric(y_pred_test_nb), as.numeric(y_test), "Naive Bayes Testing Set Predictions")

# KNN Classifier Model
k <- 3  # Set the value of k (number of neighbors)
knn <- knn(train = X_train, test = X_test, cl = y_train, k = k)
knn_accuracy <- mean(knn == y_test)
print(paste("Accuracy:", knn_accuracy))

conf_matrix_knn <- table(knn, y_test)
recall_knn <- conf_matrix_knn[2, 2] / sum(conf_matrix_knn[2, ])
precision_knn <- conf_matrix_knn[2, 2] / sum(conf_matrix_knn[, 2])
print(paste("KNN Recall:", recall_knn))
print(paste("KNN Precision:", precision_knn))



plot_knn_testing <- plot_predictions(as.numeric(knn), as.numeric(y_test), "KNN Testing Set Predictions")

# List of classifier names
classifiers <- c('Logistic Regression', 'Random Forest', 'Support Vector Machines', 'Decision Trees', 'Naive Bayes', 'k-Nearest Neighbors')

# List of accuracy values for each classifier
accuracies <- c(lr_accuracy, rfc_accuracy, svm_accuracy, dtc_accuracy, nbc_accuracy, knn_accuracy)


library(ggplot2)

# Create a data frame with classifier names and accuracy values
df <- data.frame(Classifiers = classifiers, Accuracy = accuracies)

# Create the bar plot with rotated labels
ggplot(df, aes(x = Classifiers, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_text(aes(label = sprintf("%.2f", Accuracy)), vjust = -0.3, angle = 45) +
  labs(x = "Classifiers", y = "Accuracy", title = "Accuracy of Classification Models") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  coord_flip()

# Create a data frame with model names, metrics, and categories
model_names <- c( "Random Forest", "SVM", "Decision Tree", "Naive Bayes", "KNN")
categories <- rep(c("Recall", "Precision"), times = length(model_names))
values <- c(recall_rf, precision_rf,
            recall_svm, precision_svm,
            recall_dt, precision_dt,
            recall_nb, precision_nb,
            recall_knn, precision_knn)

data_df <- data.frame(Model = rep(model_names, each = 2), Category = categories, Value = values)

# Create the bar plot using ggplot2
p <- ggplot(data_df, aes(x = Model, y = Value, fill = Category)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_text(aes(label = round(Value, 3)), position = position_dodge(width = 0.9), vjust = -0.5) +  # Include scores
  labs(title = "Recall and Precision Comparison",
       y = "Value", x = "Models") +
  scale_fill_manual(values = c("Recall" = "blue", "Precision" = "green")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p)

# Plot Logistic Regression predictions for training set
plot_logreg_training <- plot_predictions(y_pred_train_lr, y_train, "Logistic Regression Training Set Predictions")
print(plot_logreg_training)

# Plot Logistic Regression predictions for testing set
plot_logreg_testing <- plot_predictions(y_pred_test_lr, y_test, "Logistic Regression Testing Set Predictions")
print(plot_logreg_testing)

print(plot_rfc_training)
print(plot_rfc_testing)


print(plot_svm_training)
print(plot_svm_testing)


print(plot_dt_training)
print(plot_dt_testing)


print(plot_nb_training)
print(plot_nb_testing)



print(plot_knn_testing)
