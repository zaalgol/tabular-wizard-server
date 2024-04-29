

def apply_conditional_smote(classifier):
    class_counts = self.y_train.value_counts()

    smallest_class = class_counts.min()
    largest_class = class_counts.max()
    ratio = smallest_class / largest_class

    # Define a threshold below which we consider the dataset imbalanced
    # This threshold can be adjusted based on specific needs
    imbalance_threshold = 0.5  # Example threshold

    # If the ratio is below the threshold, apply SMOTE
    if ratio < imbalance_threshold:
        random_pver_sampler = RandomOverSampler(random_state=0)
        self.X_train, self.y_train = random_pver_sampler.fit_resample(self.X_train, self.y_train)
    else:
        print("The dataset is considered balanced. Skipping SMOTE.")
        
def apply_conditional_smote(classifier):
    random_pver_sampler = RandomOverSampler(random_state=0)
    self.X_train, self.y_train = random_pver_sampler.fit_resample(self.X_train, self.y_train)