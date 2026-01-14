class FeatureAwareModel:
    def __init__(self, model, feature_names, target_name=None):
        self.model = model
        self.feature_names = list(feature_names)
        self.target_name = target_name

    def predict(self, X):
        return self.model.predict(X[self.feature_names])

    def get_feature_names(self):
        return self.feature_names
