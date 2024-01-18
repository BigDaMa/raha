from abc import ABC, abstractmethod


class Correction(ABC):

    @abstractmethod
    def initialize_dataset(self, d):
        """
        This method initializes the dataset.
        """
        pass

    @abstractmethod
    def initialize_models(self, d):
        """
        This method initializes the error corrector models.
        """
        pass

    @abstractmethod
    def generate_features(self, d, cells):
        """
        This method generates a feature vector for each pair of a data error and a potential correction.
        """
        pass

    @abstractmethod
    def sample_tuple(self, d):
        """
        This method samples a tuple.
        """
        pass

    @abstractmethod
    def update_models(self, d):
        """
        This method updates the error corrector models with a new labeled tuple.
        """
        pass

    @abstractmethod
    def predict_corrections(self, d):
        """
        This method predicts corrections for each data error.
        """
        pass

    @abstractmethod
    def run(self, d):
        """
        This method runs Baran on an input dataset to correct data errors.
        """
        pass
