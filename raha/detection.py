from abc import ABC, abstractmethod


class Detection(ABC):
    @abstractmethod
    def run_strategies(self, dd):
        """
        This method runs (all or the promising) strategies.
        """
        pass

    @abstractmethod
    def generate_features(self, d):
        """
        This method generates features.
        """
        pass

    @abstractmethod
    def build_clusters(self, d):
        """
        This method builds clusters.
        """
        pass

    @abstractmethod
    def sample_tuple(self, d):
        """
        This method samples a tuple.
        """
        pass

    @abstractmethod
    def propagate_labels(self, d):
        """
        This method propagates labels.
        """
        pass

    @abstractmethod
    def predict_labels(self, d):
        """
        This method predicts the label of data cells.
        """
        pass

    @abstractmethod
    def run(self, dd):
        """
        This method runs Raha on an input dataset to detection data errors.
        """
        pass
