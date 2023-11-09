from abc import ABC, abstractmethod


class Detection(ABC):
    @abstractmethod
    def run_strategies(self, dd):
        pass

    @abstractmethod
    def generate_features(self, d):
        pass

    @abstractmethod
    def build_clusters(self, d):
        pass

    @abstractmethod
    def sample_tuple(self, d):
        pass

    @abstractmethod
    def propagate_labels(self, d):
        pass

    @abstractmethod
    def predict_labels(self, d):
        pass

    @abstractmethod
    def run(self, dd):
        pass
