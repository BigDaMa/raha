from abc import ABC, abstractmethod


class Correction(ABC):

    @abstractmethod
    def initialize_dataset(self, d):
        pass

    @abstractmethod
    def initialize_models(self, d):
        pass

    @abstractmethod
    def sample_tuple(self, d):
        pass

    @abstractmethod
    def update_models(self, d):
        pass

    @abstractmethod
    def predict_corrections(self, d):
        pass

    @abstractmethod
    def run(self, d):
        pass
