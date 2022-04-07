from abc import ABC, abstractmethod


class App(ABC):

    @abstractmethod
    def handle_key(self, key):
        pass