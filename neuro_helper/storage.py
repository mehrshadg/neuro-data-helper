from enum import Enum
import os
import glob
from abc import ABC, abstractmethod
from typing import List
from neuro_helper.rclone import RClone
import tempfile

ANYTHING = "*"


class StorageFile:
    local_path: str

    @property
    def name(self):
        return os.path.basename(self.local_path)

    @property
    def loadable_path(self):
        return self.local_path

    def __init__(self, local_path: str):
        self.local_path = local_path

    def __str__(self):
        return self.local_path

    def __lt__(self, other):
        return self.local_path < other.local_path


class RCloneStorageFile(StorageFile):
    remote_path: str
    engine: RClone
    downloaded = False

    @property
    def loadable_path(self):
        if not self.downloaded:
            self.download()
        return super().loadable_path

    def __init__(self, engine: RClone, remote_path: str):
        self.remote_path = remote_path
        super().__init__(os.path.join(tempfile.mkdtemp(), os.path.basename(remote_path)))
        self.engine = engine

    def download(self):
        if self.downloaded:
            return
        result = self.engine.copy(self.remote_path, self.local_path)
        self.downloaded = result["code"] == 0


class Storage(ABC):
    root: str
    names_pattern: str

    def __init__(self, root: str, patterns, **variables):
        self.root = root
        self.names_pattern = Storage.create_pattern(*patterns, **variables)

    @staticmethod
    def create_pattern(*patterns, **variables):
        if patterns is None:
            patterns = []
        if variables is None:
            variables = {}
        combined = ""
        for p in patterns:
            combined += str(p) if p not in variables else str(variables[p])
        return combined

    @abstractmethod
    def get_all(self, sort=True) -> List[StorageFile]:
        pass


class LocalStorage(Storage):
    def get_all(self, sort=True) -> List[StorageFile]:
        pattern = os.path.normpath(self.root + os.sep + self.names_pattern)
        files = list(map(lambda x: StorageFile(x), glob.glob(pattern)))
        if sort:
            files.sort()
        return files


class RCloneStorage(Storage):
    def __init__(self, remote: str, root: str, patterns, **variables):
        super().__init__(f"{remote}:{root}", patterns, **variables)
        self.engine = RClone()

    def get_all(self, sort=True) -> List[StorageFile]:
        files = list(map(lambda x: RCloneStorageFile(self.engine, os.path.join(self.root, x)),
                         filter(lambda x: x, self.engine.lsf(self.root, flags=["--include", self.names_pattern])["out"]
                                .decode("utf-8").split("\n"))))
        if sort:
            files.sort()
        return files
