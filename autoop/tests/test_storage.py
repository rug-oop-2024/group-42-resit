
import os
import random
import tempfile
import unittest
from pathlib import Path

from autoop.core.storage import LocalStorage, NotFoundError


class TestStorage(unittest.TestCase):

    def setUp(self):
        temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(temp_dir)

    def test_init(self):
        self.assertIsInstance(self.storage, LocalStorage)

    def test_store(self):
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = "test/path"
        self.storage.save(test_bytes, key)
        self.assertEqual(self.storage.load(key), test_bytes)
        otherkey = "test/otherpath"
        # should not be the same
        try:
            self.storage.load(otherkey)
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_delete(self):
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = "test/path"
        self.storage.save(test_bytes, key)
        self.storage.delete(key)
        try:
            self.assertIsNone(self.storage.load(key))
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_list(self):
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        random_keys = [f"test/{random.randint(0, 100)}" for _ in range(10)]
        for key in random_keys:
            self.storage.save(test_bytes, key)
        keys = self.storage.list("test")
        keypaths = [Path(os.path.normpath(key)) for key in keys]
        data = [[path.parent.name, path.name] for path in keypaths]
        keys = ["/".join(key) for key in data]
        # keys = ["/".join(key.split("/")[-2:]) for key in keys]
        self.assertEqual(set(keys), set(random_keys))
