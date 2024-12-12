import unittest

from autoop.tests.test_database import TestDatabase
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_pipeline import TestPipeline
from autoop.tests.test_storage import TestStorage

list_with_tests = [TestDatabase, TestFeatures, TestPipeline, TestStorage]

if __name__ == '__main__':
    unittest.main()
