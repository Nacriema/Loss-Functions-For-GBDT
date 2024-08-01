# run_all_tests.py
import unittest

# Import test modules
from tests import test_focal_loss

# Load test cases
loss_function_tests = unittest.TestLoader().loadTestsFromModule(test_focal_loss)

# Create a test suite combining all test cases
all_tests = unittest.TestSuite([loss_function_tests,
                                # Others test here
                                ])

# Run the test suite
unittest.TextTestRunner(verbosity=2).run(all_tests)
