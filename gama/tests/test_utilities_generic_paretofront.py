""" Contains all unit tests for """
import unittest

from gama.utilities.generic.paretofront import ParetoFront


def paretofront_test_suite():
    test_cases = [ParetoFrontUnitTestCase]
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, test_cases))


class ParetoFrontUnitTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_pareto_initialization_empty(self):
        """ Test initialization of empty front. """
        pf = ParetoFront()

        self.assertEqual(list(pf), [])
        self.assertEqual(str(pf), '[]')
        self.assertEqual(repr(pf), 'ParetoFront([])')

    def test_pareto_initialization_pareto_front(self):
        """ Test initialization with a list which only has Pareto front elements. """
        list_ = [(1, 2, 3), (3, 2, 1), (0, 5, 0)]
        pf = ParetoFront(list_)

        self.assertEqual(list(pf), [(1, 2, 3), (3, 2, 1), (0, 5, 0)])
        self.assertEqual(str(pf), '[(1, 2, 3), (3, 2, 1), (0, 5, 0)]')
        self.assertEqual(repr(pf), 'ParetoFront([(1, 2, 3), (3, 2, 1), (0, 5, 0)])')

    def test_pareto_initialization_with_inferiors(self):
        """" Test initialization with a list containing elements that should not be in the Pareto front. """
        list_ = [(1, 2), (4, 3), (4, 5), (5, 4)]
        pf = ParetoFront(list_)

        self.assertEqual(list(pf), [(4, 5), (5, 4)])
        self.assertEqual(str(pf), '[(4, 5), (5, 4)]')
        self.assertEqual(repr(pf), 'ParetoFront([(4, 5), (5, 4)])')

    def test_pareto_initialization_with_duplicates(self):
        """ Test initialization with duplicate elements in list. """
        list_ = [(1, 2), (3, 1), (1, 2)]
        pf = ParetoFront(list_)

        self.assertEqual(list(pf), [(1, 2), (3, 1)])
        self.assertEqual(str(pf), '[(1, 2), (3, 1)]')
        self.assertEqual(repr(pf), 'ParetoFront([(1, 2), (3, 1)])')

    def test_pareto_update_unique(self):
        """ Test creating Pareto front by updating one by one. """
        list_ = [(1, 2, 3), (3, 2, 1), (0, 5, 0)]

        pf = ParetoFront()
        self.assertEqual(list(pf), [])

        for i in range(len(list_)):
            pf.update(list_[i])
            self.assertEqual(list(pf), list_[:i+1])
