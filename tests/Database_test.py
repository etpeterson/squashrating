#import unittest

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_raises, assert_array_less, run_module_suite,
                           assert_warns, dec)

from Database import DBQuery, ratingDB


#class MyTestCase(unittest.TestCase):
#    def test_something(self):
#        self.assertEqual(True, False)



if __name__ == '__main__':
    run_module_suite()
    #unittest.main()
