import unittest

import app

class TestApp(unittest.TestCase):
    def test_cast_series(self):
        """
        Test conversion string into int series
        """
        past_serie = 'a b c 1.3 2 d e f3.5 4 5'
        eval_value = '3'
        past_serie, eval_value = app.cast_series(past_serie, eval_value, ' ')
        self.assertEqual(past_serie, [1,2,3,4,5])
        self.assertEqual(eval_value, 3)

    def test_get_data(self):
        """
        Test number engineering
        """
        result = app.get_data([1,2,3,4,5])
        self.assertEqual(result.get('std_dev'), 1)
        self.assertEqual(result.get('mu'), 3)

    def test_get_error_probability(self):
        """
        Test poisson distribution estimation
        """
        serie = [1,2,3,4,5]
        eval_value = 4
        mu = 3
        threshold_error = [0.5, 0.8]
        result = app.get_error_probability(serie, eval_value, mu, threshold_error)
        self.assertEqual(result.get('percentage_event'), 16.8)
        self.assertEqual(result.get('err_percentage'), 25)
