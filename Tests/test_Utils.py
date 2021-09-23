import unittest
import Utils.ConfigurationLoader as cl


class ConfigurationTest(unittest.TestCase):
    def test_load_model_config(self):
        model_config = cl.load_model_config()
        self.assertNotEqual(model_config, None)
        print(model_config)

    def test_load_wrong_model_config(self):
        env = cl.Environment(model='terrible_model')
        self.assertRaises(KeyError, lambda: cl.load_model_config(env=env))


if __name__ == '__main__':
    unittest.main()
