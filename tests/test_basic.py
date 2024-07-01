# import unittest
# from app.app import app

# class BasicTestCase(unittest.TestCase):

#     def setUp(self):
#         # creates a test client
#         app.config['TESTING'] = True
#         self.app = app.test_client()
    
#     def test_home(self):
#         # sends HTTP GET request to the application
#         # on the specified path
#         result = self.app.get('/')
        
#         # assert the response data
#         self.assertEqual(result.data, b'Hello, World!')

# if __name__ == '__main__':
#     unittest.main()
