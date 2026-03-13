import unittest
from types import SimpleNamespace

import torch

from model import get_client_class, get_model_class, get_server_class


class GCNRegistrationTests(unittest.TestCase):
    def test_gcn_classes_are_registered(self):
        self.assertIsNotNone(get_model_class('gcn'))
        self.assertIsNotNone(get_model_class('gcn_mlp'))
        self.assertIsNotNone(get_server_class('gcn'))
        self.assertIsNotNone(get_client_class('gcn'))

    def test_gcn_forward_matches_expected_shape(self):
        gcn_cls = get_model_class('gcn')
        args = SimpleNamespace(device='cpu', embedding_size=8)
        model = gcn_cls(args, 8, 8, 8, num_layers=2, dropout=0.0)

        x = torch.randn(5, 8)
        output, intermediate_embedding, ls, lm = model(x)

        self.assertEqual(output.shape, (5, 8))
        self.assertEqual(len(intermediate_embedding), 1)
        self.assertEqual(intermediate_embedding[0].shape, (8,))
        self.assertEqual(ls, 0)
        self.assertEqual(lm, 0)


if __name__ == '__main__':
    unittest.main()
