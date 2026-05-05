import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def get_class_method_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"class {class_name} not found in {path}")


class GNNFTStructureTests(unittest.TestCase):
    def test_base_server_declares_ft_stage(self):
        methods = get_class_method_names(ROOT / "model" / "base_party.py", "BaseServer")
        self.assertIn("ft_stage", methods)

    def test_lightgcn_server_implements_ft_stage(self):
        methods = get_class_method_names(ROOT / "model" / "lightgcn" / "party.py", "Server")
        self.assertIn("ft_stage", methods)


if __name__ == "__main__":
    unittest.main()
