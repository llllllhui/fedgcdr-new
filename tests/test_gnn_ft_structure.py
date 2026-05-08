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


def get_class_method_node(path: Path, class_name: str, method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return item
    raise AssertionError(f"method {class_name}.{method_name} not found in {path}")


class GNNFTStructureTests(unittest.TestCase):
    def test_base_server_declares_ft_stage(self):
        methods = get_class_method_names(ROOT / "model" / "base_party.py", "BaseServer")
        self.assertIn("ft_stage", methods)

    def test_base_server_ft_stage_delegates_to_kt_stage_without_transfer(self):
        method = get_class_method_node(ROOT / "model" / "base_party.py", "BaseServer", "ft_stage")
        non_docstring_body = [
            stmt for stmt in method.body
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str))
        ]

        self.assertEqual(len(non_docstring_body), 1)
        self.assertIsInstance(non_docstring_body[0], ast.Return)
        call = non_docstring_body[0].value
        self.assertIsInstance(call, ast.Call)
        self.assertIsInstance(call.func, ast.Attribute)
        self.assertEqual(call.func.attr, "kt_stage")
        self.assertEqual(len(call.args), 0)
        self.assertEqual(len(call.keywords), 1)
        self.assertEqual(call.keywords[0].arg, "tf_flag")
        self.assertIs(call.keywords[0].value.value, False)

    def test_lightgcn_server_implements_ft_stage(self):
        methods = get_class_method_names(ROOT / "model" / "lightgcn" / "party.py", "Server")
        self.assertIn("ft_stage", methods)


if __name__ == "__main__":
    unittest.main()
