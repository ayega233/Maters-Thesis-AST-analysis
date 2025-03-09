import ast
import difflib
import git #pip install gitpython
import astor 
import re
from pathlib import Path
import pandas as pd

class FeatureExtractor:

    def __init__(self, repo_path,repo,data_type):
        self.repo_path = f"{repo_path}/{repo}"
        self.repo = repo
        self.data_type = data_type

    def file_exists_in_commit(self, repo, commit_hash, file_path):       
        try:
            repo.git.ls_tree(commit_hash, file_path)
            return True 
        except git.exc.GitCommandError:
            return False  
    def get_file_versions(self, commit_hash, file_path):
   
        repo = git.Repo(self.repo_path)        
        commit = repo.commit(commit_hash)
        
        parent_commit = commit.parents[0] if commit.parents else None
        
        if parent_commit:
            
            current_version = None            
            try:
                current_version = repo.git.show(f"{commit_hash}:{file_path}")
            except git.exc.GitCommandError:
                current_version = None 
               
            previous_version = None           
            try:
                previous_version = repo.git.show(f"{parent_commit.hexsha}:{file_path}")
            except BaseException  as e:
                    print(f"error has occured for file_path {file_path} {e}")
            
            return previous_version, current_version
        else:
            raise ValueError("The given commit does not have a parent commit (i.e., it's the first commit).")

    def normalize_line(self,line):        
        return line.strip()

    def calculate_similarity(self,line1, line2):        
        return difflib.SequenceMatcher(None, line1, line2).ratio()
    
    def parse_code_diff(self,prev_code, curr_code, similarity_threshold=0.5):
       
        if prev_code is None:            
            curr_lines = curr_code.splitlines()
            return set(range(1, len(curr_lines) + 1)), set(), set()

        prev_lines = prev_code.splitlines()
        curr_lines = curr_code.splitlines()

        diff = list(difflib.unified_diff(prev_lines, curr_lines, lineterm='', fromfile='prev.py', tofile='curr.py'))

        added_lines = set()
        removed_lines = set()
        modified_lines = set()

        
        prev_line_num = 0
        curr_line_num = 0

        
        removed_content = {}
        added_content = {}

        for line in diff:
            
            if line.startswith('---') or line.startswith('+++'):
                continue
            hunk_header = re.match(r"^@@ -(\d+),?\d* \+(\d+),?\d* @@", line)
            if hunk_header:
                prev_line_num = int(hunk_header.group(1)) - 1
                curr_line_num = int(hunk_header.group(2)) - 1
                continue

            
            if line.startswith(' '):
               
                prev_line_num += 1
                curr_line_num += 1
            elif line.startswith('-'):
                
                prev_line_num += 1
                removed_lines.add(prev_line_num)
                removed_content[prev_line_num] = self.normalize_line(line[1:])
            elif line.startswith('+'):
                
                curr_line_num += 1
                added_lines.add(curr_line_num)
                added_content[curr_line_num] = self.normalize_line(line[1:])

        
        to_remove_from_removed = set()
        to_remove_from_added = set()

        for add_lineno, add_content in added_content.items():
            for rem_lineno, rem_content in removed_content.items():
                if rem_content == add_content:                    
                    removed_lines.discard(rem_lineno)
                    added_lines.discard(add_lineno)
                    to_remove_from_removed.add(rem_lineno)
                    to_remove_from_added.add(add_lineno)
                    break
       
        for line_num in to_remove_from_removed:
            removed_content.pop(line_num, None)
        for line_num in to_remove_from_added:
            added_content.pop(line_num, None)

        for add_lineno, add_content in list(added_content.items()):
            for rem_lineno, rem_content in list(removed_content.items()):               
                similarity = self.calculate_similarity(rem_content, add_content)
                if similarity_threshold < similarity < 1.0:  
                    modified_lines.add((rem_lineno, add_lineno))
                    removed_lines.discard(rem_lineno)
                    added_lines.discard(add_lineno)
                    removed_content.pop(rem_lineno, None)
                    added_content.pop(add_lineno, None)
                    break

        return added_lines, removed_lines, modified_lines
    

    def compare_ast_trees(self,prev_ast, curr_ast, added_lines, removed_lines):
        differences = []
        
        def compare_nodes(prev_node, curr_node, path=""):
            if type(prev_node) != type(curr_node):
                differences.append(f"Different node types at {path}: {type(prev_node)} vs {type(curr_node)}")
                return
            
            if isinstance(prev_node, ast.AST) and isinstance(curr_node, ast.AST):
                for field in prev_node._fields:
                    prev_val = getattr(prev_node, field)
                    curr_val = getattr(curr_node, field)
                    if isinstance(prev_val, list) and isinstance(curr_val, list):
                        for i, (p, c) in enumerate(zip(prev_val, curr_val)):
                            compare_nodes(p, c, path + f".{field}[{i}]")
                    else:
                        if prev_val != curr_val:
                            differences.append(f"Difference at {path}.{field}: {prev_val} vs {curr_val}")
        compare_nodes(prev_ast, curr_ast)
        return differences
    
    def parse_code(self,code):
        try:
            return ast.parse(code)    
        except SyntaxError as e:
            return None
    
    def extract_ast_features(self,code):       
        tree = ast.parse(code)        
        features = {
            "function_definitions": [],
            "class_definitions": [],
            "variable_assignments": [],
            "method_calls": []
        }        
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                features["function_definitions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                features["class_definitions"].append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        features["variable_assignments"].append(target.id)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    features["method_calls"].append(node.func.id)
        
        return features

    def compare_ast_class_features(self,prev_code, curr_code):     
        
        if curr_code is None:   
            #print("No Current code available. Treating all features are deleted.")
            prev_features = self.extract_ast_features(prev_code)
            return {
                "added_functions": 0,
                "removed_functions": len(prev_features["function_definitions"]),
                "added_classes": 0,
                "removed_classes": len(prev_features["class_definitions"]),
                "added_variables": 0,
                "removed_variables":len(prev_features["variable_assignments"]),
                "added_method_calls": 0,
                "removed_method_calls":len(prev_features["method_calls"]),
            }
        
        if prev_code is None:
            #print("No previous code available. Treating all features as newly added.")
            curr_features = self.extract_ast_features(curr_code)
            return {
                "added_functions": len(curr_features["function_definitions"]),
                "removed_functions": 0,
                "added_classes": len(curr_features["class_definitions"]),
                "removed_classes": 0,
                "added_variables": len(curr_features["variable_assignments"]),
                "removed_variables": 0,
                "added_method_calls": len(curr_features["method_calls"]),
                "removed_method_calls": 0
            }

        prev_features = self.extract_ast_features(prev_code)
        curr_features = self.extract_ast_features(curr_code)
        
        diff = {
            "added_functions": len(list(set(curr_features["function_definitions"]) - set(prev_features["function_definitions"]))),
            "removed_functions": len(list(set(prev_features["function_definitions"]) - set(curr_features["function_definitions"]))),
            "added_classes": len(list(set(curr_features["class_definitions"]) - set(prev_features["class_definitions"]))),
            "removed_classes": len(list(set(prev_features["class_definitions"]) - set(curr_features["class_definitions"]))),
            "added_variables": len(list(set(curr_features["variable_assignments"]) - set(prev_features["variable_assignments"]))),
            "removed_variables": len(list(set(prev_features["variable_assignments"]) - set(curr_features["variable_assignments"]))),
            "added_method_calls": len(list(set(curr_features["method_calls"]) - set(prev_features["method_calls"]))),
            "removed_method_calls": len(list(set(prev_features["method_calls"]) - set(curr_features["method_calls"])))
        }
        
        return diff

    def extract_features_from_diff(self,prev_code, curr_code, defect_lines):

        class_features = self.compare_ast_class_features(prev_code, curr_code)

        features = {
            "num_changes": 0,
            "added_lines": 0,
            "removed_lines": 0,
            "modified_lines": 0,
            "if_condition_changed": 0,
            "loop_range_changed": 0,
            "method_signature_changed": 0,
            "variables_renamed": 0,
            "is_defective": len(defect_lines)>2,
        }
        features.update(class_features)

        
        added_lines, removed_lines, modified_lines = self.parse_code_diff(prev_code, curr_code)
        
        if prev_code is None:
            added_lines = curr_code.splitlines()      
            features["num_changes"] = len(added_lines)
            features["added_lines"] = len(added_lines)
            return features          
        
        prev_ast = self.parse_code(prev_code)
        curr_ast = self.parse_code(curr_code)

        differences = self.compare_ast_trees(prev_ast, curr_ast, added_lines, removed_lines)
        features["num_changes"] = len(differences)
        
        
        def are_nodes_equivalent(node1, node2):
            
            if type(node1) != type(node2):
                return False
           
            if isinstance(node1, ast.Name) and isinstance(node2, ast.Name):
                return node1.id == node2.id

            
            for field, value1 in ast.iter_fields(node1):
                value2 = getattr(node2, field, None)
                #print(f"valuex {value1}{value2}")
                if isinstance(value1, list):
                    if len(value1) != len(value2):
                        return False
                    for item1, item2 in zip(value1, value2):
                        if isinstance(item1, ast.AST):
                            if not are_nodes_equivalent(item1, item2):
                                return False
                        elif item1 != item2:
                            return False
                elif isinstance(value1, ast.AST):
                    if not are_nodes_equivalent(value1, value2):
                        return False
                elif value1 != value2:
                    return False

            return True
        
        def normalize_ast(node):            
            if isinstance(node, ast.Name):                
                node.ctx = ast.Load()  
                node.id = "VAR"  
            
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            normalize_ast(item)
                elif isinstance(value, ast.AST):
                    normalize_ast(value)

        def check_if_condition_changes(prev_ast, curr_ast,defect_lines):
            
            normalize_ast(prev_ast)
            normalize_ast(curr_ast)

            prev_if_nodes = [node for node in ast.walk(prev_ast) if isinstance(node, ast.If)]
            curr_if_nodes = [node for node in ast.walk(curr_ast) if isinstance(node, ast.If)]

            
            prev_condition_line_numbers = {node.lineno: node for node in prev_if_nodes}
            curr_condition_line_numbers = {node.lineno: node for node in curr_if_nodes}

           
            for lineno in prev_condition_line_numbers:
                if lineno in curr_condition_line_numbers:
                    prev_node = prev_condition_line_numbers[lineno]
                    curr_node = curr_condition_line_numbers[lineno]

                    
                    if not are_nodes_equivalent(prev_node.test, curr_node.test):
                        #print(f"Change detected in 'if' condition at line {lineno}")
                        return True
            
            for lineno in prev_condition_line_numbers:
                if lineno not in curr_condition_line_numbers:
                    #print(f"Removed 'if' condition at line {lineno}")
                    return True

            for lineno in curr_condition_line_numbers:
                if lineno not in prev_condition_line_numbers:
                    #print(f"Added 'if' condition at line {lineno}")
                    return True

            return False  
    

        def check_loop_range_changes(prev_ast, curr_ast,defect_lines):           
            normalize_ast(prev_ast)
            normalize_ast(curr_ast)

            prev_for_nodes = [node for node in ast.walk(prev_ast) if isinstance(node, ast.For)]
            curr_for_nodes = [node for node in ast.walk(curr_ast) if isinstance(node, ast.For)]

            prev_for_line_numbers = {node.lineno: node for node in prev_for_nodes}
            curr_for_line_numbers = {node.lineno: node for node in curr_for_nodes}

            for lineno in prev_for_line_numbers:
                if lineno in curr_for_line_numbers:
                    prev_node = prev_for_line_numbers[lineno]
                    curr_node = curr_for_line_numbers[lineno]

                    if not are_nodes_equivalent(prev_node.iter, curr_node.iter):
                        #print(f"Change detected in 'for' loop range at line {lineno}")
                        return True
           
            for lineno in prev_for_line_numbers:
                if lineno not in curr_for_line_numbers:
                    #print(f"Removed 'for' loop at line {lineno}")
                    return True

            for lineno in curr_for_line_numbers:
                if lineno not in prev_for_line_numbers:
                    #print(f"Added 'for' loop at line {lineno}")
                    return True

            return False


        def check_method_signature_changes(prev_ast, curr_ast,defect_lines):
            
            normalize_ast(prev_ast)
            normalize_ast(curr_ast)

            prev_function_nodes = [node for node in ast.walk(prev_ast) if isinstance(node, ast.FunctionDef)]
            curr_function_nodes = [node for node in ast.walk(curr_ast) if isinstance(node, ast.FunctionDef)]

            prev_function_line_numbers = {node.lineno: node for node in prev_function_nodes}
            curr_function_line_numbers = {node.lineno: node for node in curr_function_nodes}

            for lineno in prev_function_line_numbers:
                if lineno in curr_function_line_numbers:
                    prev_node = prev_function_line_numbers[lineno]
                    curr_node = curr_function_line_numbers[lineno]

                    if prev_node.name != curr_node.name or len(prev_node.args.args) != len(curr_node.args.args):
                        #print(f"Change detected in method signature at line {lineno}")
                        return True
            
            for lineno in prev_function_line_numbers:
                if lineno not in curr_function_line_numbers:
                    #print(f"Removed method at line {lineno}")
                    return True

            for lineno in curr_function_line_numbers:
                if lineno not in prev_function_line_numbers:
                    #print(f"Added method at line {lineno}")
                    return True

            return False


        def check_variable_renaming(prev_ast, curr_ast,defect_lines):
           
            normalize_ast(prev_ast)
            normalize_ast(curr_ast)

            prev_name_nodes = [node for node in ast.walk(prev_ast) if isinstance(node, ast.Name)]
            curr_name_nodes = [node for node in ast.walk(curr_ast) if isinstance(node, ast.Name)]

            prev_name_ids = {node.id for node in prev_name_nodes}
            curr_name_ids = {node.id for node in curr_name_nodes}

            if prev_name_ids != curr_name_ids:
                #print(f"Variable renaming detected")
                return True

            return False
       
        features["if_condition_changed"] = 1 if check_if_condition_changes(prev_ast, curr_ast, defect_lines) else 0
        features["loop_range_changed"] = 1 if check_loop_range_changes(prev_ast, curr_ast, defect_lines) else 0
        features["method_signature_changed"] = 1 if check_method_signature_changes(prev_ast, curr_ast, defect_lines) else 0
        features["variables_renamed"] = 1 if check_variable_renaming(prev_ast, curr_ast, defect_lines) else 0

        features["added_lines"] = len(set(list(added_lines)))
        features["modified_lines"] = len(set(list(modified_lines))) 
        features["removed_lines"] = len(set(list(removed_lines)))


        return features
    
    def extract(self):
        df = pd.read_excel(f"{self.data_type}/{self.data_type}_{self.repo}.xlsx")
        features_data = []         
        for index, row in df.iterrows():    
            print(f"Processing row {self.repo} {index} {row['commit']}")
            path = row['filepath'].replace("\\", "/")
            file_content = self.get_file_versions(row['commit'],path)
            defect_lines = row['lines']
            
            
            try:
                features = self.extract_features_from_diff(file_content[0], file_content[1], defect_lines)
                features_data.append(features)

            except BaseException  as e:
                print(f"error has occured for row {index} {e}")
            
        df_f = pd.DataFrame(features_data)
        df_f.to_csv(f"{self.data_type}/{self.repo}_feature.csv", index=False)           


if __name__ == "__main__":
    #repo="localstack"
    repos = ["django"]
    for repo in repos:  
        processor = FeatureExtractor("D:/Master thesis/repos"  ,repo,"train")
        processor.extract()
        processor = FeatureExtractor("D:/Master thesis/repos"  ,repo,"test")
        processor.extract()
        processor = FeatureExtractor("D:/Master thesis/repos"  ,repo,"val")
        processor.extract()
    