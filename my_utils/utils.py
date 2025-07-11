import numpy as np
import re
import textwrap

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()
    
def extract_to_hs(input_string: str):
    code_blocks = input_string.split("```python\n")[1:]

    try:
        parameter_ranges_block = "import numpy as np\n" + code_blocks[1].split("```")[0].strip()
        if any(keyword in parameter_ranges_block for keyword in ['inf', 'np.inf', 'None']):
            return None, None
        exec_globals = {}
        exec(parameter_ranges_block, exec_globals)
        parameter_ranges = exec_globals['parameter_ranges']
    except:
        return None, None

    function_block = code_blocks[0].split("```")[0].strip()

    paren_count = 0
    in_signature = False
    signature_start_index = None
    signature_end_index = None

    # Loop through the function block to find the start and end of the function signature
    for i, char in enumerate(function_block):
        if char == "d" and function_block[i:i + 3] == 'def':
            in_signature = True
            signature_start_index = i
        if in_signature:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if char == ':' and paren_count == 0:
                signature_end_index = i
                break

    if signature_start_index is not None and signature_end_index is not None:
        function_signature = function_block[signature_start_index:signature_end_index + 1]
        for param in parameter_ranges:
            pattern = rf"(\b{param}\b[^=]*=)[^,)]+"
            replacement = r"\1 {" + param + "}"
            function_signature = re.sub(pattern, replacement, function_signature, flags=re.DOTALL)
        function_block = function_block[:signature_start_index] + function_signature + function_block[
                                                                                       signature_end_index + 1:]

    return parameter_ranges, function_block

def extract_class_name_and_code(code_string: str):
    cleaned_code = code_string.encode().decode('unicode_escape')

    class_name_match = re.search(r'class\s+(\w+)\s*[:\(]', cleaned_code)
    class_name = class_name_match.group(1) if class_name_match else None

    cleaned_code = textwrap.dedent(cleaned_code)

    return class_name, cleaned_code