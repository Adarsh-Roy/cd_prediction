

import os

def create_context_file(root_dir, output_file):
    """
    Creates a text file with the contents of all files in the given directory,
    formatted for an LLM.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(file_path, root_dir)
                    outfile.write(f'<file path="{relative_path}">\n')
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}")
                    outfile.write('\n</file>\n\n')

if __name__ == '__main__':
    create_context_file('src', 'src_context.txt')

