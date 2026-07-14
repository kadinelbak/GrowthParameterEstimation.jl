import sys

def fix_datajl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the docstring for normalize_schema: we look for the opening triple quotes
    # that have the target phrase in the docstring.
    doc_start = -1
    doc_end = -1
    target_phrase = 'Normalize a DataFrame to match the expected timeseries schema'
    i = 0
    while i < len(lines):
        if lines[i].strip() == '\"\"\"':
            # Potential start of a docstring
            j = i + 1
            # Skip empty lines
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            # Check if we find the target phrase in the following lines (until the closing triple quotes)
            k = j
            found_phrase = False
            while k < len(lines) and lines[k].strip() != '\"\"\"':
                if target_phrase in lines[k]:
                    found_phrase = True
                    break
                k += 1
            if found_phrase:
                doc_start = i
                # Now find the closing triple quotes
                while k < len(lines) and lines[k].strip() != '\"\"\"':
                    k += 1
                if k < len(lines):
                    doc_end = k
                    break
                else:
                    doc_start = -1  # Not found, reset
            i = j
        else:
            i += 1

    if doc_start == -1 or doc_end == -1:
        print('Could not find the docstring for normalize_schema with the target phrase.')
        sys.exit(1)

    # Build new lines
    new_lines = []
    # Part 1: lines before the docstring
    new_lines.extend(lines[:doc_start])
    # Part 2: the opening triple quotes line
    new_lines.append(lines[doc_start])
    # Part 3: the inner docstring lines, but we remove the erroneous line
    for i in range(doc_start + 1, doc_end):
        if lines[i].strip() == 'normalize_schema(df::DataFrame; column_map=nothing, defaults=nothing)':
            # Skip this line
            continue
        new_lines.append(lines[i])
    # Part 4: the closing triple quotes line
    new_lines.append(lines[doc_end])
    # Part 5: the function definition line (without indentation)
    new_lines.append('function normalize_schema(df::DataFrame; column_map=nothing, defaults=nothing)\\n')
    # Part 6: lines after the docstring
    new_lines.extend(lines[doc_end + 1:])

    # Write the file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == '__main__':
    fix_datajl('src/data.jl')
