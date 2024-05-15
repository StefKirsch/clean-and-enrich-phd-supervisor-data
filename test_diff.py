import difflib

def print_diff(string1, string2):
    diff = difflib.ndiff(string1, string2)
    # Convert the generator to a list of lines
    diff_list = list(diff)
    # Print each line of the differences
    for line in diff_list:
        print(line)

# Example usage
string1 = "hello world"
string2 = "hallo world"
print_diff(string1, string2)