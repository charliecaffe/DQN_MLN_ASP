with open('requirements2.txt', 'r') as longfile, open('requirements_part.txt', 'r') as shortfile, open('requirements_conda.txt', 'w') as newfile:
    long_file = set(longfile.readlines())
    short_file = set(shortfile.readlines())
    newfile.writelines(long_file-short_file)